"""Routing policies compared by the engine.

Every policy has the same signature -- given the request's blocks, the nodes, the current
time, and the cost params, return a :class:`~bench.sim.cost.Placement` (which node, whether
to transfer, and the regime that realizes). They differ only in the *choice set* and the
*information* they use, which is exactly what we want to isolate:

  * ``round_robin``   -- cyclic node, never transfer. The load-balancing baseline.
  * ``cache_local``   -- always the hot node, never transfer. Pure prefix-affinity; it
    always waits for the cache and never reacts to queue depth.
  * ``transfer_aware``-- argmin over (node, transfer) but scored on *approximate* state
    (queue depth rounded to whole requests), i.e. a realistic router without perfect info.
  * ``weighted_precise`` / ``weighted_approx`` -- the shape llm-d's EPP actually ships:
    a saturation filter, then per-scorer normalization to [0, 1], weight multiplication,
    argmax; never transfers.
    ``precise`` reads true node cache state (llm-d's precise-prefix-cache scorer), while
    ``approx`` scores from a router-side index of past routing decisions that never sees
    node-side evictions (llm-d's approximate prefix-affinity scorer). Their regret gap
    isolates *information* (stale cache beliefs) from *policy* (no transfer arm).
  * ``class_aware``   -- classifies each request from *observable* signals (message shape,
    sizes -- never the workload's ground-truth label) into tool-session / RAG / one-shot
    and applies a different strategy per class: transfer-aware for session turns, doc-prefix
    affinity with saturation spill for RAG, least-loaded for one-shots (never chase cache).
  * ``oracle``        -- argmin over (node, transfer) with perfect current state. The lower
    bound every other policy's regret is measured against.

Policies receive a :class:`RequestView` -- the request as a router sees it (block hashes
plus body-derived observables), not the generator's ``TurnRequest``.
"""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from bench.sim.blocks import matched_prefix
from bench.sim.cost import CostParams, Placement, best_placement, hot_node, predict
from bench.sim.node import Node

# GIE-style saturation filter threshold, in nominal requests of queue depth.
DEFAULT_SATURATION_DEPTH = 8


@dataclass(frozen=True)
class RequestView:
    """What the router is allowed to see about a request.

    Everything here is derivable from the request body the gateway already parses (llm-d's
    EPP sees the full JSON): the prefix block hashes plus cheap shape observables. There is
    deliberately no ground-truth workload class -- ``label`` models an optional upstream
    annotation (e.g. an Envoy-set header) and defaults to ``"unknown"``; v1 policies
    classify from the observables instead.
    """

    blocks: list[int]
    prompt_tokens: int = 0
    num_messages: int = 0
    has_tool_messages: bool = False
    last_message_tokens: int = 0
    label: str = field(default="unknown", compare=False)


# A RAG query embeds a retrieved document in its user message; a session's opening user
# message is a short question. This threshold (tokens) separates the two shapes.
_RAG_USER_TOKENS = 128


def classify(req: RequestView) -> str:
    """Infer the workload class from observables alone (never the ground-truth label).

    Tool-session turns past the first carry ``tool`` role messages. A lone user message
    with no system prompt is a one-shot. What remains is system + user: a huge user
    message means an embedded retrieved document (RAG); a short one is a session opener.
    """
    if req.has_tool_messages:
        return "tool"
    if req.num_messages <= 1:
        return "oneshot"
    return "rag" if req.last_message_tokens >= _RAG_USER_TOKENS else "tool"


Policy = Callable[[RequestView, list[Node], float, CostParams], Placement]


class RoundRobin:
    """Stateful cyclic assignment; ignores cache and load entirely."""

    def __init__(self) -> None:
        self._next = 0

    def __call__(
        self, req: RequestView, nodes: list[Node], now: float, params: CostParams
    ) -> Placement:
        node = nodes[self._next % len(nodes)]
        self._next += 1
        hot = hot_node(req.blocks, nodes)
        return predict(req.blocks, node, now, params, hot, transfer=False)


def cache_local(req: RequestView, nodes: list[Node], now: float, params: CostParams) -> Placement:
    hot = hot_node(req.blocks, nodes)
    return predict(req.blocks, hot, now, params, hot, transfer=False)


def _nominal_service(params: CostParams) -> float:
    """The coarse per-request service estimate a router thinks in (its 'queue depth' unit)."""
    return params.prefill_fixed + params.inter_token * 40


def _quantized_wait(node: Node, now: float, params: CostParams) -> float:
    """Approximate a node's queue delay as a whole number of nominal service times.

    A real router polls coarse load signals, not an exact busy-until clock; rounding the
    wait to nominal-request units models that imprecision so ``transfer_aware`` can be
    strictly weaker than the oracle.
    """
    nominal = _nominal_service(params)
    exact = node.wait(now)
    return round(exact / nominal) * nominal if nominal > 0 else exact


def transfer_aware(
    req: RequestView, nodes: list[Node], now: float, params: CostParams
) -> Placement:
    hot = hot_node(req.blocks, nodes)
    best: Placement | None = None
    for node in nodes:
        for transfer in (False, True):
            if transfer and node is hot:
                continue
            p = predict(req.blocks, node, now, params, hot, transfer=transfer)
            # Re-score TTFT using the coarse (quantized) wait the router can actually see.
            approx = p.ttft - node.wait(now) + _quantized_wait(node, now, params)
            scored = Placement(p.node_id, p.transfer, p.regime, approx)
            if best is None or (scored.ttft, scored.node_id, scored.transfer) < (
                best.ttft,
                best.node_id,
                best.transfer,
            ):
                best = scored
    assert best is not None
    return best


class _RouterIndex:
    """Router-side belief of which blocks each node holds, built from routing decisions.

    Mirrors llm-d's approximate prefix-affinity indexer: every routed request's blocks are
    recorded against the chosen node in a capacity-bounded LRU. The router never observes
    node-side evictions, so beliefs go stale exactly the way a real router-side index does.
    """

    def __init__(self, num_nodes: int, capacity: int):
        self._capacity = capacity
        self._blocks: list[OrderedDict[int, None]] = [OrderedDict() for _ in range(num_nodes)]

    def matched(self, node_id: int, req_blocks: list[int]) -> int:
        return matched_prefix(req_blocks, self._blocks[node_id])

    def record(self, node_id: int, req_blocks: list[int]) -> None:
        index = self._blocks[node_id]
        for h in req_blocks:
            index.pop(h, None)
            index[h] = None
        while len(index) > self._capacity:
            index.popitem(last=False)


class WeightedScorers:
    """llm-d EPP-style filter-then-score pipeline; never transfers, like production llm-d.

    Stage 1 is GIE's saturation filter: nodes whose (coarse) queue exceeds
    ``saturation_depth`` nominal requests are dropped from the candidate set, falling back
    to all nodes if everything is saturated. Without this stage the prefix scorer herds
    every request onto whichever node served the shared system prompt first (prefix weight
    2 beats load weight 1 whenever >50% of blocks match, regardless of queue depth) -- the
    filter is what makes production prefix-affinity self-limiting.

    Stage 2 scores each surviving node as sum(weight * scorer), scorers normalized to
    [0, 1]; the highest total wins (ties -> lowest node id). Two scorers, matching the
    default llm-d profile shape:

      * prefix scorer (weight 2) -- fraction of the request's blocks believed resident on
        the node: true cache state when ``precise``, else the ``_RouterIndex`` belief.
      * load scorer (weight 1) -- min-max-normalized inverse queue delay, seen through the
        same coarse quantized wait ``transfer_aware`` uses (a router polls load, it does
        not read ``busy_until`` clocks).
    """

    def __init__(
        self,
        precise: bool,
        prefix_weight: float = 2.0,
        load_weight: float = 1.0,
        saturation_depth: int = DEFAULT_SATURATION_DEPTH,
        index_capacity: int = 2048,
    ) -> None:
        self._precise = precise
        self._prefix_weight = prefix_weight
        self._load_weight = load_weight
        self._saturation_depth = saturation_depth
        self._index_capacity = index_capacity
        self._index: _RouterIndex | None = None

    def _prefix_score(self, node: Node, req_blocks: list[int]) -> float:
        if not req_blocks:
            return 0.0
        if self._precise:
            return node.matched(req_blocks) / len(req_blocks)
        assert self._index is not None
        return self._index.matched(node.node_id, req_blocks) / len(req_blocks)

    def __call__(
        self, req: RequestView, nodes: list[Node], now: float, params: CostParams
    ) -> Placement:
        if self._index is None:
            self._index = _RouterIndex(len(nodes), self._index_capacity)

        waits = {n.node_id: _quantized_wait(n, now, params) for n in nodes}
        limit = self._saturation_depth * _nominal_service(params)
        candidates = [n for n in nodes if waits[n.node_id] <= limit]
        if not candidates:
            candidates = list(nodes)

        lo = min(waits[n.node_id] for n in candidates)
        hi = max(waits[n.node_id] for n in candidates)

        def score(n: Node) -> float:
            load = 1.0 if hi == lo else 1.0 - (waits[n.node_id] - lo) / (hi - lo)
            return self._prefix_weight * self._prefix_score(n, req.blocks) + (
                self._load_weight * load
            )

        best = max(candidates, key=lambda n: (score(n), -n.node_id))
        if not self._precise:
            self._index.record(best.node_id, req.blocks)

        hot = hot_node(req.blocks, nodes)
        return predict(req.blocks, best, now, params, hot, transfer=False)


def class_aware(req: RequestView, nodes: list[Node], now: float, params: CostParams) -> Placement:
    """Route by inferred workload class; each class has different prefix-cache economics.

    * tool-session turn -- growing per-session prefix worth chasing: full transfer-aware
      scoring (wait / transfer / recompute on coarse load signals).
    * RAG query -- affinity to the node holding the shared document prefix, spilling
      (transfer-aware) only when that node is saturated past the GIE-style depth limit.
    * one-shot -- nothing cacheable to exploit and nothing worth polluting a hot cache
      for: least (quantized) loaded node, never chase cache.
    """
    kind = classify(req)
    if kind == "oneshot":
        node = min(nodes, key=lambda n: (_quantized_wait(n, now, params), n.node_id))
        hot = hot_node(req.blocks, nodes)
        return predict(req.blocks, node, now, params, hot, transfer=False)
    if kind == "rag":
        hot = hot_node(req.blocks, nodes)
        limit = DEFAULT_SATURATION_DEPTH * _nominal_service(params)
        if hot.matched(req.blocks) > 0 and _quantized_wait(hot, now, params) <= limit:
            return predict(req.blocks, hot, now, params, hot, transfer=False)
    # Session turns, cold RAG prefixes, and saturated-hot spills all score the full
    # (node, transfer) grid on the same coarse signals a real router polls.
    return transfer_aware(req, nodes, now, params)


def oracle(req: RequestView, nodes: list[Node], now: float, params: CostParams) -> Placement:
    return best_placement(req.blocks, nodes, now, params, allow_transfer=True)


def build_policy(name: str) -> Policy:
    if name == "round-robin":
        return RoundRobin()
    if name == "cache-local":
        return cache_local
    if name == "transfer-aware":
        return transfer_aware
    if name == "weighted-precise":
        return WeightedScorers(precise=True)
    if name == "weighted-approx":
        return WeightedScorers(precise=False)
    if name == "class-aware":
        return class_aware
    if name == "oracle":
        return oracle
    raise ValueError(f"unknown policy: {name}")


POLICY_NAMES = (
    "round-robin",
    "cache-local",
    "transfer-aware",
    "weighted-precise",
    "weighted-approx",
    "class-aware",
    "oracle",
)
