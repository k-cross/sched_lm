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
  * ``class_aware_reliability`` -- ``class_aware`` plus a per-tool re-arrival-gap estimate
    (learned from timing alone, never success/failure): a tool-session turn whose last tool
    call confidently predicts a short return soft-pins its prefix on the chosen node so it
    survives eviction until the session comes back.
  * ``oracle``        -- argmin over (node, transfer) with perfect current state. The lower
    bound every other policy's regret is measured against.

Policies receive a :class:`RequestView` -- the request as a router sees it (block hashes
plus body-derived observables), not the generator's ``TurnRequest``.
"""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from bench.sim.blocks import matched_prefix
from bench.sim.cost import CostParams, Placement, best_placement, hot_node, predict
from bench.sim.node import Node
from bench.sim.priority import EVICT_FIRST, HIGH

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
    # Timing-observable signals for the reliability-aware policy: a session/affinity header
    # and the most recent tool call in the history. Both are body/header-derivable and never
    # carry the latent success/failure outcome. ``compare=False`` keeps view identity
    # (used as a cache key in tests) independent of them.
    conversation_id: int | None = field(default=None, compare=False)
    last_tool_name: str | None = field(default=None, compare=False)


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


@dataclass
class GapStat:
    """Online estimate of a tool's re-arrival gap: EWMA mean and variance."""

    count: int = 0
    mean: float = 0.0
    var: float = 0.0


class ToolGapIndex:
    """Per-tool re-arrival-gap estimates, learned from timing alone.

    Mirrors the cheap, bounded bookkeeping a real gateway could keep: one small table keyed
    by tool signature, each entry an EWMA of the observed inter-turn gap's mean and variance.
    It never sees success/failure -- only the realized gap between a conversation's
    consecutive turns -- so it works from pure timing metadata.

    The table is capacity-bounded (an LRU); tools that fall off, and tools never yet seen,
    fall back to a static ``priors`` map, then a global ``default_prior``. That bound is the
    memory/compute guard: cost is hard-capped regardless of how many distinct tools appear.
    """

    def __init__(
        self,
        capacity: int = 512,
        alpha: float = 0.3,
        priors: dict[str, tuple[float, float]] | None = None,
        # High variance relative to the mean -> low confidence, so an unseen tool does not
        # trigger retention until it has actually been observed.
        default_prior: tuple[float, float] = (2.0, 16.0),
    ) -> None:
        self._capacity = capacity
        self._alpha = alpha
        self._priors = dict(priors) if priors else {}
        self._default_prior = default_prior
        self._stats: OrderedDict[str, GapStat] = OrderedDict()

    def record(self, tool: str, gap: float) -> None:
        """Fold one observed gap into the tool's EWMA mean and variance."""
        stat = self._stats.pop(tool, None)
        if stat is None:
            stat = GapStat()
        if stat.count == 0:
            stat.mean = gap
            stat.var = 0.0
        else:
            delta = gap - stat.mean
            stat.mean += self._alpha * delta
            # EWMA of squared deviation from the (pre-update) mean -- a standard online
            # variance tracker that needs only the running mean, no second pass.
            stat.var = (1 - self._alpha) * (stat.var + self._alpha * delta * delta)
        stat.count += 1
        self._stats[tool] = stat  # reinsert as most-recently-used
        while len(self._stats) > self._capacity:
            self._stats.popitem(last=False)

    def predict(self, tool: str | None) -> tuple[float, float]:
        """Best (mean, variance) estimate: learned stat, else prior, else default."""
        if tool is not None:
            stat = self._stats.get(tool)
            if stat is not None:  # any stored stat has been recorded at least once
                return stat.mean, stat.var
            if tool in self._priors:
                return self._priors[tool]
        return self._default_prior

    def confidence(self, tool: str | None) -> float:
        """Inverse-variance weight in [0, 1]: high for consistent (low-variance) tools."""
        return gap_confidence(*self.predict(tool))


def gap_confidence(mean: float, var: float) -> float:
    """Inverse-variance weight in [0, 1] for a (mean, variance) gap estimate.

    Normalized against the mean so it is scale-free -- a tool whose gap varies little
    relative to its mean is trusted; a bimodal fast/slow tool is not.
    """
    scale = max(mean * mean, 1e-9)
    return 1.0 / (1.0 + var / scale)


# Bound on the per-conversation last-seen table used to measure re-arrival gaps. Like every
# other router-side table here it is capacity-capped so memory stays O(active conversations),
# not O(all conversations ever) -- single-turn RAG/one-shot ids fall out under the LRU.
GAP_TRACKER_CAPACITY = 8192


def observe_gap(
    last_seen: "OrderedDict[int, float]",
    conversation_id: int | None,
    tool: str | None,
    now: float,
    capacity: int = GAP_TRACKER_CAPACITY,
) -> float | None:
    """Realized re-arrival gap since this conversation's previous turn, or None.

    Records ``now`` as the conversation's latest turn in a bounded LRU and returns the gap to
    attribute to ``tool`` (the tool whose result now sits in the history). Returns None when
    there is nothing learnable: no conversation link, no prior turn, or no tool yet (turn 0 --
    which still updates ``last_seen`` so the next turn's gap can be measured).
    """
    if conversation_id is None:
        return None
    prev = last_seen.get(conversation_id)
    last_seen[conversation_id] = now
    last_seen.move_to_end(conversation_id)
    while len(last_seen) > capacity:
        last_seen.popitem(last=False)
    if prev is None or tool is None:
        return None
    return now - prev


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


# A predicted return within this many seconds, at this confidence, is worth keeping the
# prefix warm for; slower or less-certain returns are not (the cache would be better spent
# on other traffic).
DEFAULT_SHORT_GAP = 3.0
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_RETAIN_MARGIN = 1.5


class ClassAwareReliability:
    """``class_aware`` plus a learned re-arrival-gap signal for tool sessions.

    Routing is the same class-conditioned logic as :func:`class_aware`. The addition is a
    :class:`ToolGapIndex` that learns, per tool signature, how soon a session tends to return
    after calling it -- from timing alone (the realized gap between a conversation's turns),
    never from the tool's success/failure.

    When a tool-session turn is routed and its most recent tool call **confidently** predicts
    a **short** return (low-variance, short-mean), the chosen node's copy of this prefix is
    marked :data:`~bench.sim.priority.HIGH` with a ``retain_until`` lease so it survives
    eviction until the session comes back -- turning what would have been a recompute into a
    cheap cache hit. A long predicted gap, or an unpredictable (high-variance) or unseen
    tool, adds no retention: the router does not gamble cache on an unreliable prediction.

    One-shots -- no future reuse, pure cache pollution -- are marked
    :data:`~bench.sim.priority.EVICT_FIRST` so their blocks are dropped *before* unmarked
    traffic, the below-normal rank #37003's retain-only evictor cannot express (RFC-0001 §1,
    §5). Both directives ride the returned :class:`~bench.sim.cost.Placement`; routing itself
    is unchanged from :func:`class_aware`.
    """

    def __init__(
        self,
        index: ToolGapIndex | None = None,
        short_gap: float = DEFAULT_SHORT_GAP,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        retain_margin: float = DEFAULT_RETAIN_MARGIN,
    ) -> None:
        self._index = index if index is not None else ToolGapIndex()
        self._short_gap = short_gap
        self._conf_threshold = conf_threshold
        self._retain_margin = retain_margin
        # conversation_id -> time it was last seen, to measure the realized re-arrival gap.
        self._last_seen: OrderedDict[int, float] = OrderedDict()

    def _learn(self, req: RequestView, now: float) -> None:
        """Record the realized gap since this conversation's previous turn (timing only).

        The gap belongs to the tool whose result now sits in the history -- this request's
        most recent tool call.
        """
        gap = observe_gap(self._last_seen, req.conversation_id, req.last_tool_name, now)
        if gap is not None:
            assert req.last_tool_name is not None  # observe_gap returns None when it is
            self._index.record(req.last_tool_name, gap)

    def __call__(
        self, req: RequestView, nodes: list[Node], now: float, params: CostParams
    ) -> Placement:
        self._learn(req, now)
        # Route exactly as class-aware does; classes differ only in the directive on top.
        place = class_aware(req, nodes, now, params)
        kind = classify(req)
        if kind == "oneshot":
            # No reuse: shed the prefix before it evicts anything with retention value.
            return replace(place, priority=EVICT_FIRST)
        if kind != "tool":
            return place

        # Mark the prefix HIGH if a confident, short return is predicted for this tool.
        mean, var = self._index.predict(req.last_tool_name)
        if gap_confidence(mean, var) >= self._conf_threshold and mean <= self._short_gap:
            # Keep the prefix warm at least mean * margin, but extend by one standard
            # deviation when the (confident-but-nonzero) spread pushes likely returns past
            # that -- so we don't drop the pin just before a higher-variance session returns.
            window = max(mean * self._retain_margin, mean + var**0.5)
            return replace(place, retain_until=now + window, priority=HIGH)
        return place


def oracle(req: RequestView, nodes: list[Node], now: float, params: CostParams) -> Placement:
    return best_placement(req.blocks, nodes, now, params, allow_transfer=True)


def build_policy(name: str, *, retain_margin: float = DEFAULT_RETAIN_MARGIN) -> Policy:
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
    if name == "class-aware-reliability":
        return ClassAwareReliability(retain_margin=retain_margin)
    if name == "oracle" or name in BELADY_POLICIES:
        # oracle-belady shares the oracle *router*; it differs only in node-side eviction,
        # which run_simulation realizes when told the policy is in BELADY_POLICIES. The
        # routing function alone cannot encode that, so callers must consult uses_belady().
        return oracle
    raise ValueError(f"unknown policy: {name}")


# Policies that differ from a plain oracle only in node-side eviction strategy (Belady's
# MIN). run_simulation must be given belady=True for these, so the single source of truth
# lives here rather than as a magic string re-derived in the CLI and report.
BELADY_POLICIES = frozenset({"oracle-belady"})

# Oracle baselines for the cross-policy TTFT-gap metric, strongest first; the report uses
# the best one present in a given run.
ORACLE_BASELINES = ("oracle-belady", "oracle")


def uses_belady(name: str) -> bool:
    """Whether policy ``name`` requires run_simulation(..., belady=True) to realize it."""
    return name in BELADY_POLICIES


POLICY_NAMES = (
    "round-robin",
    "cache-local",
    "transfer-aware",
    "weighted-precise",
    "weighted-approx",
    "class-aware",
    "class-aware-reliability",
    "oracle",
    "oracle-belady",
)
