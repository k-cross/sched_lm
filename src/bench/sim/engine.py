"""Discrete-event engine: replay an arrival stream under a policy and record what happens.

Requests are processed in arrival order (the stream is pre-sorted and policy-independent,
so every policy is judged on the same arrivals). For each request we:

  1. compute the oracle placement on the *current* node state -> a lower-bound TTFT,
  2. ask the policy for its placement, recompute that placement's *true* TTFT on the same
     state (so an approximate policy is charged its real cost, not its own estimate),
  3. admit the request to the chosen node -- advancing its queue and seeding its cache,
  4. record TTFT, the regime chosen, *routing regret* vs. the oracle on this simulation's
     own node state, and whether the decision was *coupled* to other nodes' load.

Note: routing regret measures how well a policy *routes* given its own cache contents. It
is **not** comparable across policies with different eviction strategies (e.g. LRU vs.
Belady's MIN). For a cross-policy distance-from-best metric, see the TTFT gap column in
the report, which uses a single oracle baseline for every row.

The result is shaped like ``traffic.BenchmarkResult`` (``ttfts``/``e2e_latencies``/
``successes``/``errors``) so the existing percentile + report code renders it unchanged,
with the extra sim-only fields carried alongside.
"""

from collections import Counter, OrderedDict
from dataclasses import dataclass, field

from bench.sim.cost import CostParams, Placement, Regime, best_placement, hot_node, predict
from bench.sim.node import BeladyNode, Node
from bench.sim.policies import Policy, RequestView, observe_gap
from bench.sim.workload import TurnRequest


def request_view(req: TurnRequest) -> RequestView:
    """Project a generated request down to what the router may observe (never ``kind``)."""
    last = req.messages[-1]["content"] if req.messages else ""
    return RequestView(
        blocks=req.block_hashes,
        prompt_tokens=len(req.tokens),
        num_messages=len(req.messages),
        has_tool_messages=any(m["role"] == "tool" for m in req.messages),
        last_message_tokens=len(last.split()),
        conversation_id=req.conversation_id,
        last_tool_name=req.tool_name,
    )


@dataclass
class KindStats:
    """Per-workload-class slice of the aggregate metrics."""

    count: int = 0
    ttfts: list[float] = field(default_factory=list)
    regrets: list[float] = field(default_factory=list)
    reused_blocks: int = 0
    total_blocks: int = 0

    @property
    def hit_rate(self) -> float:
        return self.reused_blocks / self.total_blocks if self.total_blocks else 0.0

    @property
    def mean_regret(self) -> float:
        return sum(self.regrets) / len(self.regrets) if self.regrets else 0.0


@dataclass
class SimResult:
    ttfts: list[float] = field(default_factory=list)
    e2e_latencies: list[float] = field(default_factory=list)
    successes: int = 0
    errors: int = 0
    regimes: "Counter[Regime]" = field(default_factory=Counter)
    regrets: list[float] = field(default_factory=list)
    coupled: int = 0
    reused_blocks: int = 0
    total_blocks: int = 0
    # Marked-and-unexpired blocks evicted under pressure, summed across nodes: the
    # pinned-pressure signal a router watches to tell whether it over-pinned (RFC-0001 §4).
    pinned_evictions: int = 0
    by_kind: dict[str, KindStats] = field(default_factory=dict)
    # Realized inter-turn gaps observed per tool signature (policy-independent workload
    # property): what a router-side ToolGapIndex learns from. Empty unless the workload was
    # generated with tool-reliability modeling.
    gap_by_tool: dict[str, list[float]] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        return self.reused_blocks / self.total_blocks if self.total_blocks else 0.0

    @property
    def regime_mix(self) -> dict[str, float]:
        total = sum(self.regimes.values())
        if not total:
            return {r.value: 0.0 for r in Regime}
        return {r.value: self.regimes.get(r, 0) / total for r in Regime}

    @property
    def coupled_fraction(self) -> float:
        n = self.successes
        return self.coupled / n if n else 0.0


def _resident(req_blocks: list[int], node: Node, transfer: bool, hot: Node) -> int:
    local = node.matched(req_blocks)
    xfer = max(0, hot.matched(req_blocks) - local) if transfer else 0
    return local + xfer


def _is_coupled(
    req_blocks: list[int],
    nodes: list[Node],
    now: float,
    params: CostParams,
    loaded: Placement,
) -> bool:
    """True if the optimal placement changes once the cluster's queues are emptied.

    We compare the oracle decision under the real queue state (``loaded``, already computed
    by the caller) against the decision with every node forced idle (waits zeroed, caches
    kept). If they differ, this request's optimal placement depends on *other* nodes' load
    -- it is not independent of them.
    """
    saved = [n.busy_until for n in nodes]
    for n in nodes:
        n.busy_until = now  # wait() -> 0 for all nodes
    try:
        idle = best_placement(req_blocks, nodes, now, params)
    finally:
        for n, b in zip(nodes, saved, strict=True):
            n.busy_until = b
    return (loaded.node_id, loaded.transfer) != (idle.node_id, idle.transfer)


def _build_future_uses(requests: list[TurnRequest]) -> dict[int, list[int]]:
    """Map each block hash to the sorted list of request indices where it appears."""
    uses: dict[int, list[int]] = {}
    for idx, req in enumerate(requests):
        for bh in req.block_hashes:
            uses.setdefault(bh, []).append(idx)
    return uses


def run_simulation(
    requests: list[TurnRequest],
    policy: Policy,
    num_nodes: int,
    cache_blocks: int,
    params: CostParams,
    *,
    belady: bool = False,
) -> SimResult:
    nodes: list[Node]
    if belady:
        future_uses = _build_future_uses(requests)
        nodes = [BeladyNode(i, cache_blocks, future_uses) for i in range(num_nodes)]
    else:
        nodes = [Node(i, cache_blocks) for i in range(num_nodes)]
    result = SimResult()
    last_seen: OrderedDict[int, float] = OrderedDict()

    for idx, req in enumerate(requests):
        blocks = req.block_hashes
        now = req.arrival

        # Observed re-arrival gap per tool (policy-independent): the gap since a
        # conversation's previous turn belongs to the tool whose result now sits in history.
        gap = observe_gap(last_seen, req.conversation_id, req.tool_name, now)
        if gap is not None and req.tool_name is not None:
            result.gap_by_tool.setdefault(req.tool_name, []).append(gap)

        oracle_p = best_placement(blocks, nodes, now, params)
        if _is_coupled(blocks, nodes, now, params, oracle_p):
            result.coupled += 1

        chosen: Placement = policy(request_view(req), nodes, now, params)

        # Recompute the chosen placement's true cost on the real state: an approximate
        # policy must be charged what its decision actually costs, not what it guessed.
        hot = hot_node(blocks, nodes)
        node = nodes[chosen.node_id]
        realized = predict(blocks, node, now, params, hot, transfer=chosen.transfer)

        regret = max(0.0, realized.ttft - oracle_p.ttft)
        result.ttfts.append(realized.ttft)
        result.e2e_latencies.append(realized.ttft + req.gen_tokens * params.inter_token)
        result.regimes[realized.regime] += 1
        result.regrets.append(regret)
        result.successes += 1

        resident = _resident(blocks, node, chosen.transfer, hot)
        result.reused_blocks += resident
        result.total_blocks += len(blocks)

        kind = result.by_kind.setdefault(req.kind, KindStats())
        kind.count += 1
        kind.ttfts.append(realized.ttft)
        kind.regrets.append(regret)
        kind.reused_blocks += resident
        kind.total_blocks += len(blocks)

        missing = len(blocks) - resident
        service = params.prefill_time(missing) + req.gen_tokens * params.inter_token
        node.admit(
            now,
            service,
            blocks,
            retain_until=chosen.retain_until,
            at_index=idx,
            priority=chosen.priority,
        )

    result.pinned_evictions = sum(n.pinned_evictions for n in nodes)
    return result
