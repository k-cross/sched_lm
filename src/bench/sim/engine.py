"""Discrete-event engine: replay an arrival stream under a policy and record what happens.

Requests are processed in arrival order (the stream is pre-sorted and policy-independent,
so every policy is judged on the same arrivals). For each request we:

  1. compute the oracle placement on the *current* node state -> a lower-bound TTFT,
  2. ask the policy for its placement, recompute that placement's *true* TTFT on the same
     state (so an approximate policy is charged its real cost, not its own estimate),
  3. admit the request to the chosen node -- advancing its queue and seeding its cache,
  4. record TTFT, the regime chosen, regret vs. the oracle, and whether the decision was
     *coupled* to other nodes' load.

The result is shaped like ``traffic.BenchmarkResult`` (``ttfts``/``e2e_latencies``/
``successes``/``errors``) so the existing percentile + report code renders it unchanged,
with the extra sim-only fields carried alongside.
"""

from collections import Counter
from dataclasses import dataclass, field

from bench.sim.cost import CostParams, Placement, Regime, best_placement, hot_node, predict
from bench.sim.node import Node
from bench.sim.policies import Policy
from bench.sim.workload import TurnRequest


@dataclass
class SimResult:
    ttfts: list[float] = field(default_factory=list)
    e2e_latencies: list[float] = field(default_factory=list)
    successes: int = 0
    errors: int = 0
    regimes: Counter = field(default_factory=Counter)
    regrets: list[float] = field(default_factory=list)
    coupled: int = 0
    reused_blocks: int = 0
    total_blocks: int = 0

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


def _is_coupled(req_blocks: list[int], nodes: list[Node], now: float, params: CostParams) -> bool:
    """True if the optimal placement changes once the cluster's queues are emptied.

    We compare the oracle decision under the real queue state against the decision with
    every node forced idle (waits zeroed, caches kept). If they differ, this request's
    optimal placement depends on *other* nodes' load -- it is not independent of them.
    """
    loaded = best_placement(req_blocks, nodes, now, params)
    saved = [n.busy_until for n in nodes]
    for n in nodes:
        n.busy_until = now  # wait() -> 0 for all nodes
    try:
        idle = best_placement(req_blocks, nodes, now, params)
    finally:
        for n, b in zip(nodes, saved, strict=True):
            n.busy_until = b
    return (loaded.node_id, loaded.transfer) != (idle.node_id, idle.transfer)


def run_simulation(
    requests: list[TurnRequest],
    policy: Policy,
    num_nodes: int,
    cache_blocks: int,
    params: CostParams,
) -> SimResult:
    nodes = [Node(i, cache_blocks) for i in range(num_nodes)]
    result = SimResult()

    for req in requests:
        blocks = req.block_hashes
        now = req.arrival

        oracle_p = best_placement(blocks, nodes, now, params)
        if _is_coupled(blocks, nodes, now, params):
            result.coupled += 1

        chosen: Placement = policy(blocks, nodes, now, params)

        # Recompute the chosen placement's true cost on the real state: an approximate
        # policy must be charged what its decision actually costs, not what it guessed.
        hot = hot_node(blocks, nodes)
        node = nodes[chosen.node_id]
        realized = predict(blocks, node, now, params, hot, transfer=chosen.transfer)

        result.ttfts.append(realized.ttft)
        result.e2e_latencies.append(realized.ttft + req.gen_tokens * params.inter_token)
        result.regimes[realized.regime] += 1
        result.regrets.append(max(0.0, realized.ttft - oracle_p.ttft))
        result.successes += 1

        resident = _resident(blocks, node, chosen.transfer, hot)
        result.reused_blocks += resident
        result.total_blocks += len(blocks)

        missing = len(blocks) - resident
        service = params.prefill_time(missing) + req.gen_tokens * params.inter_token
        node.admit(now, service, blocks)

    return result
