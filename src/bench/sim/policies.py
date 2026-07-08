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
  * ``oracle``        -- argmin over (node, transfer) with perfect current state. The lower
    bound every other policy's regret is measured against.
"""

from collections.abc import Callable

from bench.sim.cost import CostParams, Placement, best_placement, hot_node, predict
from bench.sim.node import Node

Policy = Callable[[list[int], list[Node], float, CostParams], Placement]


class RoundRobin:
    """Stateful cyclic assignment; ignores cache and load entirely."""

    def __init__(self) -> None:
        self._next = 0

    def __call__(
        self, req_blocks: list[int], nodes: list[Node], now: float, params: CostParams
    ) -> Placement:
        node = nodes[self._next % len(nodes)]
        self._next += 1
        hot = hot_node(req_blocks, nodes)
        return predict(req_blocks, node, now, params, hot, transfer=False)


def cache_local(
    req_blocks: list[int], nodes: list[Node], now: float, params: CostParams
) -> Placement:
    hot = hot_node(req_blocks, nodes)
    return predict(req_blocks, hot, now, params, hot, transfer=False)


def _quantized_wait(node: Node, now: float, params: CostParams) -> float:
    """Approximate a node's queue delay as a whole number of nominal service times.

    A real router polls coarse load signals, not an exact busy-until clock; rounding the
    wait to nominal-request units models that imprecision so ``transfer_aware`` can be
    strictly weaker than the oracle.
    """
    nominal = params.prefill_fixed + params.inter_token * 40
    exact = node.wait(now)
    return round(exact / nominal) * nominal if nominal > 0 else exact


def transfer_aware(
    req_blocks: list[int], nodes: list[Node], now: float, params: CostParams
) -> Placement:
    hot = hot_node(req_blocks, nodes)
    best: Placement | None = None
    for node in nodes:
        for transfer in (False, True):
            if transfer and node is hot:
                continue
            p = predict(req_blocks, node, now, params, hot, transfer=transfer)
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


def oracle(req_blocks: list[int], nodes: list[Node], now: float, params: CostParams) -> Placement:
    return best_placement(req_blocks, nodes, now, params, allow_transfer=True)


def build_policy(name: str) -> Policy:
    if name == "round-robin":
        return RoundRobin()
    if name == "cache-local":
        return cache_local
    if name == "transfer-aware":
        return transfer_aware
    if name == "oracle":
        return oracle
    raise ValueError(f"unknown policy: {name}")


POLICY_NAMES = ("round-robin", "cache-local", "transfer-aware", "oracle")
