"""The prefix-cache routing cost model: predicted TTFT for placing a request on a node.

This is where the three regimes come from. Given a request whose prefix is (partly) cached
somewhere, the predicted time-to-first-token for a candidate node is the sum of three
physically distinct costs:

    TTFT = wait(node) + transfer_time(if we seed the node) + prefill(missing tokens)

Route selection is an ``argmin`` of this over ``(node, transfer?)``. The winning branch
*is* the regime:

  * **wait-for-hot** -- send to the node that already holds the longest prefix; pay only
    its queue delay. Wins when the hot node's queue is shorter than recomputing/transferring.
  * **transfer**     -- send to a less-loaded node but copy the hot node's extra KV blocks
    over the interconnect. Wins when transfer + a shorter queue beats waiting, and beats
    recompute (i.e. bandwidth is cheap relative to prefill).
  * **recompute**    -- send to a less-loaded node and recompute the prefix from scratch.
    Wins when the uncached prefix is short, or the hot node is deep in queue and bandwidth
    is scarce.

The constants are order-of-magnitude ratios for a small model on one host, not calibrated
absolutes -- the research question is *when the argmin flips*, not the wall-clock numbers.
"""

from dataclasses import dataclass
from enum import StrEnum

from bench.sim.blocks import DEFAULT_BLOCK_SIZE
from bench.sim.node import Node


class Regime(StrEnum):
    WAIT = "wait"
    TRANSFER = "transfer"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class CostParams:
    block_size: int = DEFAULT_BLOCK_SIZE
    # Seconds of prefill compute per uncached token (the "recompute" price).
    prefill_per_token: float = 0.0002
    # Fixed per-request prefill/scheduling overhead, seconds.
    prefill_fixed: float = 0.005
    # Bytes of KV state per block (2 * layers * heads * head_dim * dtype, one block).
    block_kv_bytes: float = 800_000.0
    # Interconnect bandwidth in bytes/sec for KV-block transfer between nodes.
    bandwidth: float = 4e9
    # Fixed latency to set up a KV transfer, seconds.
    link_latency: float = 0.0005
    # Decode cost per generated token, seconds (drives how long a node stays busy).
    inter_token: float = 0.01

    def transfer_time(self, blocks: int) -> float:
        if blocks <= 0:
            return 0.0
        return blocks * self.block_kv_bytes / self.bandwidth + self.link_latency

    def prefill_time(self, missing_blocks: int) -> float:
        return missing_blocks * self.block_size * self.prefill_per_token + self.prefill_fixed


@dataclass(frozen=True)
class Placement:
    """A candidate routing decision and its predicted TTFT."""

    node_id: int
    transfer: bool
    regime: Regime
    ttft: float
    # Soft-pin expiry a policy may set to keep the chosen node's copy of this request's
    # prefix warm until the session is predicted to return. 0.0 = no retention (pure LRU).
    # Meaningful only on a policy's *returned* placement; the cost candidates from
    # ``predict``/``best_placement`` always leave it 0.0.
    retain_until: float = 0.0


def hot_node(req_blocks: list[int], nodes: list[Node]) -> Node:
    """The node holding the longest cached prefix of this request (ties -> lowest id)."""
    return max(nodes, key=lambda n: (n.matched(req_blocks), -n.node_id))


def predict(
    req_blocks: list[int],
    node: Node,
    now: float,
    params: CostParams,
    hot: Node,
    transfer: bool,
) -> Placement:
    """Predicted TTFT for serving ``req_blocks`` on ``node`` at ``now``.

    ``hot`` is the current best-cached node (from :func:`hot_node`); ``transfer`` asks to
    seed ``node`` with the blocks ``hot`` has beyond what ``node`` already holds. The
    returned :class:`Placement` carries the regime this ``(node, transfer)`` pair realizes.
    """
    total = len(req_blocks)
    local = node.matched(req_blocks)

    xfer_blocks = max(0, hot.matched(req_blocks) - local) if transfer else 0
    resident = local + xfer_blocks
    missing = total - resident

    xfer_cost = params.transfer_time(xfer_blocks)
    ttft = node.wait(now) + xfer_cost + params.prefill_time(missing)

    if xfer_blocks > 0:
        regime = Regime.TRANSFER
    elif node is hot and local > 0:
        # Reusing an existing cached prefix in place: pay the queue, not the prefill.
        regime = Regime.WAIT
    else:
        regime = Regime.RECOMPUTE

    return Placement(node_id=node.node_id, transfer=transfer, regime=regime, ttft=ttft)


def best_placement(
    req_blocks: list[int],
    nodes: list[Node],
    now: float,
    params: CostParams,
    allow_transfer: bool = True,
) -> Placement:
    """Argmin predicted-TTFT placement over every node and transfer choice (the oracle)."""
    hot = hot_node(req_blocks, nodes)
    candidates: list[Placement] = []
    for node in nodes:
        candidates.append(predict(req_blocks, node, now, params, hot, transfer=False))
        if allow_transfer and node is not hot:
            candidates.append(predict(req_blocks, node, now, params, hot, transfer=True))
    return min(candidates, key=lambda p: (p.ttft, p.node_id, p.transfer))
