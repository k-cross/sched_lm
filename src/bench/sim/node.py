"""A simulated inference node: a KV-block LRU cache plus a single-server queue.

The two pieces of state that drive routing live here. ``busy_until`` is when the node
will next be free (its queue tail), and ``cache`` is the set of KV block hashes currently
resident, ordered by recency so we can evict under a capacity bound. The capacity bound
is what makes think-time matter: while a session pauses between turns, other traffic can
push its affinity prefix out of the cache, turning a cheap wait into a recompute.
"""

from collections import OrderedDict

from bench.sim.blocks import matched_prefix


class Node:
    def __init__(self, node_id: int, cache_blocks: int):
        self.node_id = node_id
        # When the node's single server next becomes free (== queue tail time).
        self.busy_until = 0.0
        # Resident KV blocks as an LRU: keys are block hashes, insertion order = recency.
        self._cache: OrderedDict[int, None] = OrderedDict()
        self.cache_capacity = cache_blocks

    @property
    def cache(self) -> "OrderedDict[int, None]":
        return self._cache

    def matched(self, req_blocks: list[int]) -> int:
        """Longest cached leading prefix of ``req_blocks`` on this node."""
        return matched_prefix(req_blocks, self._cache)

    def wait(self, now: float) -> float:
        """Queueing delay before this node can start serving a request arriving at ``now``."""
        return max(0.0, self.busy_until - now)

    def admit(self, now: float, service_time: float, req_blocks: list[int]) -> None:
        """Advance the queue and insert the request's blocks as most-recently-used.

        Service is FIFO on one server: a request starts at ``max(now, busy_until)`` and
        the node is busy for ``service_time`` after that. All of the request's prefix
        blocks become resident (recompute or transfer both leave them cached), evicting
        the least-recently-used blocks past capacity.
        """
        start = max(now, self.busy_until)
        self.busy_until = start + service_time
        for h in req_blocks:
            self._cache.pop(h, None)
            self._cache[h] = None
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)
