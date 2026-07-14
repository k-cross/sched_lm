"""A simulated inference node: a KV-block LRU cache plus a single-server queue.

The two pieces of state that drive routing live here. ``busy_until`` is when the node
will next be free (its queue tail), and ``cache`` is the set of KV block hashes currently
resident, ordered by recency so we can evict under a capacity bound. The capacity bound
is what makes think-time matter: while a session pauses between turns, other traffic can
push its affinity prefix out of the cache, turning a cheap wait into a recompute.
"""

from bisect import bisect_right
from collections import OrderedDict

from bench.sim.blocks import matched_prefix


class Node:
    def __init__(self, node_id: int, cache_blocks: int):
        self.node_id = node_id
        # When the node's single server next becomes free (== queue tail time).
        self.busy_until = 0.0
        # Resident KV blocks as an LRU: keys are block hashes, insertion order = recency.
        self._cache: OrderedDict[int, None] = OrderedDict()
        # Optional soft-pin expiry per block: while now < retain_until, a block resists
        # eviction (the router keeps a likely-returning session's prefix warm). Blocks not
        # present here have no protection -- the default pure-LRU behavior.
        self._retain_until: dict[int, float] = {}
        # Position in the arrival stream of the request currently being admitted; only
        # lookahead eviction strategies (BeladyNode) read it.
        self._at_index = 0
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

    def admit(
        self,
        now: float,
        service_time: float,
        req_blocks: list[int],
        retain_until: float = 0.0,
        at_index: int = 0,
    ) -> None:
        """Advance the queue and insert the request's blocks as most-recently-used.

        Service is FIFO on one server: a request starts at ``max(now, busy_until)`` and
        the node is busy for ``service_time`` after that. All of the request's prefix
        blocks become resident (recompute or transfer both leave them cached), evicting
        the least-recently-used blocks past capacity.

        ``retain_until`` soft-pins this request's blocks: until that time, eviction prefers
        to drop *unprotected* (expired) blocks first, so a session predicted to return soon
        keeps its prefix warm across the gap. ``retain_until <= now`` (the default) adds no
        protection and reproduces pure LRU exactly -- including on re-admission, where any
        earlier pin on these blocks is cleared, so a block is never left protected past the
        point the router stopped asking for it.

        ``at_index`` is the request's position in the arrival stream; the base LRU node
        ignores it, but eviction strategies that look ahead (see :class:`BeladyNode`) use it
        to know how far the simulation has progressed.
        """
        self._at_index = at_index
        start = max(now, self.busy_until)
        self.busy_until = start + service_time
        pin = retain_until if retain_until > now else None
        for h in req_blocks:
            self._cache.pop(h, None)
            self._cache[h] = None
            if pin is not None:
                self._retain_until[h] = pin
            else:
                self._retain_until.pop(h, None)
        while len(self._cache) > self.cache_capacity:
            self._evict_one(now)

    def _evict_one(self, now: float) -> None:
        """Remove the victim chosen by :meth:`_pick_victim` and drop its side-table state."""
        victim = self._pick_victim(now)
        if victim is None:  # cache empty -- should not happen while over capacity
            return
        self._cache.pop(victim, None)
        self._retain_until.pop(victim, None)

    def _pick_victim(self, now: float) -> int | None:
        """LRU among still-unprotected blocks; if every block is pinned, the one expiring soonest.

        A single left-to-right (oldest-first) scan returns the first unprotected block -- the
        LRU victim, O(1) in the common no-pin case. When every resident block is still
        protected, evict the pin that expires soonest (it frees capacity with the least
        retention value forgone), not the strict-LRU block which may be pinned far longer.
        """
        soonest: int | None = None
        soonest_expiry = float("inf")
        for h in self._cache:
            expiry = self._retain_until.get(h, 0.0)
            if expiry <= now:
                return h
            if expiry < soonest_expiry:
                soonest_expiry = expiry
                soonest = h
        return soonest


class BeladyNode(Node):
    """A node whose eviction approximates Belady's MIN: drop the block whose next use is furthest.

    Overrides only victim *selection* (:meth:`_pick_victim`); insertion, removal, and the
    queue live in the base :class:`Node`. ``admit`` records the current stream position via
    ``at_index`` (see :meth:`Node.admit`), and the precomputed ``future_uses`` map (block
    hash → sorted request indices where it appears) lets the node evict the block whose next
    access is furthest in the future.

    Caveat -- this is a *strong reference baseline*, not a proven optimum. ``future_uses`` is
    built over the whole request stream without regard to which node will serve each future
    request, so a block's "next use" may actually route to a different node's cache. Belady's
    MIN optimality holds for a single reference stream and a single cache; with N per-node
    caches and online routing it is an approximation, and this metric credits only the
    *contiguous* matched prefix (not independent per-block hits) that MIN minimizes. Treat it
    as a much stronger lower-bound *estimate* than the LRU oracle, not a guaranteed bound.
    """

    def __init__(
        self,
        node_id: int,
        cache_blocks: int,
        future_uses: dict[int, list[int]],
    ):
        super().__init__(node_id, cache_blocks)
        self._future_uses = future_uses

    def _next_use(self, block_hash: int) -> float:
        """Request index of the next access to ``block_hash``, or inf if never reused."""
        uses = self._future_uses.get(block_hash)
        if uses is None:
            return float("inf")
        pos = bisect_right(uses, self._at_index)
        if pos >= len(uses):
            return float("inf")
        return float(uses[pos])

    def _pick_victim(self, now: float) -> int | None:
        """The block whose next use is furthest in the future (Belady's MIN)."""
        victim: int | None = None
        victim_next: float = -1.0
        for h in self._cache:
            nu = self._next_use(h)
            if nu == float("inf"):
                return h  # Never used again -- optimal eviction target.
            if nu > victim_next:
                victim_next = nu
                victim = h
        return victim
