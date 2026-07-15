"""A simulated inference node: a KV-block cache plus a single-server queue.

The two pieces of state that drive routing live here. ``busy_until`` is when the node
will next be free (its queue tail), and ``cache`` is the set of KV block hashes currently
resident, ordered by recency so we can evict under a capacity bound. The capacity bound
is what makes think-time matter: while a session pauses between turns, other traffic can
push its affinity prefix out of the cache, turning a cheap wait into a recompute.

Eviction follows the RFC-0001 §3 priority model, of which this file is the executable
spec. Each resident block carries an optional *retention directive* -- a numeric priority
(:mod:`bench.sim.priority`) plus a wall-clock expiry -- and the victim is chosen by
effective rank ``evict-first < unmarked < priority ascending`` (TTL expiry collapses a
mark to unmarked), LRU position breaking ties. Unmarked blocks pay no bookkeeping, so a
workload that never sets a directive is plain LRU at the same cost as before.
"""

from bisect import bisect_right
from collections import OrderedDict

from bench.sim.blocks import matched_prefix
from bench.sim.priority import HIGH, clamp_priority


class Node:
    def __init__(self, node_id: int, cache_blocks: int):
        self.node_id = node_id
        # When the node's single server next becomes free (== queue tail time).
        self.busy_until = 0.0
        # Resident KV blocks as an LRU: keys are block hashes, insertion order = recency.
        self._cache: OrderedDict[int, None] = OrderedDict()
        # Retention directive per marked block (RFC-0001 §1): numeric priority and a
        # wall-clock expiry after which the mark collapses to unmarked. Only blocks with a
        # live directive appear here; absence == unmarked == plain LRU (zero bookkeeping).
        self._priority: dict[int, int] = {}
        self._expiry: dict[int, float] = {}
        # Marked-and-unexpired blocks evicted under capacity pressure -- the router's
        # signal that it over-pinned (RFC-0001 §4, vllm:kv_cache_pinned_evictions_total).
        self.pinned_evictions = 0
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
        priority: int | None = None,
    ) -> None:
        """Advance the queue and insert the request's blocks as most-recently-used.

        Service is FIFO on one server: a request starts at ``max(now, busy_until)`` and
        the node is busy for ``service_time`` after that. All of the request's prefix
        blocks become resident (recompute or transfer both leave them cached), evicting
        blocks past capacity per the RFC-0001 §3 priority model.

        The retention directive on these blocks is ``(priority, retain_until)``:

        * ``priority=None`` (the default) keeps the legacy soft-pin behavior: a
          ``retain_until`` in the future marks the blocks at :data:`~bench.sim.priority.HIGH`
          with that expiry, so a session predicted to return soon keeps its prefix warm;
          ``retain_until <= now`` marks nothing and reproduces pure LRU exactly.
        * an explicit ``priority`` marks the blocks at that rank (``-1`` evict-first through
          ``100`` pinned); ``retain_until`` in the future is the lease, otherwise the mark
          is persistent (``inf`` -- server-capped in a real deployment).

        Either way, re-admission overwrites the prior directive on these blocks -- clearing
        it entirely on the unmarked path -- so a block is never left marked past the point
        the router stopped asking for it (mirrors the offline pin-clearing rule).

        ``at_index`` is the request's position in the arrival stream; the base node ignores
        it, but lookahead eviction strategies (see :class:`BeladyNode`) use it to know how
        far the simulation has progressed.
        """
        self._at_index = at_index
        start = max(now, self.busy_until)
        self.busy_until = start + service_time
        prio, expiry = self._directive(now, retain_until, priority)
        for h in req_blocks:
            self._cache.pop(h, None)
            self._cache[h] = None
            if prio is None:
                self._priority.pop(h, None)
                self._expiry.pop(h, None)
            else:
                self._priority[h] = prio
                self._expiry[h] = expiry
        while len(self._cache) > self.cache_capacity:
            self._evict_one(now)

    @staticmethod
    def _directive(
        now: float, retain_until: float, priority: int | None
    ) -> tuple[int | None, float]:
        """Resolve ``admit`` args into a stored ``(priority, expiry)`` mark, or unmarked.

        ``(None, _)`` means clear any directive (plain LRU). The legacy ``priority=None``
        path treats a future ``retain_until`` as a :data:`~bench.sim.priority.HIGH` lease;
        an explicit priority is clamped to ``[-1, 100]`` and, absent a future
        ``retain_until``, marks persistently (``inf``).
        """
        if priority is None:
            if retain_until > now:
                return HIGH, retain_until
            return None, 0.0
        expiry = retain_until if retain_until > now else float("inf")
        return clamp_priority(priority), expiry

    def _evict_one(self, now: float) -> None:
        """Remove the victim chosen by :meth:`_pick_victim` and drop its side-table state."""
        victim = self._pick_victim(now)
        if victim is None:  # cache empty -- should not happen while over capacity
            return
        self._cache.pop(victim, None)
        self._priority.pop(victim, None)
        self._expiry.pop(victim, None)

    def _pick_victim(self, now: float) -> int | None:
        """Lowest effective rank, LRU position breaking ties (RFC-0001 §3).

        Effective rank orders ``evict-first < unmarked < priority ascending``, with TTL
        expiry collapsing a mark to unmarked. A single oldest-first scan finds, per bucket,
        the LRU (first-seen) evict-first and unmarked block and the lowest-priority /
        soonest-expiring marked block; the lowest non-empty bucket wins. When every
        resident block is marked-and-unexpired we must sacrifice a pin -- the
        lowest-priority, soonest-expiring one -- and record the pressure (§4).

        When no block carries a directive this is a plain LRU eviction of the oldest block,
        short-circuited so the common non-agentic path stays O(1) with no per-block checks.
        """
        if not self._priority:  # no live directives -> plain LRU, evict the oldest.
            return next(iter(self._cache), None)

        evict_first: int | None = None
        unmarked: int | None = None
        marked: int | None = None
        marked_key: tuple[int, float] | None = None
        for h in self._cache:  # oldest -> newest, so first-seen == LRU within a bucket
            prio = self._priority.get(h)
            expiry = self._expiry.get(h, 0.0)
            if prio is None or expiry <= now:  # unmarked, or expired -> collapses to unmarked
                if unmarked is None:
                    unmarked = h
            elif prio < 0:
                if evict_first is None:
                    evict_first = h
            else:
                key = (prio, expiry)  # lowest priority first, then soonest-expiring
                if marked_key is None or key < marked_key:
                    marked_key = key
                    marked = h

        if evict_first is not None:
            return evict_first
        if unmarked is not None:
            return unmarked
        if marked is not None:
            self.pinned_evictions += 1
        return marked


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
