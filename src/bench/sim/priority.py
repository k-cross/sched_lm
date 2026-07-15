"""KV-cache retention priority vocabulary (RFC-0001 §1).

On the wire, priorities are strictly numeric: the 0-100 scale plus the -1 ``evict-first``
extension. The named classes below are **documentation aliases only** -- prose, policy
code, and dashboards, never transport. The offline sim's :class:`~bench.sim.node.Node` is
the executable spec for how these ranks evict (RFC-0001 §3).

Rank order under eviction, lowest (dropped first) to highest (retained hardest)::

    evict-first (-1)  <  unmarked (no directive)  <  marked (priority ascending, 0..100)

``evict-first`` is genuinely below-normal -- a third bucket drained *before* the LRU free
list -- which #37003's two-queue evictor cannot express (it can only retain harder than
LRU). ``unmarked`` is the absence of any directive (plain LRU); it is not a stored value.
"""

# Numeric priority levels (RFC-0001 §1 table). NORMAL/unmarked is the absence of a
# directive, represented as ``None`` -- never stored as a priority value.
EVICT_FIRST = -1  # below-normal: drained before the LRU free list.
HIGH = 50  # TTL optional (default 30s); the router's learned-retention level.
PINNED = 100  # TTL mandatory, server-capped.

MIN_PRIORITY = EVICT_FIRST
MAX_PRIORITY = PINNED

# Documentation aliases for prose/dashboards (never the wire). ``None`` -> "normal".
CLASS_NAMES: dict[int, str] = {EVICT_FIRST: "evict-first", HIGH: "high", PINNED: "pinned"}


def class_name(priority: int | None) -> str:
    """Human-readable class alias for a numeric priority (documentation only)."""
    if priority is None:
        return "normal"
    return CLASS_NAMES.get(priority, f"priority-{priority}")


def clamp_priority(priority: int) -> int:
    """Clamp a wire priority into the valid ``[-1, 100]`` range.

    A hint never fails a request (RFC-0001 §1): an out-of-range directive is clamped
    rather than rejected, so a malformed value degrades to the nearest valid rank.
    """
    return max(MIN_PRIORITY, min(MAX_PRIORITY, priority))
