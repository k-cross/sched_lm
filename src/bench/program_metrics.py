"""Program-level (PoC 3) aggregations over a live benchmark run (RFC-0001 §4).

Pure functions over :class:`~bench.traffic.BenchmarkResult`: no I/O, fully unit-testable.
The three headline reads for the phase-5 evidence:

  * **TTFT by turn position** — flat across turns means the session's prefix was evicted
    between turns (cache failure); decreasing means the growing prefix survived.
  * **Zero-recompute rate** — the fraction of returning turns whose server-reported
    ``cached_tokens`` cover everything the previous turn actually prefilled.
  * **Session completion time** — wall time of the whole program, the metric agentic
    schedulers ultimately optimize.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from bench.metrics import compute_percentiles
from bench.traffic import BenchmarkResult, TurnMetric

DEFAULT_BLOCK_SIZE = 16


@dataclass(frozen=True)
class ProgramMetrics:
    """Aggregated program-level measures of one route's run (None/{} when inapplicable)."""

    throughput_rps: float
    session_p50: float | None
    session_p90: float | None
    zero_recompute_rate: float | None  # returning (turn>0) turns only
    ttft_by_turn: dict[int, float]  # turn_idx -> p50 TTFT (s)
    coverage_by_turn: dict[int, float]  # turn_idx -> mean cached_tokens / prompt_tokens


def _zero_recompute_rate(turns: list[TurnMetric], block_size: int) -> float | None:
    """Fraction of returning turns whose cached prefix covers the previous turn's prompt.

    Turn *k* replays turn *k-1*'s prompt as a strict prefix, so everything the previous
    turn prefilled should still be cached: ``cached_tokens_k >= floor(prompt_{k-1} /
    block) * block`` (the cache is block-granular, so round down). The previous turn's
    *completion* is excluded on purpose -- the workload scripts its own assistant text,
    which the server never generated, so that segment can never be cached.
    """
    by_session: dict[int, list[TurnMetric]] = defaultdict(list)
    for t in turns:
        by_session[t.session_id].append(t)

    hits = total = 0
    for session_turns in by_session.values():
        session_turns.sort(key=lambda t: t.turn_idx)
        for prev, cur in zip(session_turns, session_turns[1:], strict=False):
            # Consecutive turns only: a gap means the earlier turn errored, and usage-less
            # turns (prompt_tokens == 0) cannot be judged.
            if cur.turn_idx != prev.turn_idx + 1 or prev.prompt_tokens <= 0:
                continue
            total += 1
            if cur.cached_tokens >= (prev.prompt_tokens // block_size) * block_size:
                hits += 1
    return hits / total if total else None


def compute_program_metrics(
    result: BenchmarkResult, block_size: int = DEFAULT_BLOCK_SIZE
) -> ProgramMetrics:
    throughput = result.successes / result.wall_seconds if result.wall_seconds > 0 else 0.0

    session_p50 = session_p90 = None
    if result.session_times:
        pcts = compute_percentiles(list(result.session_times.values()), [50, 90])
        session_p50, session_p90 = pcts[50], pcts[90]

    ttfts_by_turn: dict[int, list[float]] = defaultdict(list)
    coverage: dict[int, list[float]] = defaultdict(list)
    for t in result.turns:
        ttfts_by_turn[t.turn_idx].append(t.ttft)
        if t.prompt_tokens > 0:
            coverage[t.turn_idx].append(t.cached_tokens / t.prompt_tokens)

    return ProgramMetrics(
        throughput_rps=throughput,
        session_p50=session_p50,
        session_p90=session_p90,
        zero_recompute_rate=_zero_recompute_rate(result.turns, block_size),
        ttft_by_turn={
            idx: compute_percentiles(vals, [50])[50] for idx, vals in sorted(ttfts_by_turn.items())
        },
        coverage_by_turn={idx: mean(vals) for idx, vals in sorted(coverage.items())},
    )
