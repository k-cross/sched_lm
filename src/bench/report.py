import statistics
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from bench.program_metrics import ProgramMetrics
    from bench.sim.engine import SimResult


def generate_sim_report(results: "dict[str, SimResult]"):
    """Render the offline-simulator policy comparison.

    ``results`` maps a policy name to a ``bench.sim.engine.SimResult``. Unlike the live
    report, this surfaces the sim-only signals: the regime mix (how often each of the three
    routing regimes was chosen), routing regret, TTFT gap, and the coupled fraction (how
    often the optimal placement depended on other nodes' load).

    **Routing regret** is per-policy: each request's TTFT minus the greedy-oracle TTFT on
    that same simulation's node state. It measures how well the policy *routes* given its
    own cache contents — not how good those cache contents are. Both ``oracle`` and
    ``oracle-belady`` show 0.000 because they both route optimally on their own state.

    **TTFT gap** is cross-policy: each policy's p50 TTFT minus the best oracle baseline's
    p50 (``oracle-belady`` if run, else ``oracle``). It is the comparable measure of how far
    a policy is from that reference. ``oracle-belady`` is a strong *estimate* of the best
    achievable, not a proven bound (its Belady eviction is approximate under multi-node
    routing), so a policy with better cache management can post a small **negative** gap --
    read that as "faster than the reference baseline," not an impossibility.
    """
    from bench.metrics import compute_percentiles
    from bench.sim.policies import ORACLE_BASELINES

    console = Console()
    table = Table(title="Offline Prefix-Cache Routing Simulation")

    table.add_column("Policy", justify="left", style="cyan")
    table.add_column("TTFT p50 (s)", justify="right", style="yellow")
    table.add_column("TTFT p99 (s)", justify="right", style="red")
    table.add_column("Hit rate", justify="right", style="blue")
    table.add_column("Regime (wait/xfer/recomp)", justify="center", style="green")
    table.add_column("Routing regret (s)", justify="right", style="magenta")
    table.add_column("TTFT gap (s)", justify="right", style="magenta")
    table.add_column("Coupled", justify="right", style="magenta")
    # Pinned-pressure signal (RFC-0001 §4): marked-and-unexpired blocks evicted under
    # contention. High here is the leading indicator that retention is over-pinning -- worth
    # reading alongside hit rate, since a policy can win hit rate while thrashing pins.
    table.add_column("Pinned evict", justify="right", style="red")

    # Compute each policy's percentiles once, then read the best-available oracle baseline
    # for the cross-policy TTFT gap column from that same table.
    pctiles = {policy: compute_percentiles(res.ttfts) for policy, res in results.items()}
    best_oracle = next((n for n in ORACLE_BASELINES if n in results), None)
    baseline_p50 = pctiles[best_oracle].get(50, 0) if best_oracle is not None else None

    for policy, res in results.items():
        ttfts = pctiles[policy]
        mix = res.regime_mix
        mean_regret = sum(res.regrets) / len(res.regrets) if res.regrets else 0.0
        p50 = ttfts.get(50, 0)
        gap = f"{p50 - baseline_p50:.3f}" if baseline_p50 is not None else "-"
        table.add_row(
            policy,
            f"{p50:.3f}",
            f"{ttfts.get(99, 0):.3f}",
            f"{res.hit_rate * 100:.1f}%",
            f"{mix['wait'] * 100:.0f}/{mix['transfer'] * 100:.0f}/{mix['recompute'] * 100:.0f}",
            f"{mean_regret:.3f}",
            gap,
            f"{res.coupled_fraction * 100:.1f}%",
            str(res.pinned_evictions),
        )

    console.print(table)

    # With a mixed workload, break the same runs down by ground-truth request class so
    # class-blind vs class-aware policies can be compared where they actually differ.
    kinds = sorted({k for res in results.values() for k in res.by_kind})
    if len(kinds) > 1:
        by_kind = Table(title="Per-class breakdown (regret s / TTFT p50 s / hit rate)")
        by_kind.add_column("Policy", justify="left", style="cyan")
        for kind in kinds:
            by_kind.add_column(kind, justify="right")
        for policy, res in results.items():
            cells = []
            for kind in kinds:
                stats = res.by_kind.get(kind)
                if stats is None:
                    cells.append("-")
                    continue
                p50 = compute_percentiles(stats.ttfts).get(50, 0)
                cells.append(f"{stats.mean_regret:.3f} / {p50:.3f} / {stats.hit_rate * 100:.0f}%")
            by_kind.add_row(policy, *cells)
        console.print(by_kind)

    # If the workload modeled tool reliability, show the per-tool re-arrival-gap distribution
    # the reliability-aware policy learns from (mean + std -> which tools are worth keeping
    # cache warm for). It is policy-independent, so take it from any run that has it.
    gap_by_tool = next((res.gap_by_tool for res in results.values() if res.gap_by_tool), None)
    if gap_by_tool:
        tools = Table(title="Observed re-arrival gap per tool (s)")
        tools.add_column("Tool", justify="left", style="cyan")
        tools.add_column("Calls", justify="right")
        tools.add_column("Mean gap", justify="right", style="yellow")
        tools.add_column("Std gap", justify="right", style="red")
        for tool in sorted(gap_by_tool):
            gaps = gap_by_tool[tool]
            std = statistics.pstdev(gaps) if len(gaps) > 1 else 0.0
            tools.add_row(tool, str(len(gaps)), f"{statistics.mean(gaps):.2f}", f"{std:.2f}")
        console.print(tools)


def generate_report(
    results: dict[str, dict[str, float]],
    cache_hit_rates: dict[str, float],
    prefill_times: dict[str, float | None] | None = None,
    pinned_usages: dict[str, float | None] | None = None,
    pinned_evictions: dict[str, float | None] | None = None,
    program_metrics: "dict[str, ProgramMetrics] | None" = None,
):
    prefill_times = prefill_times or {}
    pinned_usages = pinned_usages or {}
    pinned_evictions = pinned_evictions or {}
    program_metrics = program_metrics or {}
    console = Console()
    table = Table(title="Emulated Prefix-Caching Benchmark Results")

    table.add_column("Strategy", justify="left", style="cyan")
    table.add_column("Requests", justify="right", style="magenta")
    table.add_column("Success Rate", justify="right", style="green")
    table.add_column("Req/s", justify="right", style="magenta")
    table.add_column("TTFT p50 (s)", justify="right", style="yellow")
    table.add_column("TTFT p99 (s)", justify="right", style="red")
    table.add_column("E2E p50 (s)", justify="right", style="yellow")
    table.add_column("Sess p50 (s)", justify="right", style="yellow")
    table.add_column("Zero-recomp", justify="right", style="green")
    table.add_column("Cache Hit Rate", justify="right", style="blue")
    table.add_column("Prefill (s)", justify="right", style="magenta")
    # Peak (max_over_time) rather than instant: EPP leases decay within seconds, so an
    # instant read after the settle wait would show ~0 regardless of run-time pressure.
    table.add_column("Pinned peak %", justify="right", style="cyan")
    table.add_column("Pinned evict", justify="right", style="red")

    for strategy, metrics in results.items():
        prefill = prefill_times.get(strategy)
        p_usage = pinned_usages.get(strategy)
        p_evict = pinned_evictions.get(strategy)
        pm = program_metrics.get(strategy)
        table.add_row(
            strategy,
            str(metrics.get("requests", 0)),
            f"{metrics.get('success_rate', 0):.1f}%",
            f"{pm.throughput_rps:.1f}" if pm is not None else "n/a",
            f"{metrics.get('ttft_p50', 0):.3f}",
            f"{metrics.get('ttft_p99', 0):.3f}",
            f"{metrics.get('e2e_p50', 0):.3f}",
            f"{pm.session_p50:.2f}" if pm is not None and pm.session_p50 is not None else "n/a",
            f"{pm.zero_recompute_rate * 100:.0f}%"
            if pm is not None and pm.zero_recompute_rate is not None
            else "n/a",
            f"{cache_hit_rates.get(strategy, 0) * 100:.1f}%",
            f"{prefill:.3f}" if prefill is not None else "n/a",
            f"{p_usage * 100:.1f}%" if p_usage is not None else "n/a",
            f"{p_evict:.0f}" if p_evict is not None else "n/a",
        )

    console.print(table)

    # Per-turn-position table (sessions workload only): flat TTFT across turn positions
    # means prefixes died between turns; decreasing means they survived (RFC-0001 §4).
    with_turns = {s: pm for s, pm in program_metrics.items() if pm.ttft_by_turn}
    if with_turns:
        turn_idxs = sorted({i for pm in with_turns.values() for i in pm.ttft_by_turn})
        turns = Table(title="TTFT by turn position (p50 s / mean prefix coverage)")
        turns.add_column("Strategy", justify="left", style="cyan")
        for idx in turn_idxs:
            turns.add_column(f"t{idx}", justify="right")
        for strategy, pm in with_turns.items():
            cells = []
            for idx in turn_idxs:
                ttft = pm.ttft_by_turn.get(idx)
                cov = pm.coverage_by_turn.get(idx)
                if ttft is None:
                    cells.append("-")
                elif cov is None:
                    cells.append(f"{ttft:.3f}")
                else:
                    cells.append(f"{ttft:.3f} / {cov * 100:.0f}%")
            turns.add_row(strategy, *cells)
        console.print(turns)
