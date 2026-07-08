from rich.console import Console
from rich.table import Table


def generate_sim_report(results: dict):
    """Render the offline-simulator policy comparison.

    ``results`` maps a policy name to a ``bench.sim.engine.SimResult``. Unlike the live
    report, this surfaces the sim-only signals: the regime mix (how often each of the three
    routing regimes was chosen), mean regret vs. the oracle, and the coupled fraction (how
    often the optimal placement depended on other nodes' load).
    """
    from bench.metrics import compute_percentiles

    console = Console()
    table = Table(title="Offline Prefix-Cache Routing Simulation")

    table.add_column("Policy", justify="left", style="cyan")
    table.add_column("TTFT p50 (s)", justify="right", style="yellow")
    table.add_column("TTFT p99 (s)", justify="right", style="red")
    table.add_column("Hit rate", justify="right", style="blue")
    table.add_column("Regime (wait/xfer/recomp)", justify="center", style="green")
    table.add_column("Regret (s)", justify="right", style="magenta")
    table.add_column("Coupled", justify="right", style="magenta")

    for policy, res in results.items():
        ttfts = compute_percentiles(res.ttfts)
        mix = res.regime_mix
        mean_regret = sum(res.regrets) / len(res.regrets) if res.regrets else 0.0
        table.add_row(
            policy,
            f"{ttfts.get(50, 0):.3f}",
            f"{ttfts.get(99, 0):.3f}",
            f"{res.hit_rate * 100:.1f}%",
            f"{mix['wait'] * 100:.0f}/{mix['transfer'] * 100:.0f}/{mix['recompute'] * 100:.0f}",
            f"{mean_regret:.3f}",
            f"{res.coupled_fraction * 100:.1f}%",
        )

    console.print(table)


def generate_report(results: dict, cache_hit_rates: dict, prefill_times: dict | None = None):
    prefill_times = prefill_times or {}
    console = Console()
    table = Table(title="Emulated Prefix-Caching Benchmark Results")

    table.add_column("Strategy", justify="left", style="cyan")
    table.add_column("Requests", justify="right", style="magenta")
    table.add_column("Success Rate", justify="right", style="green")
    table.add_column("TTFT p50 (s)", justify="right", style="yellow")
    table.add_column("TTFT p99 (s)", justify="right", style="red")
    table.add_column("E2E p50 (s)", justify="right", style="yellow")
    table.add_column("Cache Hit Rate", justify="right", style="blue")
    table.add_column("Prefill (s)", justify="right", style="magenta")

    for strategy, metrics in results.items():
        prefill = prefill_times.get(strategy)
        table.add_row(
            strategy,
            str(metrics.get("requests", 0)),
            f"{metrics.get('success_rate', 0):.1f}%",
            f"{metrics.get('ttft_p50', 0):.3f}",
            f"{metrics.get('ttft_p99', 0):.3f}",
            f"{metrics.get('e2e_p50', 0):.3f}",
            f"{cache_hit_rates.get(strategy, 0) * 100:.1f}%",
            f"{prefill:.3f}" if prefill is not None else "n/a",
        )

    console.print(table)
