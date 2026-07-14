import asyncio
import time

import click

from bench.metrics import MetricsClient, compute_percentiles
from bench.report import generate_report, generate_sim_report
from bench.sim.policies import DEFAULT_RETAIN_MARGIN, POLICY_NAMES, uses_belady
from bench.traffic import run_session_traffic, run_traffic


@click.group()
def main():
    """Emulated vLLM Prefix-Caching Benchmark Harness"""
    pass


def _run_route(route, requests, qps, concurrency, gateway_url, workload, sessions, turns, seed):
    """Drive one route with either the single-shot or the tool-calling-session workload."""
    if workload == "sessions":
        from bench.sim.workload import generate_sessions

        turn_requests = generate_sessions(sessions, turns, qps, seed=seed)
        return asyncio.run(run_session_traffic(gateway_url, turn_requests, concurrency, route))
    return asyncio.run(run_traffic(gateway_url, requests, concurrency, route, qps))


@main.command()
@click.option("--route", type=click.Choice(["round-robin", "prefix-affinity"]), required=True)
@click.option("--requests", default=100, help="Number of requests to send (single-shot workload)")
@click.option("--qps", default=5.0, help="Target queries per second")
@click.option("--concurrency", default=10, help="Max concurrent requests")
@click.option(
    "--workload",
    type=click.Choice(["single-shot", "sessions"]),
    default="single-shot",
    help="single-shot independent requests, or multi-turn tool-calling sessions",
)
@click.option("--sessions", default=50, help="Sessions to replay (sessions workload)")
@click.option("--turns", default=6, help="Turns per session (sessions workload)")
@click.option("--seed", default=0, help="Workload RNG seed (sessions workload)")
@click.option(
    "--gateway-url", default="http://localhost:8080/v1/chat/completions", help="Gateway URL"
)
def traffic(route, requests, qps, concurrency, workload, sessions, turns, seed, gateway_url):
    """Run synthetic traffic generator against a specific routing strategy"""
    click.echo(
        f"Starting {workload} traffic for route: {route} (qps={qps}, concurrency={concurrency})"
    )

    # `route` selects the gateway HTTPRoute via the x-llmd-route header:
    # `round-robin` bypasses the EPP extProc (default k8s load balancing), while
    # `prefix-affinity` falls through to the EPP-backed default route.
    result = _run_route(
        route, requests, qps, concurrency, gateway_url, workload, sessions, turns, seed
    )

    ttfts = compute_percentiles(result.ttfts)
    e2e = compute_percentiles(result.e2e_latencies)

    click.echo(f"\nCompleted {result.successes} successful requests with {result.errors} errors.")
    click.echo(
        f"TTFT (s) -> p50: {ttfts.get(50, 0):.3f}, "
        f"p90: {ttfts.get(90, 0):.3f}, "
        f"p99: {ttfts.get(99, 0):.3f}"
    )
    click.echo(
        f"E2E  (s) -> p50: {e2e.get(50, 0):.3f}, "
        f"p90: {e2e.get(90, 0):.3f}, "
        f"p99: {e2e.get(99, 0):.3f}"
    )


@main.command()
@click.option("--prometheus-url", default="http://localhost:9090", help="Prometheus URL")
def metrics(prometheus_url):
    """Fetch current cache hit rate from Prometheus"""
    client = MetricsClient(prometheus_url)
    hit_rate = asyncio.run(client.get_prefix_cache_hit_rate())
    click.echo(f"Current Prefix Cache Hit Rate: {hit_rate * 100:.1f}%")


@main.command()
@click.option(
    "--compare",
    multiple=True,
    type=click.Choice(["round-robin", "prefix-affinity"]),
    help="Routes to compare",
)
@click.option("--requests", default=100, help="Number of requests to send per route")
@click.option("--qps", default=5.0, help="Target queries per second")
@click.option("--concurrency", default=10, help="Max concurrent requests")
@click.option(
    "--workload",
    type=click.Choice(["single-shot", "sessions"]),
    default="single-shot",
    help="single-shot independent requests, or multi-turn tool-calling sessions",
)
@click.option("--sessions", default=50, help="Sessions to replay (sessions workload)")
@click.option("--turns", default=6, help="Turns per session (sessions workload)")
@click.option("--seed", default=0, help="Workload RNG seed (sessions workload)")
@click.option(
    "--gateway-url", default="http://localhost:8080/v1/chat/completions", help="Gateway URL"
)
@click.option("--prometheus-url", default="http://localhost:9090", help="Prometheus URL")
@click.option(
    "--settle",
    default=35.0,
    help="Seconds to wait after traffic for Prometheus to scrape (>= scrape interval)",
)
def report(
    compare,
    requests,
    qps,
    concurrency,
    workload,
    sessions,
    turns,
    seed,
    gateway_url,
    prometheus_url,
    settle,
):
    """Run traffic against multiple routes and generate a comparison report"""
    if not compare:
        click.echo("Please provide at least one route to compare using --compare")
        return

    results = {}
    cache_hit_rates = {}
    prefill_times = {}

    metrics_client = MetricsClient(prometheus_url)

    for route in compare:
        click.echo(f"\n--- Running traffic for route: {route} ---")

        # Snapshot the cumulative cache counters before this route's traffic so
        # we can attribute the delta to just this run, rather than a blended
        # rate() window that mixes both routes. Requires a scrape to have
        # captured the pre-traffic state -- it usually has from prior runs.
        hits_before, queries_before = asyncio.run(metrics_client.get_prefix_cache_counters())

        result = _run_route(
            route, requests, qps, concurrency, gateway_url, workload, sessions, turns, seed
        )

        ttfts = compute_percentiles(result.ttfts)
        e2e = compute_percentiles(result.e2e_latencies)

        # Wait at least one scrape interval so Prometheus has the post-traffic
        # counter values, then read the delta for this route.
        click.echo(f"Waiting {settle:.0f}s for Prometheus to scrape post-traffic metrics...")
        time.sleep(settle)
        hits_after, queries_after = asyncio.run(metrics_client.get_prefix_cache_counters())
        prefill = asyncio.run(metrics_client.get_avg_prefill_time())

        queries_delta = queries_after - queries_before
        hits_delta = hits_after - hits_before
        hit_rate = hits_delta / queries_delta if queries_delta > 0 else 0.0

        results[route] = {
            "requests": result.successes + result.errors,
            "success_rate": (result.successes / (result.successes + result.errors) * 100)
            if (result.successes + result.errors) > 0
            else 0,
            "ttft_p50": ttfts.get(50, 0),
            "ttft_p99": ttfts.get(99, 0),
            "e2e_p50": e2e.get(50, 0),
            "e2e_p99": e2e.get(99, 0),
        }
        cache_hit_rates[route] = hit_rate
        prefill_times[route] = prefill

    click.echo("\n")
    generate_report(results, cache_hit_rates, prefill_times)


def _parse_mix(ctx, param, value: str) -> dict[str, float]:
    """Parse --mix 'tool=0.5,rag=0.3,oneshot=0.2' into a validated fraction dict."""
    mix: dict[str, float] = {}
    try:
        for part in value.split(","):
            name, _, frac = part.partition("=")
            mix[name.strip()] = float(frac)
    except ValueError as e:
        raise click.BadParameter(f"expected name=frac[,name=frac...], got {value!r}") from e
    unknown = set(mix) - {"tool", "rag", "oneshot"}
    if unknown:
        raise click.BadParameter(f"unknown classes {sorted(unknown)}; use tool, rag, oneshot")
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise click.BadParameter(f"fractions must sum to 1, got {sum(mix.values()):g}")
    return mix


@main.command()
@click.option(
    "--policy",
    "policies",
    multiple=True,
    type=click.Choice(POLICY_NAMES),
    default=POLICY_NAMES,
    help="Routing policies to compare (repeatable)",
)
@click.option("--nodes", default=4, help="Number of simulated inference nodes")
@click.option("--sessions", default=200, help="Number of tool-calling sessions")
@click.option("--turns", default=6, help="Turns per session")
@click.option("--tool-result-tokens", default=64, help="Tokens injected per tool result")
@click.option("--think-time", default=2.0, help="Inter-turn think time (s)")
@click.option("--qps", default=10.0, help="Aggregate target turn arrival rate")
@click.option(
    "--burst-cv",
    default=1.0,
    help="Coefficient of variation of session inter-arrivals: 1.0 = Poisson, >1 = bursty",
)
@click.option(
    "--mix",
    default="tool=1.0",
    callback=_parse_mix,
    help="Workload class fractions, e.g. tool=0.5,rag=0.3,oneshot=0.2 (must sum to 1)",
)
@click.option("--rag-docs", default=16, help="RAG retrieved-document pool size")
@click.option("--rag-doc-tokens", default=1024, help="Tokens per RAG document")
@click.option(
    "--rag-zipf",
    default=1.1,
    help="RAG doc popularity skew (weight 1/rank^s); lower = flatter, no hot docs",
)
@click.option("--block-size", default=16, help="KV block size in tokens")
@click.option("--bandwidth", default=4e9, help="KV-transfer interconnect bandwidth (bytes/s)")
@click.option("--cache-blocks", default=2048, help="KV cache capacity per node, in blocks")
@click.option(
    "--tool-reliability",
    is_flag=True,
    default=False,
    help="Model per-tool reliability so re-arrival gaps vary by tool (enables the "
    "class-aware-reliability signal)",
)
@click.option(
    "--retain-margin",
    default=DEFAULT_RETAIN_MARGIN,
    help="class-aware-reliability: keep a prefix warm for predicted_gap x this margin",
)
@click.option("--seed", default=0, help="Workload RNG seed")
def simulate(
    policies,
    nodes,
    sessions,
    turns,
    tool_result_tokens,
    think_time,
    qps,
    burst_cv,
    mix,
    rag_docs,
    rag_doc_tokens,
    rag_zipf,
    block_size,
    bandwidth,
    cache_blocks,
    tool_reliability,
    retain_margin,
    seed,
):
    """Offline discrete-event simulation of prefix-cache routing policies.

    Models per-node KV caches, queues, and a KV-transfer interconnect so TTFT derives from
    node state, then compares each policy against an oracle argmin (regret) and reports the
    regime mix (wait / transfer / recompute) and how coupled decisions are to other nodes.
    """
    from bench.sim.cost import CostParams
    from bench.sim.engine import run_simulation
    from bench.sim.policies import build_policy
    from bench.sim.workload import generate_mixed

    params = CostParams(block_size=block_size, bandwidth=bandwidth)
    mix_desc = ",".join(f"{k}={v:g}" for k, v in mix.items())
    click.echo(
        f"Generating {sessions} sessions x {turns} turns "
        f"(qps={qps}, mix={mix_desc}, seed={seed})..."
    )
    requests = generate_mixed(
        sessions,
        turns,
        qps,
        mix=mix,
        rag_docs=rag_docs,
        rag_doc_tokens=rag_doc_tokens,
        rag_zipf=rag_zipf,
        tool_result_tokens=tool_result_tokens,
        think_time=think_time,
        burst_cv=burst_cv,
        tool_reliability=tool_reliability,
        block_size=block_size,
        seed=seed,
    )

    results = {}
    for name in policies:
        click.echo(f"Simulating policy: {name} ({len(requests)} requests across {nodes} nodes)")
        policy = build_policy(name, retain_margin=retain_margin)
        results[name] = run_simulation(
            requests, policy, nodes, cache_blocks, params, belady=uses_belady(name)
        )

    click.echo("\n")
    generate_sim_report(results)


if __name__ == "__main__":
    main()
