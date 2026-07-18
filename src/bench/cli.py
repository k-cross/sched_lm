import asyncio
import re
import time

import click

from bench.metrics import MetricsClient, compute_percentiles
from bench.report import generate_report, generate_sim_report
from bench.sim.policies import DEFAULT_RETAIN_MARGIN, POLICY_NAMES, uses_belady
from bench.sim.priority import MAX_PRIORITY, MIN_PRIORITY
from bench.traffic import run_session_traffic, run_traffic

# Go duration grammar (a sequence of signed decimal number + unit), the form the forked
# sim's ttl= parses. Validated client-side so a typo is rejected instead of silently
# falling back to the server's default lease.
_GO_DURATION_RE = re.compile(r"^\d+(\.\d+)?(ns|us|µs|ms|s|m|h)([0-9.]+(ns|us|µs|ms|s|m|h))*$")

ONLINE_PRESETS = {
    "fast": {
        "workload": "sessions",
        "sessions": 5,
        "turns": 3,
        "qps": 2.0,
        "concurrency": 2,
    },
    "experiment": {
        "workload": "sessions",
        "sessions": 50,
        "turns": 6,
        "qps": 10.0,
        "concurrency": 20,
        "tool_reliability": True,
    },
}

OFFLINE_PRESETS = {
    "fast": {
        "sessions": 10,
        "turns": 3,
        "qps": 5.0,
        "nodes": 2,
    },
    "experiment": {
        "sessions": 200,
        "turns": 6,
        "qps": 10.0,
        "mix": {"tool": 0.5, "rag": 0.3, "oneshot": 0.2},
        "tool_reliability": True,
        "burst_cv": 1.5,
    },
}


def _apply_preset(ctx, preset: str, presets: dict):
    if not preset:
        return
    p = presets.get(preset)
    if not p:
        return
    for k, v in p.items():
        if k in ctx.params and ctx.get_parameter_source(k) == click.core.ParameterSource.DEFAULT:
            ctx.params[k] = v


@click.group()
def main():
    """Emulated vLLM Prefix-Caching Benchmark Harness"""
    pass


def _kv_priority_header(priority, ttl, scope):
    """Build the RFC-0001 x-kv-cache-priority header value, or None when no priority is set.

    Grammar (router path, §2): ``<int>[; ttl=<Go duration>][; scope=<id>]``. The forked
    llm-d-inference-sim parses this; ttl is a Go duration string (e.g. ``3s``, ``500ms``).
    Raises ``click.UsageError`` when ttl/scope are given without a priority (they would be
    silently dropped) or when ttl is not a valid Go duration.
    """
    if priority is None:
        if ttl or scope:
            raise click.UsageError("--kv-ttl/--kv-scope require --kv-priority")
        return None
    if ttl and not _GO_DURATION_RE.match(ttl):
        raise click.UsageError(f"--kv-ttl {ttl!r} is not a Go duration (e.g. 3s, 500ms, 1m30s)")
    value = str(priority)
    if ttl:
        value += f"; ttl={ttl}"
    if scope:
        value += f"; scope={scope}"
    return value


def _run_route(
    route,
    requests,
    qps,
    concurrency,
    gateway_url,
    workload,
    sessions,
    turns,
    seed,
    kv_priority=None,
    tool_reliability=False,
    session_offset=0,
):
    """Drive one route with either the single-shot or the tool-calling-session workload.

    ``session_offset`` shifts session ids (and thus per-session prompt text) so multiple
    arms of one report run replay identical arrival streams over disjoint prefixes --
    without it, the second arm warm-starts from the first arm's cache.
    """
    if workload == "sessions":
        from bench.sim.workload import generate_sessions

        turn_requests = generate_sessions(
            sessions,
            turns,
            qps,
            seed=seed,
            tool_reliability=tool_reliability,
            session_id_offset=session_offset,
        )
        return asyncio.run(
            run_session_traffic(gateway_url, turn_requests, concurrency, route, kv_priority)
        )
    return asyncio.run(run_traffic(gateway_url, requests, concurrency, route, qps, kv_priority))


@main.command()
@click.option(
    "--route",
    type=click.Choice(["round-robin", "prefix-affinity", "class-aware-reliability"]),
    required=True,
)
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
@click.option(
    "--kv-priority",
    type=click.IntRange(MIN_PRIORITY, MAX_PRIORITY),
    default=None,
    help=f"RFC-0001 escape hatch: send x-kv-cache-priority on every request "
    f"({MIN_PRIORITY} evict-first .. {MAX_PRIORITY} pinned). Proves the directive path "
    f"to the forked sim.",
)
@click.option("--kv-ttl", default=None, help="Lease for --kv-priority, a Go duration (e.g. 3s)")
@click.option("--kv-scope", default=None, help="Session/program scope for --kv-priority")
@click.option(
    "--tool-reliability",
    is_flag=True,
    default=False,
    help="Model per-tool reliability in the sessions workload: gaps vary by tool and "
    "assistant messages carry OpenAI tool_calls, giving the EPP's kv-cache-priority "
    "plugin a tool signature to learn per-tool re-arrival gaps from (RFC-0001 §5).",
)
@click.option("--preset", type=click.Choice(["fast", "experiment"]), help="Apply preset defaults")
@click.pass_context
def traffic(
    ctx,
    route,
    requests,
    qps,
    concurrency,
    workload,
    sessions,
    turns,
    seed,
    gateway_url,
    kv_priority,
    kv_ttl,
    kv_scope,
    tool_reliability,
    preset,
):
    """Run synthetic traffic generator against a specific routing strategy"""
    _apply_preset(ctx, preset, ONLINE_PRESETS)
    qps = ctx.params["qps"]
    concurrency = ctx.params["concurrency"]
    workload = ctx.params["workload"]
    sessions = ctx.params["sessions"]
    turns = ctx.params["turns"]
    tool_reliability = ctx.params["tool_reliability"]

    click.echo(
        f"Starting {workload} traffic for route: {route} (qps={qps}, concurrency={concurrency})"
    )

    kv_priority_header = _kv_priority_header(kv_priority, kv_ttl, kv_scope)
    if kv_priority_header is not None:
        click.echo(f"Attaching x-kv-cache-priority: {kv_priority_header}")

    # `route` selects the gateway HTTPRoute via the x-llmd-route header:
    # `round-robin` bypasses the EPP extProc (default k8s load balancing), while
    # `prefix-affinity` falls through to the EPP-backed default route.
    result = _run_route(
        route,
        requests,
        qps,
        concurrency,
        gateway_url,
        workload,
        sessions,
        turns,
        seed,
        kv_priority_header,
        tool_reliability=tool_reliability,
    )

    ttfts = compute_percentiles(result.ttfts)
    e2e = compute_percentiles(result.e2e_latencies)

    click.echo(f"\nCompleted {result.successes} successful requests with {result.errors} errors.")
    if result.errors:
        click.echo(f"  errors: {result.error_summary()}")
        if any("load shed" in r for r in result.error_reasons):
            click.echo(
                "  note: the EPP-backed routes apply real admission control against the "
                "live replicas; lower --qps or scale the sim deployment for high-load runs."
            )
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
    type=click.Choice(["round-robin", "prefix-affinity", "class-aware-reliability"]),
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
@click.option(
    "--tool-reliability",
    is_flag=True,
    default=False,
    help="Model per-tool reliability in the sessions workload (see `bench traffic`); "
    "required for the EPP's kv-cache-priority plugin to emit directives during a report "
    "run -- without tool_calls on the wire it has nothing to learn per-tool gaps from.",
)
@click.option("--preset", type=click.Choice(["fast", "experiment"]), help="Apply preset defaults")
@click.pass_context
def report(
    ctx,
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
    tool_reliability,
    preset,
):
    """Run traffic against multiple routes and generate a comparison report"""
    from bench.program_metrics import compute_program_metrics

    _apply_preset(ctx, preset, ONLINE_PRESETS)
    requests = ctx.params["requests"]
    qps = ctx.params["qps"]
    concurrency = ctx.params["concurrency"]
    workload = ctx.params["workload"]
    sessions = ctx.params["sessions"]
    turns = ctx.params["turns"]
    tool_reliability = ctx.params["tool_reliability"]

    if not compare:
        click.echo("Please provide at least one route to compare using --compare")
        return
    if len(set(compare)) != len(compare):
        # Per-arm results are keyed by route name, so a repeated route would silently
        # clobber the earlier arm's row. Reject it rather than drop data.
        raise click.UsageError("--compare routes must be distinct")

    results = {}
    cache_hit_rates = {}
    prefill_times = {}
    pinned_usages = {}
    pinned_evictions_map = {}
    program_metrics = {}

    metrics_client = MetricsClient(prometheus_url)

    for arm_index, route in enumerate(compare):
        click.echo(f"\n--- Running traffic for route: {route} ---")

        # Snapshot the cumulative cache counters before this route's traffic so
        # we can attribute the delta to just this run, rather than a blended
        # rate() window that mixes both routes. Requires a scrape to have
        # captured the pre-traffic state -- it usually has from prior runs.
        hits_before, queries_before = asyncio.run(metrics_client.get_prefix_cache_counters())
        evictions_before = asyncio.run(metrics_client.get_pinned_evictions())
        run_started = time.monotonic()

        result = _run_route(
            route,
            requests,
            qps,
            concurrency,
            gateway_url,
            workload,
            sessions,
            turns,
            seed,
            tool_reliability=tool_reliability,
            # Same seed, disjoint session ids per arm: identical arrival streams over
            # unique prompt text, so arm N+1 cannot replay arm N's cached prefixes.
            session_offset=arm_index * sessions * 100,
        )

        ttfts = compute_percentiles(result.ttfts)
        e2e = compute_percentiles(result.e2e_latencies)

        # Wait at least one scrape interval so Prometheus has the post-traffic
        # counter values, then read the delta for this route.
        click.echo(f"Waiting {settle:.0f}s for Prometheus to scrape post-traffic metrics...")
        time.sleep(settle)
        hits_after, queries_after = asyncio.run(metrics_client.get_prefix_cache_counters())
        prefill = asyncio.run(metrics_client.get_avg_prefill_time())
        evictions_after = asyncio.run(metrics_client.get_pinned_evictions())
        # Peak over this arm's whole window (traffic + settle): EPP leases decay within
        # seconds, so an instant post-settle gauge read would miss the run-time pressure.
        window = int(time.monotonic() - run_started) + 1
        pinned_peak = asyncio.run(metrics_client.get_peak_pinned_usage(window))

        queries_delta = queries_after - queries_before
        hits_delta = hits_after - hits_before
        hit_rate = hits_delta / queries_delta if queries_delta > 0 else 0.0
        evictions_delta = evictions_after - evictions_before

        results[route] = {
            "requests": result.successes + result.errors,
            "success_rate": (result.successes / (result.successes + result.errors) * 100)
            if (result.successes + result.errors) > 0
            else 0,
            "ttft_p50": ttfts.get(50, 0),
            "ttft_p99": ttfts.get(99, 0),
            "e2e_p50": e2e.get(50, 0),
        }
        cache_hit_rates[route] = hit_rate
        prefill_times[route] = prefill
        pinned_usages[route] = pinned_peak
        pinned_evictions_map[route] = evictions_delta
        program_metrics[route] = compute_program_metrics(result)
        if result.errors:
            click.echo(f"  errors: {result.error_summary()}")

    click.echo("\n")
    generate_report(
        results,
        cache_hit_rates,
        prefill_times,
        pinned_usages,
        pinned_evictions_map,
        program_metrics,
    )


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
@click.option("--preset", type=click.Choice(["fast", "experiment"]), help="Apply preset defaults")
@click.pass_context
def simulate(
    ctx,
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
    preset,
):
    """Offline discrete-event simulation of prefix-cache routing policies.

    Models per-node KV caches, queues, and a KV-transfer interconnect so TTFT derives from
    node state, then compares each policy against an oracle argmin (regret) and reports the
    regime mix (wait / transfer / recompute) and how coupled decisions are to other nodes.
    """
    _apply_preset(ctx, preset, OFFLINE_PRESETS)
    nodes = ctx.params["nodes"]
    sessions = ctx.params["sessions"]
    turns = ctx.params["turns"]
    qps = ctx.params["qps"]
    burst_cv = ctx.params["burst_cv"]
    mix = ctx.params["mix"]
    tool_reliability = ctx.params["tool_reliability"]

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
