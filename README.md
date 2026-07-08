# llm-d-emulation-bench

A local benchmark harness for studying **LLM inference routing / prefix-cache
scheduling** — entirely on a single Apple Silicon Mac, no GPU or cloud spend
required.

## What this is (and why)

Multi-replica LLM serving lives or dies on one question: *when a request's
prompt prefix is already cached on some node, where should the request go?*
There are three competing answers, and picking wrong is expensive:

1. **Wait** for the node that has the prefix cached, even if it's busy.
2. **Transfer** the cached KV blocks to an idle node over the interconnect.
3. **Recompute** the prefix from scratch on an idle node.

Real vLLM + a real GPU cluster makes this expensive to iterate on. This repo
gives you two ways to study the same question cheaply:

- **A live stack** — `k3d` + the real [`llm-d`](https://github.com/llm-d)
  Gateway/EPP router in front of `llm-d-inference-sim` replicas (a lightweight
  stand-in for vLLM that reproduces its Prometheus prefix-cache metrics and KV
  events without needing a GPU). This exercises the *actual* routing brain,
  but the simulator can only express "wait" vs. "recompute" — it has no
  KV-transfer path.
- **An offline discrete-event simulator** (`src/bench/sim/`) — a pure-Python
  model of N nodes, each with a KV-block LRU cache and a queue, plus a
  KV-transfer interconnect. TTFT is *derived* from that state rather than
  hard-coded, so all three regimes are observable, and you can compare routing
  **policies** (including an optimal oracle) on identical synthetic traffic in
  well under a second — no cluster needed.

Both paths share the same synthetic workload: multi-turn, tool-calling
sessions built around a realistic ~4K-token shared system prompt, so the
prefix-sharing "shape" of the traffic is representative of a real
tool-using assistant rather than one-shot unrelated prompts.

## Architecture

### Live stack

1.  **k3d Cluster** — a lightweight Kubernetes cluster running locally on Docker.
2.  **llm-d Gateway (kgateway)** — Envoy-based API Gateway handling incoming traffic.
3.  **llm-d Endpoint Picker (EPP)** — the routing brain; makes decisions based on prefix-cache state and backend load.
4.  **vLLM Simulators** — 2-4 replica pods running `llm-d-inference-sim`, emulating vLLM's prefix-caching behavior and exposing ZeroMQ KV-cache events + Prometheus metrics.
5.  **Traffic Generator** — the `bench` CLI, an `aiohttp` client that drives synthetic workload and measures TTFT (Time To First Token).

### Offline simulator (`src/bench/sim/`)

| Module | Role |
|---|---|
| `blocks.py` | vLLM-style rolling block-hash chain over tokens — the unit prefix caching actually matches on. |
| `node.py` | A `Node`: KV-block LRU cache (capacity-bounded, so a paused session can be evicted) + a single-server queue (`busy_until`). |
| `cost.py` | The cost model: predicted TTFT for `(request, node, transfer?)` = `wait + transfer_time + prefill(missing)`. The `argmin` over this *is* the wait/transfer/recompute decision. |
| `workload.py` | Generates seedable, multi-turn tool-calling sessions with a growing shared prefix. |
| `policies.py` | Routing policies compared against each other and against the oracle. **This is where you plug in a new policy — see below.** |
| `engine.py` | Discrete-event loop: replays a workload under a policy, records TTFT, regime mix, regret vs. oracle, and a "coupling" metric. |

## Prerequisites

*   **OrbStack** (recommended for Apple Silicon), **Docker Desktop**, or **Colima** — only needed for the live stack, not the offline simulator.
*   **Nix**: install via Determinate Systems if not present: `curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s install`
*   **direnv** (recommended): so the `devenv` shell auto-activates when you `cd` into the repo. Without it, run `devenv shell` manually each session.

## Getting Started

New to the repo? The offline simulator needs no cluster and is the fastest way to see something work:

```bash
git clone <this-repo>
cd sched_lm
direnv allow        # or: devenv shell
# uv sync runs automatically on shell entry
uv run bench simulate --nodes 4 --sessions 200 --turns 6 --qps 10
```

That prints a comparison table across policies (round-robin, cache-local,
transfer-aware, oracle) with no Docker/Kubernetes involved. See
[Offline simulation](#offline-simulation-no-cluster-needed) below for what the
columns mean.

### Live stack

To exercise the real `llm-d` router:

1.  **Enter the Development Environment:**
    ```bash
    devenv shell
    ```
    *(Note: The first time you run this, Nix will download and configure `k3d`, `kubectl`, `helm`, and Python with `uv`. `KUBECONFIG` is set to `.k3d/kubeconfig.yaml`.)*

2.  **Create the Kubernetes Cluster:**
    ```bash
    cluster-create
    ```

3.  **Deploy the `llm-d` Stack & Simulators:**
    ```bash
    deploy-llmd
    ```

4.  **Deploy Monitoring (Prometheus/Grafana):**
    ```bash
    deploy-monitoring
    ```

Other cluster lifecycle scripts: `cluster-status`, `cluster-delete`.

## Benchmarking

### Live stack

Once the cluster and simulators are running, use the `bench` CLI to generate
traffic and test routing strategies:

```bash
# Test with round-robin routing (baseline)
uv run bench traffic --route round-robin --requests 200 --qps 10

# Test with prefix-affinity routing (optimized)
uv run bench traffic --route prefix-affinity --requests 200 --qps 10

# Generate a comparison report
uv run bench report --compare round-robin --compare prefix-affinity --qps 10

# Same, but replay multi-turn tool-calling sessions instead of one-shot requests
uv run bench report --compare round-robin --compare prefix-affinity --workload sessions --qps 10
```

`--route` selects the gateway path via an `x-llmd-route` header: `round-robin`
hits an HTTPRoute with no EPP extProc (default k8s load balancing across the
sim pods), while `prefix-affinity` uses the EPP-backed default route that
steers by KV-cache state. `--qps` paces request arrivals open-loop;
`--concurrency` caps in-flight requests. `--workload sessions` replays the
same tool-calling session generator used by the offline simulator
(`--sessions`, `--turns`, `--seed`), so you can compare how the real EPP
handles a growing, tool-call-shaped prefix against the offline oracle's
predictions.

The `report` command attributes prefix-cache hit rate to each route by reading
the counter delta around that route's traffic, so it waits `--settle` seconds
(default 35s, ≥ the Prometheus scrape interval) after each run before reading
metrics.

### Offline simulation (no cluster needed)

```bash
uv run bench simulate \
  --policy round-robin --policy cache-local --policy transfer-aware --policy oracle \
  --nodes 4 --sessions 200 --turns 6 --qps 10 --seed 1
```

Key options: `--nodes`, `--sessions`, `--turns`, `--tool-result-tokens`
(controls how many unique tokens each tool call injects, i.e. how much prefix
sharing survives), `--think-time` (inter-turn gap, races against cache
eviction), `--block-size`, `--bandwidth` (interconnect speed — lower it to see
the oracle abandon the transfer regime in favor of recompute), `--cache-blocks`
(per-node KV capacity), `--seed`.

The report columns:

| Column | Meaning |
|---|---|
| TTFT p50/p99 | Predicted time-to-first-token under this policy. |
| Hit rate | Fraction of prefix blocks served from cache vs. recomputed. |
| Regime (wait/xfer/recomp) | % of requests resolved by each of the three regimes. |
| Regret (s) | Mean `policy_TTFT − oracle_TTFT` per request; 0 for the oracle by definition. |
| Coupled | % of requests whose optimal placement changes once every *other* node is forced idle — i.e. how much this policy's decisions depend on cluster-wide state rather than the request alone. |

Try sweeping `--qps` (load) and `--bandwidth` (interconnect speed) — regret for
naive policies and the coupled fraction should both climb as contention rises,
and the regime mix should shift from transfer toward recompute as bandwidth
gets scarce.

## Contributing a routing/scheduling policy

The offline simulator is designed so a new policy is a small, self-contained
addition that gets measured automatically — no engine or CLI changes needed.

1.  Open `src/bench/sim/policies.py`. A policy is any callable matching the
    `Policy` signature:

    ```python
    def my_policy(
        req_blocks: list[int], nodes: list[Node], now: float, params: CostParams
    ) -> Placement:
        ...
    ```

    You have access to each node's `matched(req_blocks)` (cached prefix
    length), `wait(now)` (queue delay), and the shared `predict(...)` /
    `best_placement(...)` helpers in `cost.py` if you want to score candidates
    the same way the oracle does. Return a `Placement(node_id, transfer,
    regime, ttft)` — see `cache_local` or `transfer_aware` for reference
    implementations, including one (`RoundRobin`) that needs to carry state
    across calls.

2.  Register it in `build_policy()` and add its name to `POLICY_NAMES` in the
    same file. That's the only wiring required — `bench simulate --policy
    <your-name>` will pick it up immediately, and it's automatically compared
    against the oracle (regret) and scored for cluster coupling.

3.  Add a test in `tests/test_sim.py` alongside the existing policy tests —
    at minimum assert your policy never produces negative regret, and add a
    scenario where you can predict what it should do (e.g. "with a single hot,
    idle node, this policy should match the oracle").

4.  Run the comparison to see where it lands:

    ```bash
    uv run bench simulate --policy round-robin --policy cache-local \
      --policy transfer-aware --policy oracle --policy <your-name> \
      --nodes 4 --sessions 200 --turns 6 --qps 10 --seed 1
    ```

    Sweep `--qps`, `--think-time`, and `--bandwidth` to see how your policy's
    regret and coupled fraction respond to load, cache-eviction pressure, and
    interconnect speed relative to the existing baselines.

If your policy needs a genuinely different cost signal (e.g. a predictive
model instead of `CostParams`'s closed-form estimate), it can still return
`Placement`s from `cost.predict()`/`best_placement()`'s node/transfer
candidates — the engine only depends on the `Policy` call signature and the
`Placement` shape, not on how the decision was made internally.

For the **live stack**, new routing behavior belongs in the `llm-d`/EPP layer
itself (`infra/llm-d/values.yaml`, `infra/llm-d/inference-pool.yaml`) rather
than in this repo's Python — the offline simulator is the fast iteration loop
for policy ideas; promote a validated idea to a real EPP change once you're
convinced by the offline numbers.

## Tests

```bash
uv run pytest                        # unit tests: traffic generator + sim (blocks/cost/policies/workload/engine)
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Metrics

You can view real-time metrics by querying the Prometheus instance, or via the helper command:
```bash
uv run bench metrics
```

## Cleaning Up

To tear down the cluster and clean up resources:
```bash
cluster-delete
```
