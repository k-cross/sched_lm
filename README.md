# llm-d-emulation-bench

Benchmark harness for LLM inference routing / prefix-cache scheduling. Runs on
a single Apple Silicon Mac — no GPU or cloud spend.

## The question

When a request's prompt prefix is already cached on some node, where should it
go? Three answers, and the wrong one is expensive:

1. **Wait** for the node holding the prefix, even if busy.
2. **Transfer** the cached KV blocks to an idle node over the interconnect.
3. **Recompute** the prefix from scratch on an idle node.

Two ways to study it here:

- **Live stack** — `k3d` + the real [`llm-d`](https://github.com/llm-d)
  Gateway/EPP router in front of `llm-d-inference-sim` replicas (a GPU-free
  vLLM stand-in reproducing its prefix-cache metrics + KV events). Exercises the
  actual router, but the simulator has no KV-transfer path (wait vs. recompute
  only).
- **Offline simulator** (`src/bench/sim/`) — pure-Python model of N nodes, each
  with a KV-block LRU cache, a queue, and a KV-transfer interconnect. TTFT is
  *derived* from node state, so all three regimes are observable and policies
  (including an optimal oracle) compare on identical traffic in under a second.

Both share one workload: multi-turn tool-calling sessions on a ~4K-token shared
system prompt, so prefix sharing looks like a real tool-using assistant.

## Related simulators

None expose the wait/transfer/recompute decision as a cheap policy experiment
with an oracle to measure regret against — the one job `src/bench/sim/` has:

| Simulator | Models | Gap |
|---|---|---|
| [llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim) | Fake vLLM pod: API + prefix-cache metrics + KV events. | No KV transfer — wait vs. recompute only. (Used by our live stack.) |
| [Vidur](https://github.com/microsoft/vidur) | Operator-level engine sim, multi-replica, basic routing. | Replica-centric, no inter-node transfer; needs GPU profiling traces. |
| [LLMServingSim2.0](https://arxiv.org/abs/2511.07229) / [Frontier](https://arxiv.org/abs/2508.03148) | Hardware-level co-sim; disaggregation; prefix caching (Frontier adds KV transfer). | Fidelity-oriented; not a sub-second loop, no regret-vs-oracle. |
| [BLIS](https://inference-sim.github.io/inference-sim/latest/) | Go DES; weighted-scorer routing matching llm-d's EPP; tiered GPU+CPU KV. | KV tiering is vertical (host offload), not inter-instance; no oracle. Closest miss — good mid-fidelity check. |
| [DynoSim](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/) | Rust DES of NVIDIA Dynamo: KV-aware routing, G1/G2/G3 tiers, NIXL, autoscaling. | KV moves between workers, but targets Dynamo capacity planning, not llm-d policy iteration with per-request regret. |

So the offline sim stays small (~600 lines): a cost model whose `argmin` *is*
the three-regime decision, per-request regret vs. an oracle, a coupling metric,
and the same workload generator as the live harness.

## Architecture

**Live stack:** k3d cluster → llm-d Gateway (kgateway, Envoy) → EPP (routing
brain: prefix-cache state + backend load) → 2–4 `llm-d-inference-sim` replicas
(KV events + Prometheus metrics) ← `bench` CLI traffic generator measuring TTFT.

**Offline simulator (`src/bench/sim/`):**

| Module | Role |
|---|---|
| `blocks.py` | vLLM-style rolling block-hash chain — the unit prefix caching matches on. |
| `node.py` | `Node`: capacity-bounded KV-block LRU + single-server queue (`busy_until`). |
| `cost.py` | Cost model: TTFT for `(request, node, transfer?)` = `wait + transfer_time + prefill(missing)`. The `argmin` *is* the wait/transfer/recompute decision. |
| `workload.py` | Seedable multi-turn tool-calling sessions with a growing shared prefix. |
| `policies.py` | Routing policies compared against each other and the oracle. Plug in new ones here. |
| `engine.py` | Discrete-event loop: records TTFT, regime mix, regret vs. oracle, coupling. |

## Prerequisites

- **OrbStack** / **Docker Desktop** / **Colima** — live stack only.
- **Nix**: `curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s install`
- **direnv** (recommended) — auto-activates the `devenv` shell on `cd`; otherwise run `devenv shell` per session.

## Quick start

The offline simulator needs no cluster:

```bash
direnv allow        # or: devenv shell; uv sync runs on shell entry
uv run bench simulate --nodes 4 --sessions 200 --turns 6 --qps 10
```

Prints a comparison table across policies (round-robin, cache-local,
transfer-aware, weighted-precise, weighted-approx, class-aware, oracle).

## Offline simulation

```bash
uv run bench simulate \
  --policy round-robin --policy cache-local --policy transfer-aware --policy oracle \
  --nodes 4 --sessions 200 --turns 6 --qps 10 --seed 1
```

Options: `--nodes`, `--sessions`, `--turns`, `--seed`, `--block-size`,
`--cache-blocks` (per-node KV capacity), and:

- `--tool-result-tokens` — unique tokens per tool call, i.e. how much prefix sharing survives.
- `--think-time` — inter-turn gap; races against cache eviction.
- `--burst-cv` — CV of session inter-arrivals: 1.0 = Poisson, higher = bursty gamma (stresses wait-vs-recompute).
- `--bandwidth` — interconnect speed; lower it to see the oracle drop transfer for recompute.
- `--mix tool=0.5,rag=0.3,oneshot=0.2` — workload class fractions (default all tool).
  RAG queries share retrieved documents from a `--rag-docs` pool with Zipf popularity
  (`--rag-zipf`, `--rag-doc-tokens`); one-shots are unique tokens, pure cache pollution.
  With a mixed workload the report adds a per-class breakdown (regret / p50 / hit rate).

**Policies:** See [`docs/policies.md`](docs/policies.md) for a detailed breakdown of the routing policies, their purpose, and their production readiness (Tier 0 to Tier 2).

**Report columns:**

| Column | Meaning |
|---|---|
| TTFT p50/p99 | Predicted time-to-first-token. |
| Hit rate | Prefix blocks served from cache vs. recomputed. |
| Regime | % of requests resolved by wait / transfer / recompute. |
| Routing regret (s) | Mean `policy_TTFT − oracle_TTFT` on the policy's *own* cache state. Measures routing quality only; 0 for both `oracle` and `oracle-belady`. |
| TTFT gap (s) | `policy_p50 − oracle-belady_p50`; how far a policy's aggregate latency is from the theoretical best (a single comparable baseline for every row). |
| Coupled | % of requests whose optimal placement changes once every *other* node is forced idle — how much a decision depends on cluster-wide state. |

Sweep `--qps` and `--bandwidth`: regret and coupled fraction climb with
contention; the regime mix shifts from transfer to recompute as bandwidth drops.

## Live stack

```bash
devenv shell        # first run: Nix fetches k3d/kubectl/helm/uv; KUBECONFIG → .k3d/kubeconfig.yaml
cluster-create
deploy-llmd
deploy-monitoring   # Prometheus/Grafana
```

Lifecycle: `cluster-status`, `cluster-delete`.

Traffic and comparison:

```bash
uv run bench traffic --route round-robin --requests 200 --qps 10
uv run bench traffic --route prefix-affinity --requests 200 --qps 10
uv run bench report --compare round-robin --compare prefix-affinity --qps 10
uv run bench report --compare round-robin --compare prefix-affinity --workload sessions --qps 10
```

`--route` sets the `x-llmd-route` header: `round-robin` hits an HTTPRoute with
no EPP extProc (default k8s balancing); `prefix-affinity` uses the EPP-backed
route that steers by KV-cache state. `--qps` paces arrivals open-loop;
`--concurrency` caps in-flight requests. `--workload sessions` replays the
offline session generator (`--sessions`, `--turns`, `--seed`) against the real
EPP. `report` reads the prefix-cache counter delta around each run, waiting
`--settle` seconds (default 35, ≥ the Prometheus scrape interval) first.

Metrics: `uv run bench metrics`.

## Adding a policy

A policy is any callable with this signature; the engine wires it in
automatically:

```python
def my_policy(
    req: RequestView, nodes: list[Node], now: float, params: CostParams
) -> Placement:
    ...
```

`RequestView` is the request as a router sees it: `blocks` (prefix hashes) plus
body observables (`prompt_tokens`, `num_messages`, `has_tool_messages`,
`last_message_tokens`) — never the workload's ground-truth class. Nodes expose
`matched(blocks)` (cached prefix length) and `wait(now)` (queue delay);
`cost.predict(...)` / `best_placement(...)` score candidates the way the oracle
does. Return `Placement(node_id, transfer, regime, ttft)` — see `cache_local`,
`transfer_aware`, `class_aware`, or the stateful `RoundRobin`.

1. Register in `build_policy()` and `POLICY_NAMES` (`policies.py`) — the only wiring needed.
2. Add a test in `tests/test_sim.py`: assert non-negative regret, plus a scenario with a predictable outcome (e.g. single hot idle node → matches oracle).
3. Compare: `uv run bench simulate --policy <name> ...` and sweep `--qps` / `--think-time` / `--bandwidth`.

A policy needing a different cost signal can still return `Placement`s from
`predict()` candidates — the engine only depends on the call signature and
`Placement` shape.

## Promoting a policy to real llm-d routing

A validated sim policy becomes a real routing decision in llm-d's EPP (the
[inference scheduler](https://llm-d.ai/docs/architecture/Components/inference-scheduler)).
Three tiers, by effort:

**Tier 0 — config only.** Weights, saturation thresholds, and cache-visibility
mode are EPP configuration, no code
([`EndpointPickerConfig` YAML](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/config-text/);
here: `infra/llm-d/values.yaml`). Maps: `prefix_weight`/`load_weight` → scorer
weights in a scheduling profile; `weighted-precise` vs `weighted-approx` →
llm-d's [precise prefix-cache-aware mode](https://llm-d.ai/docs/guide/Installation/precise-prefix-cache-aware)
(KV events) vs the default router-side index; saturation depth → GIE's
queue-threshold filter.

**Tier 1 — a Go plugin.** `class-aware` promotes as a custom plugin in
[`llm-d-inference-scheduler`](https://github.com/llm-d/llm-d-inference-scheduler/):
either a `Scorer` (score pods 0–1; see the
[existing scorers](https://pkg.go.dev/github.com/llm-d/llm-d-inference-scheduler/pkg/scheduling/plugins/scorer))
or, better for per-class routing, a **ProfileHandler** that picks a scheduling
profile per request. The EPP sees the full request body, so `RequestView`'s
observables (tool role present, message count, user-message size) translate
directly; `classify()` becomes the profile pick, each class a profile with
different scorer weights. Ship it: implement + register the plugin, build the
EPP image, reference it in the config YAML, point the `InferencePool` at the
image, and validate on this repo's k3d stack (`bench report --workload
sessions`) before touching GPUs. `RequestView.label` anticipates GIE's
[body-based routing](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/pkg/bbr/README.md)
setting a class header upstream of the EPP.

**Tier 2 — a serving-stack change.** The transfer regime is not expressible in
stock llm-d: there is no inter-replica KV-transfer path (hence the `weighted-*`
policies' residual regret vs the oracle). It requires a vLLM KV connector tier
(LMCache, NIXL as in P/D disaggregation, or shared CPU offload), after which a
custom scorer prices `wait + pull(missing) + prefill(rest)` — literally
`cost.py`. The sim's job is to show that residual regret justifies that
infrastructure before anyone builds it.

The loop: **offline sim** (seconds, decides if the idea wins) → **k3d live
stack** (validates routing behavior — hit rates, herding, spill — with the real
EPP against `llm-d-inference-sim`; not latency, the sims run no kernels) →
**real vLLM on GPUs** (the only place absolute TTFT numbers mean anything).

## Checks

```bash
uv run pytest
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Cleanup

```bash
cluster-delete
```
