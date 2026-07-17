# RFC-0001 Phase 4 — EPP emission (implementation plan)

**Status:** in progress (started 2026-07-17) · **Scope this pass:** custom EPP image with a
`PreRequest` plugin emitting `x-kv-cache-priority`, real `inferencepool` chart values, extProc
body-mode fix, sim-fork image pin, `workload.py` `tool_calls` emission · **RFC:**
prototype-plan step 4 and §2/§5 of the RFC[^1].

## Context

Phases 1–3 built the retention mechanism and its backchannel: the offline sim is the
executable spec of the priority model, the forked `llm-d-inference-sim` honors
`x-kv-cache-priority` with a rank-aware evictor, and pinned occupancy/pressure flow back via
Prometheus and trailing `BlockStored` fields[^2]. But every directive so far originated from
the *client* escape hatch (`bench traffic --kv-priority/-ttl/-scope`). Phase 4 closes the loop
on the **router side**: the EPP itself classifies requests, learns per-tool re-arrival gaps,
and injects `x-kv-cache-priority: <int>[; ttl=<Go duration>][; scope=<id>]` after scheduling —
`50; ttl=…; scope=…` on confident short predicted tool-gaps, `-1` (evict-first) for one-shots —
per RFC §5.

Two blocking defects found while planning (verified against llm-d-router v0.9.2 source in the
Go module cache and the live cluster):

1. **The extProc config starves the EPP entirely.** llm-d-router's
   `pkg/epp/handlers/server.go` only calls `director.HandleRequest` in the
   `ProcessingRequest_RequestBody` branch on EndOfStream. With the GatewayExtension's
   `processingMode.requestBodyMode: NONE` (current `infra/llm-d/inference-pool.yaml`), a POST's
   headers arrive with EoS=false and no body phase ever comes — scheduling never runs, so today
   *no* route actually consults the EPP. Fix: `requestBodyMode: BUFFERED` on the shared
   GatewayExtension. This is needed for **all** EPP routes, not just class-aware — prefix
   scoring needs the prompt too.
2. **The deployed sim is not the fork.** Pods run `ghcr.io/llm-d/llm-d-inference-sim:latest`
   (digest created 2026-06-21, predating the fork branch). Without building and pinning the
   fork image, the emitted header lands on a backend that ignores it and the pinned gauges
   never move.

Load-bearing facts the plan relies on (verified in the v0.9.2 module cache):

- **No fork of llm-d-router is needed.** Its `cmd/epp/main.go` is ~15 lines calling
  `runner.NewRunner().Run(ctx)` from the importable `cmd/epp/runner` package; out-of-tree
  plugins register via `fwkplugin.Register(pluginType, factory)`
  (`pkg/epp/framework/interface/plugin/registry.go`). `PreRequest(ctx, req, schedResult)`
  mutates `req.Headers`; `handlers/request.go generateHeaders` forwards every non-system-owned
  header, and `x-kv-cache-priority` is not system-owned (confirms RFC §5).
- **Config format** is llm-d's own `apiVersion: llm-d.ai/v1alpha1, kind: EndpointPickerConfig`
  (reference: the module's `deploy/config/epp-config.yaml`), not GIE's.
- **Drop-in chart compatibility**: the GIE `inferencepool` chart's EPP args (`--pool-name`,
  `--config-file`, `--zap-encoder json`, …) are all accepted by llm-d-router's runner, and its
  `Dockerfile.epp` exposes the same ports (9002 gRPC / 9003 health / 9090 metrics) — a custom
  image slots in via image override + `pluginsCustomConfig`.
- `infra/llm-d/values.yaml`'s current 16 lines use keys the chart ignores (the RFC's "likely
  inert" suspicion, confirmed) — full rewrite. Version skew: `setup.sh` pins chart/CRDs v1.4.0
  while llm-d-router v0.9.2 requires GIE v1.5.0 → align to v1.5.x.
- `workload.py` serializes tool turns without `tool_calls`/name, so per-tool gap keying has
  nothing to key on E2E — small serialization-only change in scope.

Decisions locked this session: **minimal EPP config** — one default scheduling profile
(`prefix-cache-scorer` + `max-score-picker` + `single-profile-handler`) plus the new
`kv-cache-priority` PreRequest plugin; the existing `ClassAwareReliability` ProfileHandler
stays compiled-in and registered but **unwired** (profile-routing A/B is phase 5). The
`workload.py` `tool_calls` change **is** in scope.

Deferred to phase 5 (unchanged from phase 3's carry-over[^2]): hint-on/off A/B harness,
program-level measures (session completion time, zero-recompute rate, TTFT-by-turn),
non-agentic throughput guardrail, routing-weight interactions (RFC §5 "v2"), upstream
feedback post.

## Deliverables

### A. Discovery (read-only; confirm predictions before code)

- **[D1] Live EPP behavior**: dump the rendered plugins ConfigMap; tail `gaie-epp` logs during
  a `bench traffic` run with `x-llmd-route: class-aware-reliability`. *Predicted:* headers
  arrive, scheduling never runs (body-NONE starvation confirmed live).
- **[D2] kgateway CRD**: confirm `BUFFERED` is a legal `requestBodyMode` value in
  `gatewayextensions.gateway.kgateway.dev`.
- **[D3] Chart schema**: `helm show values …/inferencepool --version v1.5.0`; confirm
  `inferenceExtension.image.*`, `flags`, `pluginsConfigFile`, `pluginsCustomConfig` key names;
  diff vs v1.4.0.
- **[D4] Plugin type strings**: exact registered names in v0.9.2's `cmd/epp/runner/runner.go`
  (`prefix-cache-scorer`, `max-score-picker`, `single-profile-handler`; whether `decode-filter`
  matters without P/D).
- **[D5] Tool names on parsed messages**: whether tool-role messages surface a function name
  after the OpenAI parser (`Message.ToolCalls []any`). *Predicted:* only via assistant-message
  `tool_calls` → drives the classify/tool-key extraction below.

### B. Plugin library (`src/gateway-plugin/`, pure Go)

Mirror `src/bench/sim/policies.py` semantics and constants: `alpha=0.3`, index LRU cap 512,
`default_prior=(2.0, 16.0)`, gap-tracker cap 8192, `short_gap=3.0`, `conf_threshold=0.5`,
`retain_margin=1.5`, retention window `max(mean·1.5, mean+√var)`,
`confidence = 1/(1+var/max(mean², 1e-9))`.

- `toolgap.go` — replace the single-global EWMA `ToolGapIndex` with the per-tool version:
  map + LRU (cap 512), `Record(tool, gap)` (EWMA mean/var exactly as the Python), `Predict(tool)`
  with priors→default-prior fallback, `Confidence`. Injectable clock for tests.
- `session.go` — `observe_gap` port: bounded LRU (cap 8192) conversation-key → last-seen;
  returns the realized gap or none. Conversation key: `x-session-id` header when present, else
  an FNV-1a hash of (system message + first user message) — stable as a session grows. The
  RFC §2 prefix-hash-chain fallback scope is deferred: `approximateprefix` deletes its
  per-request plugin state in its own `PreRequest`, so cross-plugin reads race (risk R4;
  upstream feedback material).
- `classify.go` — extract classification from `class_aware_reliability.go` into a shared func
  (tool if any `role=="tool"` or `Tools` non-empty; oneshot if ≤1 message; rag if last message
  ≥512 chars — documented ~4 chars/token proxy for the Python 128-token threshold; else tool).
  Add last-tool-key extraction: walk back to the latest assistant message with `ToolCalls`,
  pull `function.name` via map assertion; fallback key `"unknown-tool"`.
- `prerequest.go` — new plugin type `kv-cache-priority` implementing
  `requestcontrol.PreRequest`: oneshot → header `-1`; tool → observe gap, `Record`, `Predict`;
  if confident and short → `50; ttl=<window, ms-rounded Go duration>; scope=<key>`; rag or
  unconfident → no header. Values must parse under the fork's `pkg/retention` grammar.
  Thresholds overridable via the factory's JSON `parameters`.
- `factory.go` — `FactoryFunc` for `kv-cache-priority`; keep `ClassAwareReliability`
  registered under its own type (unwired).
- Tests — table tests mirroring the Python expectations: priors before first observation,
  EWMA convergence, LRU eviction at cap, `observe_gap` turn-0/cap behavior, classify ordering,
  and exact header strings from `PreRequest` on synthetic requests (incl. the no-header cases).

### C. Binary + images + k3d import

- `src/gateway-plugin/cmd/epp/main.go` — mirror llm-d-router's main; `fwkplugin.Register(...)`
  for both plugins before `runner.NewRunner().Run(...)`.
- `src/gateway-plugin/Dockerfile` — adapted from the module's `Dockerfile.epp`: golang ≥1.26
  builder, `CGO_ENABLED=0 GOARCH=arm64`, distroless static final stage; tag
  `sched-lm/gaie-epp:rfc0001`. Plain docker build (OrbStack present; no `ko` in devenv).
- Sim fork image: `docker build -t ghcr.io/llm-d/llm-d-inference-sim:rfc0001
  third_party/llm-d-inference-sim` (the fork has its own Dockerfile).
- `devenv.nix` — new scripts `build-epp` / `build-sim`: docker build `--platform linux/arm64`
  + `k3d image import <tag> -c $K3D_CLUSTER_NAME`.

### D. Infra rewrite

- `infra/llm-d/values.yaml` — full rewrite to the real chart schema (exact keys per D3):
  custom EPP image under `inferenceExtension.image.*`, `flags`, resources,
  `pluginsConfigFile: kv-priority-plugins.yaml`, and `pluginsCustomConfig` embedding the
  `EndpointPickerConfig`: plugins `prefix-cache-scorer`, `max-score-picker`,
  `single-profile-handler`, `kv-cache-priority`; one `default` scheduling profile. Move
  `inferencePool.modelServers.matchLabels.app=vllm-sim` here from the `setup.sh` `--set`.
- `infra/llm-d/setup.sh` — bump GIE CRDs kustomize ref + chart `--version` to the aligned
  v1.5.x; drop the redundant `--set`.
- `infra/llm-d/inference-pool.yaml` — sim image → `:rfc0001` with
  `imagePullPolicy: IfNotPresent`; GatewayExtension `requestBodyMode: NONE → BUFFERED`.
- `src/bench/sim/workload.py` — emit OpenAI-spec `tool_calls` (with `function.name` from the
  session's tool) on assistant messages of tool turns; serialization-only, offline-sim
  behavior byte-identical; pytest updated.

### E. Deploy + E2E

1. `deploy-llmd`; rollout status; EPP logs show plugin instantiation incl.
   `kv-cache-priority`; InferencePool reconciled with 2 endpoints; sims run `:rfc0001`.
2. Baseline via the client escape hatch (`--kv-priority 50 --kv-ttl 30s`) → pinned gauges
   move (isolates sim-side regressions from EPP work).
3. EPP emission run: session workload with `x-llmd-route: class-aware-reliability` and *no*
   `--kv-priority`. Assert: EPP logs run the PreRequest plugin; sim sees
   `x-kv-cache-priority: 50; ttl=…; scope=…` on warm tool turns and `-1` on one-shots;
   Prometheus `vllm:kv_cache_pinned_usage_perc > 0` and `vllm:kv_cache_priority_blocks`
   nonzero; scheduling visibly engages (proves the BUFFERED flip un-starved the path).

### F. Docs + close-out

RFC prototype-plan step 4 → ✅ with a dated summary (BUFFERED requirement, sim-image gap,
chart alignment, deferred prefix-hash scope divergence); this doc's status updated as
deliverables land.

## Suggested order

A (discovery) → B (plugin lib, pure Go, testable offline) → C (binary/images) → D (infra) →
E (deploy/E2E) → F (docs).

## Verification

- **Go**: `cd src/gateway-plugin && go build ./... && go test ./...`; image smoke:
  `docker run --rm sched-lm/gaie-epp:rfc0001 --help`; `crictl images | rg rfc0001` inside the
  k3d node after import.
- **Infra**: `helm template gaie oci://…/inferencepool --version <v1.5.x> -f
  infra/llm-d/values.yaml` renders our image + the `kv-priority-plugins.yaml` ConfigMap;
  `kubectl apply --dry-run=server -f infra/llm-d/inference-pool.yaml` accepts BUFFERED.
- **Python**: `uv run pytest`, `uv run ruff check src/ tests/`.
- **E2E**: deliverable E above.

## Risks / notes

- **R1** v1.4→v1.5 chart values drift — mitigated by D3 before the rewrite.
- **R2** BUFFERED changes the default prefix-affinity route too (the EPP is finally
  consulted, plus per-request body buffering latency) — expected and desired, but re-baseline
  existing route comparisons afterward; the chart's fail-open default bounds the blast radius.
- **R3** v1.5.x CRDs installing next to existing v1.4.0 objects — the chart-owned `gaie`
  InferencePool upgrades in place; confirm during deploy.
- **R4** Prefix-hash fallback scope (RFC §2) not cleanly implementable at `PreRequest` in
  v0.9.2 (state race with `approximateprefix`'s delete-on-read) — v1 uses
  session-header/first-message-hash scope; recorded as upstream feedback.
- Submodule/jj gitlink caveat from phase 2 still applies if the fork needs new commits:
  commit inside the submodule + push to the fork, then record the gitlink at the git level.

[^1]: [`rfc-0001-kv-cache-priority-directives.md`](./rfc-0001-kv-cache-priority-directives.md)
    — §2 (transports/header grammar), §5 (EPP emission mechanism), prototype-plan step 4.
[^2]: [`rfc-0001-phase-3-plan.md`](./rfc-0001-phase-3-plan.md) — stats backchannel, and the
    phase-5 carry-over list (A/B harness, program-level measures, throughput guardrail).
