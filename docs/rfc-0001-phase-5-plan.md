# RFC-0001 Phase 5 — Evidence + upstream feedback (implementation plan)

**Status:** ✅ done (2026-07-18) — all deliverables A–F landed; evidence and the #37003
comment draft are in `rfc-0001-upstream-feedback.md`. Headline: the router-side retention
path runs end-to-end (on-arm pins 6.1–6.6% of cache with zero client directives, off-arm
0%), but on this workload it gives no reuse benefit (the EPP pins short-gap sessions LRU
already retains, so zero-recompute is flat across arms) and under cache pressure reproduces
the over-pinning collapse (pins 52%, success 100%→89%). The non-agentic guardrail shows no
overhead. A code-review pass hardened the instrumentation (usage-parse no longer aborts a
run on a malformed chunk; session-completion time counts only fully-served sessions;
peak-pinned reads None-vs-0.0% honestly). · **Scope this pass:** hint-on/off A/B via a
route gate in the EPP plugin, PoC-3 program-level measures (TTFT-by-turn-position, session
completion time, zero-recompute rate, non-agentic throughput guardrail), the Prometheus
pod-annotation scrape fix, peak (`max_over_time`) pinned queries, evidence runs, and the
upstream feedback doc for vllm#37003 · **RFC:** prototype-plan step 5 and §4/§7 of the
RFC[^1]. Routing-weight interactions (§5 "v2") stay out of scope — future-work note only.

## Context

Phases 1–4 built the full directive path: the offline sim is the executable spec of the
priority model, the forked `llm-d-inference-sim` honors `x-kv-cache-priority` with a
rank-aware evictor and exports pinned metrics, the stats backchannel flows via Prometheus
and trailing `BlockStored` fields, and the EPP's `kv-cache-priority` PreRequest plugin
autonomously emits directives from observed per-tool re-arrival gaps[^2][^3]. Phase 5
produces the evidence: `bench report` comparing `prefix-affinity` (directives off) against
`class-aware-reliability` (directives on) on session workloads, with the PoC-3 measures,
then a results doc carrying the §7 divergence list, ready for posting to vllm#37003 while
its feedback window is open.

Load-bearing facts verified while planning:

1. **The Prometheus "missing scrape job" is a YAML duplicate-key bug, not missing config.**
   `infra/monitoring/prometheus-values.yaml` declares `prometheus:` twice at top level —
   line 1 carries the correct annotation-based `kubernetes-pods` scrape job under
   `prometheusSpec.additionalScrapeConfigs`, line 42 carries the NodePort service. Helm's
   YAML parse takes the last key, silently dropping the entire first block. The sim pods
   already carry the right `prometheus.io/*` annotations. This is phase 4's follow-up
   item 7[^3].
2. **`bench report` never fires directives today.** The `report` command has no
   `--tool-reliability` option, so `_apply_preset` drops the experiment preset's
   `tool_reliability: True` (it only sets keys already present in `ctx.params`) and
   `_run_route` defaults it to `False` — no `tool_calls` reach the wire, so the EPP has
   nothing to learn per-tool gaps from.
3. **The two EPP routes are behaviorally identical today.** Both HTTPRoutes attach the same
   extProc, kgateway v2.3.6 ignores the EPP's endpoint pick (phase-4 finding 3[^3]), and
   the `kv-cache-priority` plugin emits on every route. An A/B run today measures noise —
   phase 5 needs a directives-off switch.
4. **Zero-recompute is client-observable without fork changes.** The fork supports
   `stream_options.include_usage`; the final streamed chunk carries `usage.prompt_tokens`
   and `prompt_tokens_details.cached_tokens`. (Non-final chunks serialize `"usage":null`,
   so the parse must key on the `"prompt_tokens"` substring.)
5. **Pinned gauges decay before the post-run scrape.** EPP-emitted TTLs are ~seconds while
   `report` waits `--settle 35` before an instant query — it reads post-decay ~0. Peak
   queries (`max_over_time` over the run window) are needed; the pinned-evictions *counter*
   delta is scrape-timing-proof either way.

Design decisions locked this session:

- **Two arms, not 2×2.** With endpoint picks ignored, the directive header is the *only*
  live difference between routes, so `prefix-affinity` = EPP engaged / directives OFF and
  `class-aware-reliability` = same data path / directives ON. The other two 2×2 cells are
  duplicates of these. This is simultaneously the RFC step-5 comparison and phase 3's
  carried-over hint-on/off A/B — it cleanly isolates the retention effect (framing *and*
  honest caveat for the upstream doc). `round-robin` remains available as a no-EPP context
  row.
- **Directives-off switch = route gate in the plugin.** `KVCachePriority.PreRequest`
  returns early unless `x-llmd-route: class-aware-reliability`, placed before `decide()` so
  the off arm neither emits nor trains the gap index. One Go change, one image rebuild
  total; both arms run in a single `bench report` invocation with identical extProc timing.
  Rejected: per-arm Helm overlay (rollout between arms kills the single-command A/B) and a
  client low-priority header (no no-op value exists — marked-0 still outlives unmarked).
- **Zero-recompute rule** (turn k>0 of a session):
  `cached_tokens_k ≥ (prompt_tokens_{k-1} // block_size) · block_size` — everything the
  previous turn actually prefilled is still resident, block-aligned. The reusable prefix is
  the previous *prompt*, not prompt+completion: the workload scripts its own assistant
  text, which the sim never generated, so the scripted segment is always a miss. Mean
  prefix coverage `cached_tokens / prompt_tokens` per turn position is the continuous
  companion metric.
- **Cross-arm cache isolation via `session_id_offset`.** Session text tags are keyed only
  by `sid`, so a second arm at the same seed replays byte-identical prefixes and
  warm-starts from the first arm's cache. Offsetting sids per arm gives unique per-arm
  text while the shared seed keeps arrival streams identical (paired comparison). The
  shared system prompt stays warm across arms — symmetric, disclosed in the doc.
- **Session completion time** = first-turn dispatch → last-turn completion, p50/p90 across
  sessions. It includes scripted think-time gaps, which are identical across arms, so
  deltas are pure latency signal.
- **Non-agentic guardrail** = the existing `single-shot` workload through both arms plus
  achieved throughput (successes / wall seconds); on the on-arm, one-shots get marked `-1`
  by design, and the claim measured is "no latency/throughput regression on unannotated
  traffic".

## Deliverables

### A. Prometheus scrape fix

- `infra/monitoring/prometheus-values.yaml` — merge the duplicate `prometheus:` keys into
  one mapping (NodePort service + `prometheusSpec` with the existing
  `additionalScrapeConfigs` block unchanged). Verify with `helm template` locally, then
  live: the `kubernetes-pods` scrape pool lists the sim pods and
  `vllm:kv_cache_pinned_usage_perc` returns series after smoke traffic.

### B. Bench instrumentation (pure Python, testable offline)

- `src/bench/sim/workload.py` — `generate_sessions(..., session_id_offset: int = 0)`;
  `sid = session_id_offset + i` in every tag/`session_id`/`conversation_id`. RNG
  consumption order untouched → offset=0 output byte-identical.
- `src/bench/traffic.py` — frozen dataclass `TurnMetric(session_id, turn_idx, ttft, e2e,
  prompt_tokens, completion_tokens, cached_tokens)`; `BenchmarkResult` grows
  `turns: list[TurnMetric]`, `session_times: dict[int, float]`, `wall_seconds: float`.
  `_stream_chat` gains `turn_idx`, sends `stream_options: {"include_usage": true}` and
  parses the final usage chunk; `run_session_traffic` records per-session wall time; both
  runners set `wall_seconds`.
- `src/bench/program_metrics.py` (new) — `ProgramMetrics(throughput_rps, session_p50,
  session_p90, zero_recompute_rate, ttft_by_turn, coverage_by_turn)` +
  `compute_program_metrics(result, block_size=16)`: pure aggregations over
  `BenchmarkResult` implementing the decisions above.
- `src/bench/metrics.py` — `get_peak_pinned_usage(range_seconds)` and
  `get_peak_priority_blocks(range_seconds)` via `max(max_over_time(...[Ns]))`, reusing
  `_scalar`.

### C. EPP route gate (Go)

- `src/gateway-plugin/prerequest.go` — consts `RouteHeader = "x-llmd-route"`,
  `DirectiveRoute = "class-aware-reliability"`; early return at the top of `PreRequest`
  (before `decide()`, so the off arm neither emits nor trains). Tests: existing emission
  cases gain the route header; new cases assert no emission *and* no gap-index mutation
  for missing-header and `prefix-affinity` requests.
- Rebuild + import: `build-epp`, rollout-restart the EPP deployment.

### D. CLI/report wiring

- `src/bench/cli.py` — `report` gains `--tool-reliability` (threaded into `_run_route`;
  the experiment preset key now lands). `_run_route` gains `session_offset` forwarded to
  `generate_sessions`; the report loop passes `offset = arm_index · sessions · 100`. Per
  arm: record run start, and after settle query peaks with
  `range_seconds = elapsed`, compute `ProgramMetrics`, pass both to `generate_report`.
- `src/bench/report.py` — main table adds `Req/s`, `Sess p50 (s)`, `Zero-recomp`; the
  pinned column becomes `Pinned peak %`. New second table (rendered only when turn data
  exists): "TTFT by turn position (p50 s)" — per route, a p50-TTFT row and a
  mean-coverage row. Flat vs decreasing TTFT across turn positions is the headline read.

### E. Evidence runs

1. Fast smoke A/B:
   `uv run bench report --compare prefix-affinity --compare class-aware-reliability
   --workload sessions --preset fast --tool-reliability` — both tables render; on-arm
   shows nonzero pinned peak, off-arm zero.
2. Sessions A/B (experiment grade): `--preset experiment` (50×6, qps 10, concurrency 20,
   tool-reliability), seeds 0/1/2; restart EPP + sims between invocations for cold state.
3. Guardrail: `--workload single-shot --requests 300 --qps 10` through both arms.

### F. Upstream feedback doc + close-out

- `docs/rfc-0001-upstream-feedback.md` (new) — testbed + honest caveats (simulator
  latency model, shared placement because kgateway ignores picks — the A/B isolates the
  directive alone); method (arms, workload params, metric definitions verbatim); results
  tables (session A/B incl. TTFT-by-turn; non-agentic guardrail) with exact repro
  commands; reading the evidence; the five §7 divergences condensed from the RFC; future
  work (routing-weight v2, Tier-2 KV-transfer analog).
- RFC prototype-plan step 5 → ✅ with dated summary; this doc's status updated as
  deliverables land.

## Suggested order

A+B (offline, parallel) → C (Go gate) → D (wiring) → E (deploy, smoke, evidence) → F (docs).

## Verification

- **Python**: `uv run pytest`, `uv run ruff check src/ tests/`. New tests: usage-chunk
  parse + `TurnMetric`/`session_times`/`wall_seconds` capture (`test_traffic.py`),
  zero-recompute block-rounding edges and turn-0 exclusion (`test_program_metrics.py`,
  new), peak-query strings/parsing (`test_metrics.py`), `session_id_offset` invariants
  (workload tests).
- **Go**: `cd src/gateway-plugin && go build ./... && go test ./...`.
- **Infra**: `helm template` the monitoring values; live target count via
  `/api/v1/targets`; pinned gauge series present after smoke traffic.
- **E2E gate check**: session run on `prefix-affinity` → no "directive emitted" in EPP
  logs; on `class-aware-reliability` → emitted, and
  `vllm:kv_cache_priority_blocks{priority="high"}` moves.

## Risks / notes

- **R1** Pinned gauges under-observed even with peak queries (~4 s TTLs vs 30 s scrape
  interval): experiment runs last minutes so scrapes land mid-pressure; fallback is the
  phase-4 technique — port-forward a sim pod and curl `/metrics` mid-run. The
  pinned-evictions counter delta is robust regardless.
- **R2** EPP admission control sheds load at experiment qps (phase-4 note): watch
  "load shed" tallies; lower `--qps` or scale the sims — keep arms identical.
- **R3** Usage-chunk parse fragility: non-final chunks carry `"usage":null`; the parse
  keys on the `"prompt_tokens"` substring, covered by unit test.
- **R4** Shared system prompt stays warm across arms: symmetric by design; disclosed in
  the doc. Full isolation available via sim rollout restart between report invocations.
- **R5** Absolute numbers are simulator-shaped (fixed `prefill-overhead`/`itl` latency
  model): frame all evidence as relative A/B deltas in the upstream doc.

[^1]: [`rfc-0001-kv-cache-priority-directives.md`](./rfc-0001-kv-cache-priority-directives.md)
    — §4 (bench metrics / PoC-3 alignment), §7 (divergence list), prototype-plan step 5.
[^2]: [`rfc-0001-phase-3-plan.md`](./rfc-0001-phase-3-plan.md) — stats backchannel and the
    original phase-5 carry-over list.
[^3]: [`rfc-0001-phase-4-plan.md`](./rfc-0001-phase-4-plan.md) — EPP emission, and the E2E
    findings this phase inherits (FULL_DUPLEX_STREAMED, kgateway pick-ignoring, the
    missing scrape job).
