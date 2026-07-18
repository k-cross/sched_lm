# RFC-0001 Phase 3 — Stats backchannel (implementation plan)

**Status:** ✅ done (2026-07-16) — all deliverables A–F landed and verified; hint-on/off A/B
and program-level PoC-3 measures carried into phase 5 as scoped. · **Scope this pass:** stats
+ ZMQ backchannel (hint-on/off A/B and program-level PoC-3 measures deferred to phase 5) · **RFC:**
[`rfc-0001-kv-cache-priority-directives.md`](./rfc-0001-kv-cache-priority-directives.md)
§4, prototype-plan step 3.

## Context

Phases 1–2 built the retention *mechanism*: the offline sim honors the priority
vocabulary, and the forked `llm-d-inference-sim` honors `x-kv-cache-priority` directives
with a rank-aware evictor (`pkg/kv-cache/block_cache.go`) that already tracks a
`pinnedEvictions` pressure counter and a per-block `retention` map. What's missing is the
**backchannel**: the router/EPP and this bench harness have no way to see whether their
directives help or over-pin. RFC §4 defines that channel — Prometheus occupancy/pressure
metrics, and trailing `BlockStored` event fields so the distributed indexer learns pin
state — and prototype-plan step 3 grows the Python bench to read them.

Decisions locked this session: scope is **core stat loop + ZMQ backchannel**; the
hint-on/off A/B harness and program-level measures (session completion time,
zero-recompute rate, TTFT-by-turn) are **deferred to phase 5**. The v0.9.0 adapter compat
check is done by **vendoring the real adapter** as a test dependency, not golden fixtures.

Outcome: `pinned_evictions` becomes visible in both the offline sim table and live
Prometheus; a marked prefix's occupancy and eviction pressure are queryable; and the ZMQ
event stream carries priority/retain_until without breaking the deployed
`llm-d-kv-cache@v0.9.0` adapter.

## Deliverables

### A. Fork — Prometheus metrics (`third_party/llm-d-inference-sim`) ✅ done

Three new metrics on the sim's existing registry, following the `vllm:`/`model_name`
convention already used by `kvCacheUsagePercentage`:

| Metric | Type | Source |
|---|---|---|
| `vllm:kv_cache_priority_blocks{priority=evict_first\|high\|pinned}` | GaugeVec | resident marked blocks per band, bucketed from `bc.retention` |
| `vllm:kv_cache_pinned_usage_perc` | Gauge | marked-and-unexpired blocks / `bc.maxBlocks` |
| `vllm:kv_cache_pinned_evictions_total` | Counter | the existing `bc.pinnedEvictions` |

- **Registration**: mirror `kvCacheUsagePercentage` in `pkg/llm-d-inference-sim/metrics.go`
  (declare fields ~L137, `NewGaugeVec`/`NewCounterVec` + `registry.Register` ~L303, zero in
  `setInitialPrometheusMetrics` ~L401).
- **Wiring seam**: `MetricInfo` (`pkg/common/utils.go`) only carries `Value` — no per-band
  label, no counter delta. Add a small **stats-snapshot channel** (new struct:
  pinned-usage-perc + `[3]int` band counts) pushed from `blockCache` where it already
  recomputes/pushes usage — end of `startRequest` (~L315) and `finishRequest` (~L359) —
  plus increment the counter at the one site where `bc.pinnedEvictions++` already fires
  (`pickBlockToEvict`, block_cache.go:515). Band bucketing reuses
  `retention.{EvictFirstPriority,HighPriority,PinnedPriority}`. Consume in `fake_metrics.go`
  next to the existing `kvCacheUsagePercentage` consumer (~L207).
- Preserve the zero-overhead invariant: when `len(bc.retention)==0`, push zeroed bands (no
  per-block iteration) so non-agentic workloads stay on the fast path.

### B. Fork — ZMQ `BlockStored` trailing fields ✅ done

Append the two RFC-specified fields to **both** encodings in
`pkg/kv-cache/kv_cache_sender.go`:
- Positional array `msgpackBlockStoredEvent` (L49): `[12] priority (*int)`,
  `[13] retain_until (*float64 unix seconds)` as trailing `omitempty` fields after
  `ExtraKeys`.
- Map `blockStoredEvent` (L69): `priority` / `retain_until` keys.
- Populate from the block's `retentionMark` at store time; thread the mark through
  `EventData` (L110) alongside the existing `parentHash`, built in `KVEventSender.Run`
  (~L192). `nil` for unmarked blocks keeps current non-agentic output byte-identical.
- **No new event tags** — priority changes re-emit `BlockStored` (idempotent upsert), per
  the RFC's verified adapter constraint.

### C. Fork — compat test (vendored real adapter) ✅ done

- `go get github.com/llm-d/llm-d-kv-cache@v0.9.0` as a **test-only** dependency.
- New `pkg/kv-cache/retention_event_compat_test.go`: encode our `BlockStored` (with
  trailing fields), decode through the real `VLLMAdapter`, asserting (1) known-tag events
  with extra trailing fields decode cleanly (fields ignored), and (2) an unknown event tag
  fails the whole batch — locking in the assumption behind "no new event types in v1."
- Watch for transitive-dep drag on the fork `go.mod`; if v0.9.0 pulls heavy/incompatible
  requirements, fall back to a thin golden-fixture assertion (flag before proceeding).

### D. Python bench — read the new metrics (`src/bench`) ✅ done

- `metrics.py` `MetricsClient`: add `get_pinned_usage()`, `get_pinned_evictions()` (counter
  delta across a run, like `get_prefix_cache_counters`), and per-band
  `get_priority_blocks()`. Reuse the existing `_scalar`/`query` helpers.
- `report.py` `generate_report`: add **Pinned usage %** and **Pinned evictions** columns;
  thread values from the call site in `cli.py`.

### E. Python bench — surface `pinned_evictions` in the offline sim table ✅ done

`report.py` `generate_sim_report`: added a **Pinned evict** column reading the already-present
`SimResult.pinned_evictions`. Verified on the over-pinning repro (5209 evicts for
`class-aware-reliability` vs 0 for `class-aware`; →0 at cache-blocks 5000). `docs/policies.md`
§"Known issue" caveat updated.

### F. Docs ✅ done

- Update `rfc-0001-kv-cache-priority-directives.md` prototype-plan step 3 → ✅ done for the
  shipped slice, noting A/B + program-level measures carried into phase 5.
- Drop the "Not yet surfaced in `bench simulate`" caveat in `docs/policies.md` §Known issue →
  ✅ done (now reads "Surfaced as the **Pinned evict** column").
- Mark this plan doc's status done as each deliverable lands → ✅ done.

## Suggested order

E (sim column, isolated) → A + D (live stat loop, E2E-verifiable) → B + C (ZMQ + compat) →
F (docs). Tests alongside: Go tests in `pkg/kv-cache` for A/B/C; a `metrics.py` unit test
with a stubbed Prometheus response and a `report.py` snapshot for D/E.

## Verification

- **Fork**: `go test ./pkg/kv-cache/... ./pkg/retention/...`. Run the sim binary locally,
  send a `--kv-priority` request via `bench traffic`, and
  `curl :<port>/metrics | rg kv_cache_priority` to confirm the three series populate and
  `pinned_evictions_total` climbs under filler pressure. Inspect a captured ZMQ batch for
  `priority`/`retain_until` on marked blocks and their absence on unmarked ones.
- **Offline sim (E)**: `uv run bench simulate` on the over-pinning repro from
  `docs/policies.md` (3 nodes, `--cache-blocks 200`, `--mix tool=1.0`, `--tool-reliability
  --seed 21`) — new column high there, ~0 with a large cache.
- **Python**: `uv run pytest`, `uv run ruff check src/ tests/`, `uv run basedpyright`
  (diff against baseline noise).

## Notes / caveats

- Pre-existing lint error unrelated to this work: mid-file `import pytest` at
  `tests/test_traffic.py:61` (E402 + I001). Sweep in F only if desired.
- Submodule/jj gitlink caveat from phase 2 still applies: commit inside the submodule + push
  to the fork, then record the gitlink in the parent (git-level, not jj).
