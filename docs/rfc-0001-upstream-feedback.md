# RFC-0001 — Evidence and feedback for vllm#37003

**Status:** ready to post (2026-07-18). Distilled from the phase-5 evidence runs[^1] of the
`llm-d-emulation-bench` prototype. This is the material for a comment on the upstream
`RetentionDirective` RFC[^2] while its feedback window is open.

## What this is

An independent, end-to-end prototype of **router-originated** KV-cache retention: an llm-d
EPP plugin classifies each request, learns per-tool re-arrival gaps from timing alone, and
emits a compact `x-kv-cache-priority` directive that a forked `llm-d-inference-sim`[^3]
honors with a rank-aware evictor. It plays the "PoC 3" role in the retention story — the
routing layer, not the model server or the client, deciding what to keep warm.

The bench runs on k3d/ARM64 against the simulator, not GPU vLLM, so the numbers below are
**relative A/B deltas**, not absolute performance. Two arms are compared:

- `prefix-affinity` — the EPP is engaged (identical scheduling and extProc path) but
  directive emission is gated **off**.
- `class-aware-reliability` — the same data path with directive emission **on**.

Because the pinned kgateway (v2.3.6) ignores the EPP's endpoint pick[^1], both arms share
placement, so the directive header is the *only* live difference between them — the A/B
isolates the retention directive itself. That is a feature for attribution and a caveat for
absolute numbers.

### Testbed caveats (read before the tables)

- **The simulator's TTFT is a fixed latency model** (`--prefill-overhead=50ms`,
  `--inter-token-latency=10ms`), not a function of uncached tokens. It reports
  `cached_tokens` but does not discount prefill for them, so **TTFT here cannot show the
  latency benefit of prefix reuse** — retention shows up only as cache *occupancy* and
  *eviction* behavior. Every TTFT column below is therefore near-flat across arms and turn
  positions; treat it as a null instrument, not a null result.
- Placement is shared across arms (above), so this measures the directive's effect on a
  fixed set of pods, not router-driven load balancing (RFC §5 "v2", out of scope[^1]).
- `bench`'s Prometheus prefix-cache-hit-rate reads 0% on this fork (the sim does not move
  `vllm:prefix_cache_hits_total`); the trustworthy reuse signal is the client-observed
  `cached_tokens` coverage and the derived zero-recompute rate.

## Method

Workload: 50 tool-calling sessions × 6 turns, per-tool reliability on (assistant turns
carry OpenAI `tool_calls` so the EPP can key per-tool gaps), paced by the modeled arrival
stream. Arms run in one `bench report` invocation with disjoint session-id offsets, so both
replay identical arrival streams over non-overlapping prefixes (neither warm-starts from the
other's cache). Sim KV cache is 4096 blocks/pod (65 k tokens) unless noted. EPP + sims are
cold-restarted between seeds.

Metric definitions: **zero-recompute** = fraction of returning turns (turn *k*>0) whose
server-reported `cached_tokens` cover the previous turn's whole prompt, block-aligned
(`cached_tokens_k ≥ ⌊prompt_{k-1}/16⌋·16`); the previous turn's scripted completion is
excluded because the sim never generated it. **Pinned peak %** = `max_over_time` of the
per-pod pinned-usage gauge over the run window (EPP leases decay in seconds, so an instant
read is ~0). **Session p50/p90** = wall time from a session's first-turn dispatch to
last-turn completion, counting only fully-served sessions. **Cache peak %** =
`max_over_time` of overall KV-cache utilization over the window (how full/contended the
cache got). **Evictions** = `sum(increase(vllm:kv_cache_evictions_total[window]))` — every
block evicted to make space for a new one, across all priority bands; `increase()` corrects
for the counter resets caused by memory-limited sim pods OOM-restarting under pressure.
`vllm:kv_cache_evictions_total` was added to the fork for this evidence (the direct
eviction-pressure signal, distinct from the pinned-eviction subset).

Reproduce:

```
uv run bench report --compare prefix-affinity --compare class-aware-reliability \
  --workload sessions --sessions 50 --turns 6 --qps 5 --concurrency 10 \
  --tool-reliability --seed {0,1,2}
```

## Results

### 1. Session A/B, cache un-saturated (4096 blocks/pod), 3 seeds

| seed | arm | success | req/s | TTFT p50 | TTFT p99 | zero-recomp | pinned peak % | pinned evict |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0 | prefix-affinity | 100.0% | 2.5 | 0.061 | 0.073 | 25% | 0.0% | 0 |
| 0 | class-aware | 99.0% | 2.5 | 0.061 | 0.932 | 24% | 6.1% | 0 |
| 1 | prefix-affinity | 100.0% | 2.3 | 0.060 | 0.070 | 24% | 0.0% | 0 |
| 1 | class-aware | 97.3% | 2.2 | 0.060 | 6.925 | 25% | 6.5% | 0 |
| 2 | prefix-affinity | 100.0% | 2.2 | 0.061 | 0.076 | 26% | 0.0% | 0 |
| 2 | class-aware | 97.7% | 2.2 | 0.067 | 2.546 | 27% | 6.6% | 0 |

The mechanism works end-to-end: the on-arm pins 6.1–6.6% of cache while the off-arm pins
exactly 0%, with no client-supplied directives anywhere. But **retention delivers no
reuse benefit on this workload** — zero-recompute is statistically indistinguishable
between arms (24–27% both), and per-turn `cached_tokens` coverage is ~88–100% on *both*
arms. **The eviction counter (added after these three seeds, confirmed on a seed-0 repeat)
reads 0 on both arms with a cache peak of ~5.5%**: the working set never fills a 4096-block
cache, so nothing is evicted and there is simply nothing for retention to protect. The
directive re-states what LRU would do for free, changing occupancy accounting without
changing outcomes — while adding tail latency (on-arm TTFT p99 0.9–6.9 s vs off-arm
≤ 0.08 s) and a 1–3% success-rate cost from the extra work.

### 2. Session A/B under cache pressure (512 blocks/pod), 3 seeds

Shrinking the cache to 512 blocks/pod forces real eviction, so the eviction and
contention counters become meaningful:

| seed | arm | success | zero-recomp | cache peak % | evictions | pinned peak % |
|---|---|--:|--:|--:|--:|--:|
| 0 | prefix-affinity | 99.7% | 25% | 44.7% | 3650 | 0.0% |
| 0 | class-aware | 97.0% | 35% | 44.7% | 1689 | 49.6% |
| 1 | prefix-affinity | 100.0% | 24% | 44.3% | 3600 | 0.0% |
| 1 | class-aware | 96.3% | 25% | 43.8% | 3007 | 47.7% |
| 2 | prefix-affinity | 95.0% | 33% | 42.4% | 1424 | 0.0% |
| 2 | class-aware | 98.7% | 25% | 45.9% | 3773 | 61.5% |
| — | **prefix-affinity mean** | 98.2% | 27% | 43.8% | **2891** | 0.0% |
| — | **class-aware mean** | 97.3% | 28% | 44.9% | **2823** | 52.9% |

This is the direct measurement the earlier evidence could only infer. The on-arm pins
**~53%** of the cache, yet **total eviction churn (~2.8 k, means within 2%) and zero-recompute
(~27–28%) are statistically identical to the off-arm** — with large per-seed variance
(evictions swing 1.4 k–3.8 k on *both* arms, driven by which memory-limited sim pods
OOM-restart). Pinning half the cache neither reduces eviction pressure nor improves reuse:
the pins protect the confidently-short-gap prefixes an LRU already retains, while the churn
from the non-reused working set is unaffected. (One seed showed the on-arm evicting far
*less* — 1689 vs 3650 — which looked like a halving until the other two seeds cancelled it;
the counter's value here is precisely that it exposed the variance rather than letting a
lucky seed stand.) Cache-peak parity (~44% both) confirms the contention level is the same;
the directives only relabel which blocks are protected.

Pushing further — the same 512-block cache at the saturating experiment rate (qps 10,
concurrency 20) — tips the on-arm into the **over-pinning collapse** the offline sim
documents as a known failure mode[^4]: success fell to 88.7% (HTTP 504s) and throughput to
1.7 vs 2.5 req/s, pinning ~52% of cache with no reuse gain. So the directive's effect ranges
from inert (large cache) through neutral-but-costly (moderate pressure) to net-harmful
(saturation) — never a win on this workload.

### 3. Non-agentic guardrail (single-shot, 300 req, qps 10)

| arm | success | req/s | TTFT p50 | TTFT p99 | E2E p50 |
|---|--:|--:|--:|--:|--:|
| prefix-affinity | 100.0% | 9.9 | 0.061 | 0.068 | 0.275 |
| class-aware | 100.0% | 9.8 | 0.065 | 0.075 | 0.258 |

On unannotated single-shot traffic the directive path adds no measurable overhead: matched
throughput (9.8 vs 9.9 req/s), both arms 100% success, TTFT/E2E within noise. One-shots are
marked evict-first, which does not pin (0% pinned both arms). The retention machinery is
inert when nothing asks for retention.

## What this says for #37003

The prototype's headline for the upstream design is a **caution about `pinned`/high
retention without pressure-aware accounting**: a router that pins on a short-gap prediction
can, under contention, do net harm because it competes with the very LRU behavior that
already serves short-gap reuse. Two design implications fall out directly:

- The value of router-side retention is concentrated in **long-gap** reuse (a session that
  idles long enough for LRU to evict it, then returns) — but that is exactly the regime
  where a timing-only predictor is least confident. A retention API is only as good as the
  scope's ability to predict, and the server needs a cheap way to **bound the damage** of a
  wrong pin.
- That damage bound is a TTL on every retained block and a below-normal (`evict-first`)
  tier, which motivates two of the divergences below.

### Divergences from #37003 (each intentional)

1. **`evict-first` / below-normal marking** — a third queue drained before the LRU free
   list. The routing layer needs a way to mark one-shots as *worth less* than unmarked
   traffic; #37003 today can only retain harder, not de-prioritize. The guardrail run marks
   one-shots evict-first with zero throughput cost, and it is the natural relief valve for
   the over-pinning in §2.
2. **Demotion by the router** — the reset-to-unmarked rule and the router ceiling both
   require downgrading blocks a scope does not own, which #37003's "only the owning scope
   downgrades" rule forbids. Proposed resolution: a privileged **infrastructure scope** for
   the routing layer, which sees fleet-wide pressure no single orchestrator can.
3. **TTL clock** — wall-clock from receipt, uniform across priorities, vs the sketched
   seconds-after-last-access. Upstream marks expiry semantics TBD, so this is a proposal.
4. **`pinned` requires a TTL** — stricter than the persistent `duration=null` example.
   §2 is the empirical argument: unbounded persistence lets a mispredicted pin hold cache
   until manual intervention; a mandatory TTL makes over-pinning self-healing. Unbounded
   persistence should be server policy, not a client right.
5. **Compact router header** — a flat `x-kv-cache-priority: <int>[; ttl=…][; scope=…]`
   grammar alongside the upstream JSON `X-KVCache-Retention-Policy`, normalized server-side.
   If upstream prefers, the router emits the JSON header and this divergence disappears.

## Future work (not in this evidence)

- **Routing-weight interactions** (RFC §5 "v2"): prefer live-pin nodes and penalize
  high pinned pressure. Deferred because the pinned kgateway ignores EPP endpoint picks on
  this testbed, so it cannot be shown live here; the direction is decided, the weights are
  open.
- A **long-gap reuse workload** and a sim whose TTFT derives from uncached tokens, to
  measure the latency win this testbed's fixed-latency model cannot express.
- Pressure-aware pin admission (cap concurrent live pins; calibrate confidence thresholds
  against the pinned-eviction counter) — the mitigation the §2 result argues for.

[^1]: [`rfc-0001-phase-5-plan.md`](./rfc-0001-phase-5-plan.md) — the evidence-run plan,
    arm design, and the kgateway/endpoint-pick and simulator caveats.
[^2]: vllm#37003 — the upstream `RetentionDirective` RFC this feedback targets.
[^3]: [`rfc-0001-kv-cache-priority-directives.md`](./rfc-0001-kv-cache-priority-directives.md)
    — this repo's RFC; §2 (header grammar), §5 (EPP emission), §7 (the divergence list
    reproduced above).
[^4]: [`policies.md`](./policies.md) — "Known issue — over-pinning under moderate cache
    pressure": the offline `bench simulate` repro (tool hit rate 0.80 vs 0.93, 5209
    pinned-evictions) that §2 confirms end-to-end.
