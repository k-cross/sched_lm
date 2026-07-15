# RFC 0001: KV-cache retention directives from llm-d to vLLM

- **Status**: Draft — open for async iteration
- **Created**: 2026-07-15 (revised same week to align with upstream RFC vllm#37003 and its design doc)
- **Scope**: this bench (prototype) → upstream `llm-d-router` / `vllm`
- **Prototype targets**: forked `llm-d-inference-sim` standing in for vLLM; custom EPP plugin standing in for llm-d-router changes

## Summary

Add a per-request cache-retention directive path from llm-d's endpoint picker (EPP) to
the vLLM backend, plus richer cache stats flowing back — prototyped end-to-end in this
bench. It aligns with two upstream efforts:

- **vllm#37003**[^1] and its design doc[^2] — a live RFC (working
  implementation, feedback window open) defining a `RetentionDirective` API:
  token-range scoped, numeric `priority` 0–100, optional `duration` TTL, attached via
  `extra_body` or an `X-KVCache-Retention-Policy` JSON header, with an optional
  `retention_scope` (session ID). Its two-queue evictor (LRU free list for unmarked
  blocks + priority min-heap for marked ones) is the vLLM-side mechanism.
- **The llm-d "Agentic Runtime" vision doc**[^3] — the umbrella: make the KV-cache
  "the primary medium of coordination" for stateful agentic programs. Frameworks taint
  requests with program IDs, priorities, and context lifetimes; llm-d manages
  lifecycle; vLLM executes. #37003 is its **PoC 2** (context-aware eviction); this repo
  is already a lightweight **PoC 3** (workload-driven benchmarking with program-level
  metrics).

**What this RFC adds on top of #37003** — a third hint origin. #37003's principle is
"orchestrator defines policy; vLLM executes," covering hints the *framework* already
knows (system prompts, tool definitions, session lifecycles). But the router sees what
no single orchestrator can: cross-tenant traffic, fleet-wide cache state, and *learned*
workload regularities. This bench's `ClassAwareReliability` policy is the concrete
example — it learns per-tool re-arrival gaps (EWMA mean + variance, from timing alone)
and predicts when a paused session will return. That knowledge lives in the EPP, not the
client. So:

- **Frameworks** originate *semantic* directives (this range is a ToolDefinition, pin
  it for the session) — the #37003 `extra_body` path, passed through the gateway.
- **llm-d** originates *learned* directives (this tool-call gap will be ~2s, keep the
  prefix warm; this request is a one-shot, evict it first) — injected by the EPP as a
  request header, reconciled with any framework directive.
- **vLLM** honors both as hints (never correctness-affecting) and reports occupancy and
  eviction-pressure stats so the layers above can tell whether their directives help.

Where this design departs from #37003's current draft, the difference is called out
inline and collected in [§7](#7-divergences-from-37003-feedback-to-file) as feedback to
file while the upstream window is open — nothing here is a silent fork.

## Motivation

The workload evidence in #37003 matches what this bench's offline simulator was built to
study: in agentic loops, >90% of a turn's tokens are reused prefix, yet 40–60% of
session wall time is spent paused on tool calls — exactly the window in which LRU, blind
to the session's imminent return, evicts the prefix under competing load. Resume becomes
a full recompute of a 70K–200K token context. Retention value is highly skewed — an
Alibaba production study found 10% of KV blocks account for 77% of reuses[^4] —
and systems that exploit it report large wins: Continuum, 1.12–3.66× delay reduction on
SWE-Bench[^5]; KVFlow, up to 2.19× speedup under concurrent multi-agent
load[^6]; Marconi, up to 34.4× higher token hit rates and 71.1% lower
TTFT[^7].

Locally, the same gap is documented in this repo[^8]: the offline sim's
`class-aware-reliability` policy soft-pins prefixes (`retain_until` in
`src/bench/sim/node.py` / `policies.py`) and wins on tool-session hit rate, but the live
stack has no way to carry that decision to the backend, and the harness scrapes back
only two prefix-cache counters.

Earlier prior art and why it stalled — and why #37003 / this design don't:

- vllm#23083[^9] (pinned prefixes, closed not-planned): unbounded pins are an
  OOM/starvation footgun. Here and in #37003, priorities are **leases** — TTL'd,
  server-capped, soft under pressure — and the server exports a pressure metric.
- vllm#44775[^10] (`cache_control` markers, closed): body-schema change without
  benchmark evidence. #37003 ships an implementation; this bench supplies the
  router-driven half of the evidence (PoC 3's role).

## Design

### 1. Directive schema: adopt #37003's, with one extension

To stay convergent with upstream, the prototype adopts the `RetentionDirective` shape as
its canonical vocabulary rather than inventing a parallel one:

```
RetentionDirective:
  start: int              # token index, inclusive
  end: int | null         # token index, exclusive; null = end of prompt
  priority: int           # 0–100
  duration: float | null  # TTL seconds; null = persistent (server-capped)
retention_scope: string   # session/program identity (vision doc: program-id taint)
```

**On the wire, priorities are strictly numeric** — the 0–100 scale plus the −1
extension below. The named classes below are **documentation aliases only**: prose,
policy code, and dashboards, never transport.

| class | priority | notes |
|---|---|---|
| `pinned` | 100 | TTL mandatory; server-capped (`--kv-cache-priority-max-ttl`, default 5m) |
| `high` | 50 | TTL optional (`--kv-cache-priority-default-ttl=30s`) |
| `normal` | — | unmarked; plain LRU |
| `evict-first` | −1 (extension) | see below |

Requiring a TTL for `pinned` is stricter than #37003's static-retention example
(priority 100, `duration=null`): the lease rule is what answers #23083's OOM objection,
so unbounded persistence is a server-side policy decision, not a client right
(divergence, §7).

**Extension: `evict-first` / ephemeral blocks.** #37003's two-queue evictor can only
retain *harder* than LRU — it cannot express "this block is worth *less* than unmarked
traffic." (Terminology trap: upstream describes priority 0 as "evict first," but a
marked priority-0 block still outlives every *unmarked* block, since the LRU free list
drains before the priority heap.) Our `evict-first` (−1) is genuinely below-normal: a
third bucket drained *before* the LRU free list. The vision doc's PE IV explicitly
wants this (early-evict intermediate `ReasoningBranch` thoughts; one-shot pollution in
this bench's workload model). Among the candidate shapes — dedicated queue, negative
priorities in the existing heap, or a boolean `ephemeral` flag — the dedicated queue
wins: negative priorities force every eviction through the heap, and the flag, while
simplest, still needs its own drain order anyway. To be filed as #37003 feedback — one
more queue, same zero-overhead-for-unmarked property.

Directive semantics (matching #37003 where possible + our lease rules):

- TTLs apply uniformly across all priority levels and count down in **wall-clock** time
  from directive receipt. Expiry demotes the block to unmarked — no unpin message
  needed. (#37003's design doc sketches `duration` as seconds *after last access* and
  marks expiry semantics TBD; wall-clock is our position to file as feedback, §7.
  Dynamic TTL extension can layer on later.)
- A later request touching the same blocks with no directive resets them to unmarked
  (mirrors the offline sim's `Node.admit` pin-clearing). Diverges from #37003's
  block-stamping rule — "any scope can escalate; only the owning scope can downgrade or
  clear" — see §7.
- Malformed directives are logged once and ignored — a hint never fails a request.
- Pins are soft under pressure: if every evictable block is marked and unexpired, evict
  the soonest-expiring and count a **pinned-pressure eviction** (the router's signal
  that it over-pinned). Capacity is never held hostage.

**Block sharing.** #37003 stamps priority on the *block*, not the request, because
blocks are shared: the same system-prompt blocks may back many sessions across many
`retention_scope`s. A whole-prefix router directive therefore touches blocks other
scopes depend on. The prototype's rule: a shared block carries the *maximum* of the
live directives covering it, and the reset-to-unmarked rule applies only when *no* live
directive covers the block. This matches upstream's escalate-freely posture while
making the demotion cases (reset, router ceiling) explicit divergences to negotiate
(§7); upstream's own open question on shared-block queue assignment is the right thread.

**Decode tokens are out of scope for the router path.** #37003 pairs
`RetentionDirective` with a `DecodeRetentionPolicy` for tokens generated during
decoding. The router only reasons about the prompt prefix (what its learned signals
describe), so decode-token retention stays framework-originated end-to-end.

### 2. Two transports, by hint origin

**Framework path (upstream-native):** `RetentionDirective`s in `extra_body` +
`retention_scope`, per #37003. The gateway and EPP pass these through untouched;
token-range scoping works here because the framework knows its segment boundaries
(SystemPrompt vs ToolDefinition vs ReasoningBranch — vision PE III).

**Router path (this RFC's addition):** a request header injected by the EPP after
scheduling:

```
x-kv-cache-priority: <int>[; ttl=<Go duration>][; scope=<id>]
```

- Scope is the **whole prefix** of the request it rides on — the router cannot see
  semantic segment boundaries, and the request's block-hash chain is the unit its
  learned signals (re-arrival gaps, request class) actually describe.
- A header because the extProc already runs in headers-SEND mode (no data-plane
  changes); mutating `extra_body` from the EPP would mean body buffering + rewriting on
  every request. #37003 already defines a JSON header (`X-KVCache-Retention-Policy`);
  we use a separate compact one because the router's directive is a fixed three-field
  tuple — flat `key=value` parsing stays trivial on both sides, no JSON encoding in the
  extProc. Whether router hints should instead reuse the upstream header is worth
  raising as #37003 feedback (§7); either way the backend normalizes both transports
  into the same internal representation.
- **Precedence — router ceiling.** When a framework directive and the router header
  cover the same blocks, the **router wins downward**: its value caps the effective
  priority, so a greedy or malicious client cannot pin harder than the fleet can
  afford. A bounded overshoot threshold lets a framework exceed the ceiling by a fixed
  margin — enough that well-behaved clients with genuinely better information aren't
  clamped, without per-client abuse tracking. This requires the router to *demote*,
  which #37003's scope-ownership rule forbids for non-owning scopes — the upstream ask
  is a privileged infrastructure scope (§7).
- Session identity: `retention_scope` when the client supplies it; otherwise the router
  derives a fallback scope from the prefix-hash chain its indexer already tracks. The
  derived fallback is what upstream likely needs specified — it works with zero client
  cooperation.
- EPP derives `ttl` from its retention horizon: `ttl = retain_until − now`.

### 3. Eviction mechanism

Normative semantics (the offline sim in `src/bench/sim/node.py` is the executable
spec): effective rank `evict-first < unmarked < priority ascending`, TTL expiry
collapses to unmarked; victim = min `(effective_rank, LRU position)` among evictable
blocks; all-marked → evict soonest-expiring + bump the pressure counter.

The sim fork mirrors #37003's proposed vLLM structure so the prototype exercises the
same mechanism upstream would ship: existing LRU `free_block_queue` for unmarked
blocks, a priority min-heap keyed `(effective_priority, last_freed_time)` for marked
ones, plus the evict-first bucket (our extension) drained first. Zero allocations or
checks for unmarked blocks — preserving #37003's "no overhead for non-agentic
workloads" principle.

### 4. Backward channel: stats

**Prometheus** (following the sim's existing `vllm:` naming and `model_name` labeling):

| Metric | Type | Meaning |
|---|---|---|
| `vllm:kv_cache_priority_blocks{priority=...}` | gauge vec | resident marked blocks per class/priority band |
| `vllm:kv_cache_pinned_usage_perc` | gauge | marked-and-unexpired blocks / max blocks |
| `vllm:kv_cache_pinned_evictions_total` | counter | marked blocks evicted under pressure |

**ZMQ KVEvents**: append trailing fields to the existing `BlockStored` event — array
positions `[12] priority (int|nil)`, `[13] retain_until (float64 unix seconds|nil)`;
map-format keys `priority` / `retain_until`. Verified constraint: the deployed
`llm-d-kv-cache@v0.9.0` adapter[^11] ignores extra trailing fields on known
events but **fails the whole batch on an unknown event tag** — so no new event types in
v1; priority changes re-emit `BlockStored` (idempotent upsert, batched to manage update
volume), expiry computed by consumers from `retain_until`. This lines up with PoC 2's
"track semantic block types in the distributed indexer": the same two fields are what
the indexer needs. Typed semantic tags (SystemPrompt / ToolDefinition / ReasoningBranch
/ ExecutionCheckpoint — vision PE III) follow the same path: piggyback as trailing
fields first, then migrate to a versioned `BlockPriorityChanged` / semantic-tag event
once the API is stable and consumers negotiate, with a standard deprecation path.

**Bench metrics (PoC 3 alignment)**: beyond per-request TTFT, `bench report` should
surface the vision's program-level measures — session (program) completion time,
zero-recompute hit rate for returning sessions, pinned-cache utilization, and **TTFT by
turn position** (flat = cache failure; decreasing = prefix survival), matching #37003's
planned PoC evaluation. Every hint-on/off A/B run also reports **throughput on the
non-agentic control traffic** as a guardrail: #37003's first design principle is zero
overhead for unannotated workloads, and the evidence should demonstrate it, not assume
it.

### 5. EPP emission mechanism (llm-d-router side)

Verified against `llm-d-router v0.9.2` source: **no router patch is needed** for the
header path.

- A plugin implementing `requestcontrol.PreRequest` runs after scheduling
  (`pkg/epp/requestcontrol/director.go`) and before header serialization.
- `SchedulingRequest.Headers` aliases the request-context header map, and
  `handlers/request.go generateHeaders` serializes every non-system-owned header into
  the extProc `HeaderMutation` sent to Envoy → backend. `x-kv-cache-priority` is not
  system-owned.
- Plugins declared in `EndpointPickerConfig` are auto-wired into request control when
  they type-assert to `PreRequest`.

The prototype plugin (`src/gateway-plugin/`): classify the request (tool / rag /
oneshot, reusing `class_aware_reliability.go`), track per-tool re-arrival gaps (EWMA
mean + variance, matching `ToolGapIndex` in `src/bench/sim/policies.py`), and emit
`50; ttl=<mean·margin>; scope=<session>` on confident short predicted gaps, `-1` for
one-shots. Session identity per §2 (client `retention_scope` / `x-session-id`, falling
back to the router-derived prefix-hash scope).

**Routing interactions (v2).** Once the indexer sees pin state via the trailing event
fields, scorers should (a) prefer nodes holding a live pin for the session's prefix and
(b) penalize nodes reporting high pinned pressure (`vllm:kv_cache_pinned_usage_perc` /
`pinned_evictions_total`) — a node with no headroom to evict is a bad place to send new
work. Both directions are decided; their *weights* relative to existing scorers are an
open question.

### 6. Division of labor with the vision's pillars

| Vision pillar | This RFC's coverage |
|---|---|
| PE I program-aware scheduling | out of scope (orthogonal per #37003: block priority ≠ request priority) |
| PE II proactive state movement (Move/Pin/Evict control plane) | out of scope; the bench's Tier-2 KV-transfer gap is the local analog — future work |
| PE III semantic KV-cache (typed blocks) | partially: framework path carries ranges; typed tags deferred to v2 events |
| PE IV context-aware eviction | core of this RFC (= PoC 2, + the evict-first extension) |
| PE V workload-driven benchmarking | this repo is the instrument (= PoC 3); session workload + reliability signal are the canonical scenario ("Use Case 2: multi-turn ReAct loops") |

### 7. Divergences from #37003 (feedback to file)

Collected so the upstream conversation has one list; each is intentional, and each
lands as a comment on #37003 while its window is open (prototype plan step 5):

1. **`evict-first` / below-normal marking** (§1): a third queue drained before the LRU
   free list. PE IV needs *some* way to mark blocks worth less than unmarked traffic;
   #37003 currently can only retain harder.
2. **Demotion by the router** (§1, §2): the reset-to-unmarked rule and the router
   ceiling both require downgrading blocks a scope doesn't own, which #37003's "only
   the owning scope can downgrade" rule forbids. Proposed resolution: a privileged
   infrastructure scope for the routing layer — it sees fleet-wide pressure no single
   orchestrator can.
3. **TTL clock** (§1): wall-clock from receipt, uniform across priorities, vs the
   design doc's sketched seconds-after-last-access. Upstream marks expiry semantics
   TBD, so this is a proposal, not a fork.
4. **`pinned` requires a TTL** (§1): stricter than the persistent `duration=null`
   example; unbounded persistence should be server policy, not a client right.
5. **Compact router header** (§2): a flat header grammar alongside the upstream JSON
   `X-KVCache-Retention-Policy`, normalized server-side — or, if upstream prefers, the
   router emits the JSON header and this divergence disappears.

## Prototype plan (this repo)

Phased so each step is independently testable; EPP work is last (most uncertainty).

1. **Offline sim parity** — extend `Placement`/`Node` from `retain_until`-only to the
   priority vocabulary (`src/bench/sim/{node,cost,engine,policies}.py`); the sim
   becomes the executable spec. Backward compatible; existing tests pass unchanged.
2. **Sim fork honors directives** — fork `llm-d/llm-d-inference-sim` (submodule
   `third_party/`); implement the two-queue + evict-first evictor in `pkg/kv-cache/`;
   parse the header first, `extra_body` `RetentionDirective`s second (header needs no
   body plumbing). Client-side `--kv-priority` escape hatch in `bench traffic` proves
   the path before any EPP work.
3. **Stats** — §4 metrics + trailing event fields in the fork (compat encoded as a fork
   test against the real `llm-d-kv-cache@v0.9.0` adapter); `bench metrics`/`report`
   grow pinned columns, hint-on/off A/B, session-completion-time / zero-recompute /
   TTFT-by-turn measures, and the non-agentic throughput guardrail.
4. **EPP emission** — custom EPP image with the `PreRequest` plugin; rewrite
   `infra/llm-d/values.yaml` to the real GIE chart schema (current keys don't match the
   chart and are likely inert). Discovery first: dump the live EPP config; confirm
   whether body-NONE extProc mode starves classification (may need
   `requestBodyMode: BUFFERED` on the class-aware route only).
5. **Evidence + upstream feedback** — `bench report` comparing `prefix-affinity` vs
   `class-aware-reliability` on session workloads (tool-session hit rate, pinned
   gauges, pressure counter, session completion time, non-agentic regression). Post
   results + the §7 divergence list to #37003 while its feedback window is open.

## Open questions (iterate here)

1. **Router-ceiling threshold**: how much may a framework directive exceed the router's
   ceiling before it is clamped? A fixed margin avoids per-client abuse tracking, but
   the right magnitude needs the fairness simulation below to calibrate.
2. **Routing weights (v2)**: preferring live-pin nodes and penalizing high pinned
   pressure are decided directions (§5), but the weighting against existing scorers is
   unclear — as is the shape of the calculation combining all routing factors. Needs
   the bench's A/B machinery pointed at routing, not just eviction.
3. **Multi-tenant fairness**: per-client or per-`retention_scope` pin budgets, or is
   max-TTL cap + pressure metric enough for v1? (#37003 is silent on budgets.)
   Important but deferred — needs dedicated fairness simulation work; revisit after the
   prototype phases land. Early evidence this is real, not hypothetical: the phase-1 sim
   already shows `class-aware-reliability` losing tool-class hit rate to plain
   `class-aware` at some cache sizes — concurrent sessions' pins evict each other under
   contention, worse than letting LRU pick the true least-recently-used block[^12]. The
   `pinned_evictions` counter (§4) tracks it; whether a pressure metric alone is enough
   to self-correct, or a budget is needed, is exactly what this open question asks.

[^1]: [vllm-project/vllm#37003 — "[RFC]: Context-Aware KV-Cache Retention API
    (Prioritized Evictions)"](https://github.com/vllm-project/vllm/issues/37003).
    Feedback window opened 2026-03-13; working implementation and early benchmarking
    exist.

[^2]: M. Ayoub, ["Context-Aware KV-Cache Retention API" — design doc behind
    #37003](https://docs.google.com/document/d/1kRKAZBG7te38tqv9Twxyyc-Pdkk2JZPm7gqHHnNhFpE):
    evidence against LRU, design principles (zero overhead for non-agentic workloads;
    orchestrator defines policy, engine executes; API as the product), two-structure
    evictor, tiered retention model, `DecodeRetentionPolicy`, planned PoC evaluation.

[^3]: ["llm-d as the Agentic Runtime" — Northstar vision
    doc](https://docs.google.com/document/d/1C3wRYLSZ9GPT2454MvDC6-8T_gvMPsPL_-ISVTZj4FU):
    five pillars (PE I–V), three PoCs. This RFC ≈ PoC 2 at the llm-d↔vLLM boundary;
    this repo ≈ PoC 3.

[^4]: Alibaba production KV-cache workload study (USENIX ATC '25), as cited in
    the #37003 design doc[^2]: 10% of KV blocks account for 77% of all reuses;
    workload-aware eviction yields up to 41.9% latency reduction vs LRU.

[^5]: Continuum — TTL-based KV retention for tool-calling agents
    ([arXiv:2511.02230](https://arxiv.org/abs/2511.02230)): 1.12–3.66× delay reduction
    on SWE-Bench.

[^6]: KVFlow — workflow-aware KV eviction for multi-agent serving
    ([arXiv:2507.07400](https://arxiv.org/abs/2507.07400)): up to 2.19× speedup over
    LRU under concurrent multi-agent workloads.

[^7]: Marconi — prefix caching for hybrid LLMs (MLSys '25,
    [arXiv:2411.19379](https://arxiv.org/abs/2411.19379)): up to 34.4× higher token hit
    rates, 71.1% lower TTFT.

[^8]: This repo: README §"Promoting a policy to real llm-d routing" (and the
    cache-pinning limitation note it replaces); `docs/policies.md` —
    `class-aware-reliability` soft-pin design (offline).

[^9]: [vllm-project/vllm#23083 — pinned prefix
    caching](https://github.com/vllm-project/vllm/issues/23083), closed not-planned.

[^10]: [vllm-project/vllm#44775 — `cache_control` retention
    markers](https://github.com/vllm-project/vllm/issues/44775), closed.

[^11]: [llm-d KV Cache Manager (architecture
    docs)](https://llm-d.ai/docs/architecture/Components/kv-cache-manager) and
    [llm-d/llm-d-kv-cache](https://github.com/llm-d/llm-d-kv-cache) — the existing
    backward channel this RFC extends.

[^12]: `docs/policies.md` §"Known issue — over-pinning under moderate cache pressure",
    reproducible via `bench simulate` with 3 nodes, `--cache-blocks 200`,
    `--mix tool=1.0`, `--tool-reliability --seed 21`.
