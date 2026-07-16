# Routing Policies

The offline simulator compares several routing policies. Every policy shares the same signature, returning a placement decision (node, transfer, and regime). They differ in the *choice set* and the *information* they use.

## Baseline Policies

* **`round-robin`**: Cyclic node assignment. Never transfers. The load-balancing baseline.
* **`cache-local`**: Pure prefix-affinity. Always routes to the "hot" node (the node with the longest matching prefix) and never transfers KV blocks. It ignores queue depth and will always wait for a cache hit.

## Production (Tier 0)

These policies replicate the exact behavior of `llm-d`'s shipped EPP pipeline: a saturation filter (drops nodes queued past a threshold to prevent herding), followed by weighted scorers (prefix affinity ×2, load ×1), and an argmax. They never transfer.

* **`weighted-precise`**: Reads the true node cache state (llm-d's precise-prefix-cache scorer).
* **`weighted-approx`**: Scores from a router-side index of past routing decisions that never sees node-side evictions (llm-d's approximate prefix-affinity scorer).
* *Purpose:* The routing-regret gap between `approx` and `precise` isolates the cost of stale cache beliefs (information), while the gap between `precise` and `oracle` isolates the cost of missing the transfer arm (policy). The TTFT gap shows how far each is from the theoretical best (`oracle-belady`).

## Custom Plugin (Tier 1)

* **`class-aware`**: Classifies each request from observables only (message shape and sizes) without ground-truth labels. It applies different strategies per class:
  * *Tool-session turns:* transfer-aware scoring.
  * *RAG queries:* doc-prefix affinity with saturation spill.
  * *One-shots:* least-loaded node (never chase cache).
* *Purpose:* Represents a policy that can be implemented today as a custom Go plugin in `llm-d`.

* **`class-aware-reliability`**: `class-aware` plus a re-arrival-gap signal for tool sessions. The router keeps a small, capacity-bounded table keyed by tool signature, holding the EWMA **mean and variance** of the inter-turn gap observed after that tool is called — learned from **timing alone** (the gap between a conversation's consecutive turns), never from the tool's success/failure (which the gateway often cannot observe). When a tool-session turn's most recent tool call **confidently** (low variance) predicts a **short** return, the chosen node's copy of that prefix is marked `priority=HIGH` (RFC-0001 §1) with a lease until the session comes back — turning a would-be recompute into a cache hit. One-shots are marked `priority=EVICT_FIRST` (no reuse, pure pollution). A long, unpredictable (high-variance), or unseen tool adds no retention; unseen tools fall back to static priors, so cost stays `O(#tools)`.
* *Purpose:* Same shippability as `class-aware` (the EPP sees every request arrival and serves every turn, so it can aggregate per-tool gaps at a single point) while exploiting a signal orthogonal to prefix shape: *when* a session will return.
* **Known issue — over-pinning under moderate cache pressure.** `class-aware-reliability`'s tool-class hit rate is not monotonically ≥ `class-aware`'s: at some cache sizes it is measurably *worse* (e.g. 3 nodes, `--cache-blocks 200`, `--mix tool=1.0`, `--tool-reliability --seed 21`: tool hit rate 0.80 vs. plain `class-aware`'s 0.93). The mechanism is that `Node._pick_victim` (`src/bench/sim/node.py`) sacrifices the soonest-expiring pin once every resident block is marked-and-unexpired — under contention that can mean *other* sessions' pins evict each other before any of them return, a worse outcome than letting plain LRU pick the true least-recently-used block. `Node.pinned_evictions` (summed into `SimResult.pinned_evictions`, RFC-0001 §4's "pinned-pressure eviction" signal) is high exactly in this regime and near-zero once the cache is large enough to hold every pin without contention — it is the leading indicator that retention is doing more harm than good. Surfaced as the **Pinned evict** column in `bench simulate`'s report table (RFC-0001 prototype-plan phase 3, "Stats"): 5209 for `class-aware-reliability` on the repro above vs. 0 for plain `class-aware`, dropping to 0 once the cache is large enough to hold every pin. Calibrating `--retain-margin` / confidence thresholds against `pinned_evictions`, or capping how many sessions may hold a live pin at once, is open work[^1].

## Hypothetical / Experimental (Tier 2)

These policies model a KV-block transfer regime that is **not currently expressible** in the stock `llm-d` stack because there is no inter-replica KV-transfer path. They represent hypothetical future infrastructure (e.g., a vLLM KV connector tier).

* **`transfer-aware`**: An `argmin` over `(node, transfer)` scored on *approximate* state (queue depth rounded to whole requests). It models a realistic router trying to utilize KV transfers without perfect information.
* **`oracle`**: An `argmin` over `(node, transfer)` with perfect current state and standard LRU cache eviction. Provides a routing-quality lower bound — every policy's *routing regret* is measured against it — but its LRU caches can still make sub-optimal eviction decisions.
* **`oracle-belady`**: Same greedy routing as `oracle`, but nodes evict by **Belady's MIN** (look ahead in the request stream and evict the block whose next use is furthest away) instead of LRU. This is the simulator's *strongest reference baseline* — near-best routing **and** near-best eviction — and the *TTFT gap* of every other policy is measured against it. It is an **estimate**, not a proven lower bound: Belady's MIN is optimal only for a single reference stream and a single cache, whereas here `future_uses` is built globally (ignoring which node will serve each future request) and the metric credits only the *contiguous* matched prefix. So a policy with better cache management can occasionally post a small *negative* TTFT gap, and `oracle-belady`'s hit rate is expected — but not guaranteed by construction — to beat LRU's.

## Metrics

The report surfaces two distance metrics that answer different questions:

* **Routing regret (s)**: Per-policy. Each request's TTFT minus the greedy-oracle TTFT on that same simulation's own node state. Measures how well the policy *routes* given its own cache contents — not how good those cache contents are. Both `oracle` and `oracle-belady` show 0.000 because they both route optimally on their own state. Useful for isolating routing-algorithm quality (e.g. the gap between `weighted-approx` and `weighted-precise` is pure information regret).

* **TTFT gap (s)**: Cross-policy. Each policy's p50 TTFT minus the best oracle baseline's p50 (`oracle-belady` if run, else `oracle`). This is the single comparable measure of how far a policy is from that reference, capturing **both** routing quality and cache-management quality. For example, `class-aware-reliability` has a small routing regret but an even smaller TTFT gap because its cache pinning makes up for the routing imprecision. Because the baseline is an estimate (see `oracle-belady`), a strong policy can show a small *negative* gap — read it as "faster than the reference," not an impossibility.

[^1]: [`docs/rfc-0001-kv-cache-priority-directives.md`](rfc-0001-kv-cache-priority-directives.md)
    § "Open questions", item 3 (multi-tenant fairness).
