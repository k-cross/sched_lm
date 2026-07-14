# Routing Policies

The offline simulator compares several routing policies. Every policy shares the same signature, returning a placement decision (node, transfer, and regime). They differ in the *choice set* and the *information* they use.

## Baseline Policies

* **`round-robin`**: Cyclic node assignment. Never transfers. The load-balancing baseline.
* **`cache-local`**: Pure prefix-affinity. Always routes to the "hot" node (the node with the longest matching prefix) and never transfers KV blocks. It ignores queue depth and will always wait for a cache hit.

## Production (Tier 0)

These policies replicate the exact behavior of `llm-d`'s shipped EPP pipeline: a saturation filter (drops nodes queued past a threshold to prevent herding), followed by weighted scorers (prefix affinity ×2, load ×1), and an argmax. They never transfer.

* **`weighted-precise`**: Reads the true node cache state (llm-d's precise-prefix-cache scorer).
* **`weighted-approx`**: Scores from a router-side index of past routing decisions that never sees node-side evictions (llm-d's approximate prefix-affinity scorer).
* *Purpose:* The regret gap between `approx` and `precise` isolates the cost of stale cache beliefs (information), while the gap between `precise` and `oracle` isolates the cost of missing the transfer arm (policy).

## Custom Plugin (Tier 1)

* **`class-aware`**: Classifies each request from observables only (message shape and sizes) without ground-truth labels. It applies different strategies per class:
  * *Tool-session turns:* transfer-aware scoring.
  * *RAG queries:* doc-prefix affinity with saturation spill.
  * *One-shots:* least-loaded node (never chase cache).
* *Purpose:* Represents a policy that can be implemented today as a custom Go plugin in `llm-d`.

## Hypothetical / Experimental (Tier 2)

These policies model a KV-block transfer regime that is **not currently expressible** in the stock `llm-d` stack because there is no inter-replica KV-transfer path. They represent hypothetical future infrastructure (e.g., a vLLM KV connector tier).

* **`transfer-aware`**: An `argmin` over `(node, transfer)` scored on *approximate* state (queue depth rounded to whole requests). It models a realistic router trying to utilize KV transfers without perfect information.
* **`oracle`**: An `argmin` over `(node, transfer)` with perfect current state. This provides the theoretical lower bound every other policy's regret is measured against.
