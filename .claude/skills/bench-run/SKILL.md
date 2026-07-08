---
name: bench-run
description: Run the end-to-end vLLM prefix-caching benchmark — bring up the k3d cluster, deploy monitoring + llm-d, run traffic for each routing strategy, and print a comparison report. User-triggered because it provisions a cluster.
disable-model-invocation: true
---

Run the full benchmark workflow. Accept optional `$ARGUMENTS` to override requests/qps (default: 200 requests, 10 qps).

1. **Preflight**: confirm the container runtime is up (`docker info` or `orbctl status`). If the devenv shell isn't active, note that `direnv` should have loaded it.
2. **Cluster**: run `cluster-status`; if no cluster exists, `cluster-create`.
3. **Deploy**: `deploy-monitoring`, then `deploy-llmd`. Wait for pods to be Ready (`cluster-status` / `kubectl get pods`).
4. **Run traffic** for each routing strategy:
   - `uv run bench traffic --route round-robin --requests 200 --qps 10`
   - `uv run bench traffic --route prefix-affinity --requests 200 --qps 10`
   (substitute any values passed in `$ARGUMENTS`.)
5. **Metrics**: `uv run bench metrics` to fetch the prefix cache hit rate from Prometheus.
6. **Report**: `uv run bench report --compare round-robin --compare prefix-affinity` and summarize the TTFT/E2E and cache-hit differences for the user.

Do NOT run `cluster-delete` unless the user asks — the cluster persists between runs intentionally.
