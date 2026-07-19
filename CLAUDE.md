# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`llm-d-emulation-bench`: a local LLM-routing benchmark harness that runs vLLM prefix-caching experiments on k3d (Kubernetes), managed with devenv/Nix. Python package lives in `src/bench/`; infra manifests in `infra/`.

## Environment

- The dev environment is a Nix/devenv shell auto-activated by `direnv` (`.envrc` → `use devenv`). Assume it is already loaded; if a tool is missing, the shell hasn't activated — run `direnv allow` / `devenv shell`.
- Requires a running container runtime (OrbStack/Docker/Colima) for k3d. Targets Apple Silicon (ARM64 simulator).
- `uv sync` runs automatically on shell entry; `KUBECONFIG` points at `.k3d/kubeconfig.yaml`.

## Commands

- Run Python CLI via `uv run`, never `python -m`: `uv run bench traffic|metrics|report ...` (entry point is `bench.cli:main`).
- Lint: `uv run ruff check src/ tests/` — Format: `uv run ruff format src/ tests/`.
- Cluster lifecycle (devenv scripts): `cluster-create`, `cluster-delete`, `cluster-status`, `deploy-monitoring`, `deploy-llmd`. Image build+import: `build-epp` (custom Gateway EPP), `build-sim` (llm-d-inference-sim fork).

## Code style

- Ruff-enforced, differing from defaults: line length **100**, double quotes, target py312. Lint set: E, F, W, I, B, C4, UP.

## Testing

- Run tests with `uv run pytest`. When adding non-trivial logic, add pytest tests in `tests/`. Don't scaffold a framework for trivial changes — `ruff check` is the baseline.

## Version control

- This repo uses **Jujutsu (`jj`)** as the primary VCS, with git as the backing store. Prefer `jj` commands for commits/history; avoid `git commit`/`git rebase` unless asked.

## Gotchas

- vLLM simulator model is hardcoded to `Qwen/Qwen2-0.5B` (`infra/llm-d/inference-pool.yaml`).
- Prometheus discovers targets via pod annotations (`prometheus.io/scrape`), not ServiceMonitors.
