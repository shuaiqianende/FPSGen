# Research V2 Progress Log

This directory is the Git-tracked, reviewable status record for the FPSGen
Research V2 experiments.  Each update records the exact experiment scope,
configuration, observed results, decision, and the next action.  It is kept
separate from machine-local artifacts under `outputs/` and `experiments/`.

For the repository-wide map of active experiments, configurations, scripts,
and local artifact locations, see [PROJECT_INDEX.md](../PROJECT_INDEX.md).

## Conventions

- Commit only Markdown, source code, manifests, and small aggregate results.
- Do not commit checkpoints, cached point tensors, PLY files, TensorBoard
  events, or other large artifacts.
- Each update must distinguish a completed result from a running job.
- Do not claim a training method is effective until its fixed validation probe
  has completed; training loss is only a short-run stability signal.

## Updates

- [2026-09-28: Gate results and short-run status](2026-09-28_gate_and_short_run_status.md)
