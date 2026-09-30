# Development guide

## Supported environments

FPSGen keeps a conservative baseline environment and a separate PyTorch 2
research environment.  CUDA extensions must be built inside the environment
that will load them; do not reuse build artifacts across PyTorch versions.

| Purpose | Environment definition | PyTorch |
| --- | --- | --- |
| Baseline pipeline and released compatibility | `environment.yml` | 1.13.0 + CUDA 11.7 |
| Dense-BEV research and current Stage-1 runs | `env/environment_pt20_cu117.yml` | 2.0.1 + CUDA 11.7 |

Install extensions only after activating the selected environment.  See
[INSTALL.md](INSTALL.md) for the required build order.

## Fast validation

Run these checks from the repository root before changing a training path:

```bash
python -m py_compile fpsgen/train_bev.py fpsgen/train_teacher.py fpsgen/train_student.py
python -m pytest -q tests
```

For dense-BEV architecture changes, the focused suite is faster:

```bash
python -m pytest -q tests/test_bev_backbones.py tests/test_bev_backbone_architecture.py
```

GPU smoke runs additionally require `TRAIN_DATABASE`, a prepared dataset, and
the CUDA extensions.  They are not substitutes for the CPU test suite.

## Training/configuration conventions

- Keep dataset paths out of committed YAML.  Set `TRAIN_DATABASE` at runtime.
- Give every formal run a unique `experiment.id`; this isolates TensorBoard
  events and checkpoints under `experiments/<id>/`.
- Preserve historical defaults unless a research config explicitly enables a
  `runtime` or `train` optimization such as pinned memory or compilation.
- Never overwrite a running experiment directory.  Resume with `--checkpoint`
  or use a new experiment ID.
- For a new long-running BEV experiment, use a finite 16-step preflight before
  a multi-epoch launch and preserve the log alongside the formal run.

## Reproducibility conventions

All three training entry points use the shared `seed_training(42)` routine,
which seeds Python, NumPy, CPU/CUDA PyTorch and disables cuDNN benchmark mode.
DataLoader worker seeding remains opt-in through
`train.deterministic_worker_init`; this prevents changing historical runs.

## Reporting conventions

Use `train/loss_mse` for Stage-1 training-dynamics comparisons.  When scalar
logging is every 100 steps, report 500-step values as the mean of the five
scalar samples in that interval, and label incomplete windows explicitly.
Training loss alone is not a generation-quality result.
