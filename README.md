# FPSGen

Official implementation of **FPSGen: Flexible Point Cloud Scene Generation
with BEV-Supported Transport Flows** ([paper](https://arxiv.org/abs/2607.26645)).
FPSGen supports flexible LiDAR/vehicle/road condition combinations and includes
the complete three-stage training, unified inference, evaluation, and data
loading pipeline.

This repository contains source code and configuration templates only. Dataset
files, experiment logs, local checkpoints, generated point clouds, and server
paths are excluded by `.gitignore`. A release checklist is provided in
[`docs/PUBLISHING.md`](docs/PUBLISHING.md).

## Method overview

FPSGen decomposes large-scale point-cloud scene generation into:

1. **Flexible Condition BEV Flow Prior.** A conditional velocity field
   generates the structural prior \(B=[D,H,M]\), comprising normalized density,
   maximum height, and occupancy.
2. **Teacher Transport Mapping.** A teacher learns a source-indexed clean
   endpoint \(\mathcal P_1^\dagger\) from an independent BEV-sampled source
   \(\mathcal P_0\).
3. **Approximate-OT Point Flow.** The student learns the straight transport
   velocity from \(\mathcal P_0\) to \(\mathcal P_1^\dagger\), conditioned on
   the generated BEV prior and the active condition tuple.

At inference, BEV Flow and PointFlow are integrated sequentially with
classifier-free guidance and forward Euler updates. See
[the method-to-code guide](docs/METHOD.md) for notation, tensor conventions,
and implementation entry points.

## Installation

The reproducible environment is Python 3.9, PyTorch 1.13.0 with CUDA 11.7,
and MinkowskiEngine 0.5.4. Follow [the installation guide](docs/INSTALL.md) in
order: PyTorch/CUDA, MinkowskiEngine, Python packages, then the two local CUDA
extensions. The guide includes a fail-fast environment check.

## Local CUDA operators

`fpsgen/models/ChamferDistancePytorch/` is included in this repository under
its upstream MIT license. Build it in the active environment with:

```bash
pip install -e fpsgen/models/ChamferDistancePytorch/chamfer3D
```

- Chamfer distance is required by teacher training and point-cloud evaluation.
  Its local installation exposes `chamfer3D.dist_chamfer_3D.chamfer_3DDist`.
- PointNet2 is retained as an optional extension interface. The current
  three-stage FPSGen pipeline does not import it, but an environment that uses
  PointNet2-based auxiliary operations must provide the `pointnet2_ops` Python
  package and its compiled CUDA backend.
- Compile each operator against the same Python, PyTorch, CUDA toolkit, and C++
  compiler used to run FPSGen. Rebuild after changing any of those components.

Verify all runtime dependencies and the Chamfer forward/backward kernel before
training (the selected physical GPU is remapped to logical `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/check_environment.py
CUDA_VISIBLE_DEVICES=3 python scripts/verify_cuda_ops.py
```

The GPU number is only an example. Any available device can be selected by
changing `CUDA_VISIBLE_DEVICES`; inside the process it is always addressed as
`cuda:0`.

## Dataset preparation

The SemanticKITTI/KITTI-odometry root must contain sequence folders such as
`00/`, with `calib.txt`, `poses.txt`, and preprocessed `input_/` and
`gt_/` NumPy arrays. FPSGen training reads those arrays directly; dataset
preprocessing is intentionally outside this lean training repository.

Set the training dataset root without editing YAML. The provided configurations
intentionally contain no machine-specific paths:

```bash
export TRAIN_DATABASE=/path/to/KITTI_Odometry
```

See [dataset format](docs/DATASET.md) for the expected per-sequence files.
For a one-step, from-scratch training validation, see
[the reproducibility guide](docs/REPRODUCIBILITY.md).

## Training

Run the full three-stage training pipeline in order:

```bash
python -m fpsgen.train_bev --config configs/train_bev.yaml
python -m fpsgen.train_teacher --config configs/train_teacher.yaml
python -m fpsgen.train_student --config configs/train_student.yaml
```

Before stage 3, set `train.teacher_checkpoint` in
`configs/train_student.yaml` to the checkpoint produced by stage 2. Pass
`--checkpoint path/to/model.ckpt` to resume a stage or `--weights` to evaluate
an existing model.

The checked-in full-training defaults are BEV: 500 epochs with batch size 8;
Teacher: 5 epochs with batch size 2; Student: 10 epochs with batch size 2.
For a one-batch/one-epoch validation, use the `configs/smoke_*.yaml` templates.

The complete configurations are in `configs/`; they use the full training
sequences rather than a reduced subset.

## Inference and evaluation

Set the BEV checkpoint explicitly because it is a separate stage-1 model:

```bash
export FPSGEN_BEV_CHECKPOINT=/path/to/bev.ckpt
python -m fpsgen.inference --diff /path/to/student.ckpt --refine ""
```

For the pose-cropped completion benchmark on the SemanticKITTI test sequence:

```bash
export FPSGEN_OUTPUT_DIR=outputs/seq08_completion
PYTHONPATH="$PWD/fpsgen/models/ChamferDistancePytorch:$PYTHONPATH" \
python -m fpsgen.utils.eval_path_multirange \
  --diff /path/to/student.ckpt \
  --path /path/to/KITTI_Odometry --dataset SemanticKITTI --sequences 08 \
  --img-steps 10 --point-steps 1 --cond-weight 2 --cond-mode 100 \
  --select-mode frame --interval-frames 100 --no-dcd \
  --no-save-pcd --save-ply
```

This evaluator uses raw `velodyne` input and a pose-cropped `map_clean.npy`
reference. It is distinct from generation evaluation, which uses prepared
`input/*.npy` and fixed-cardinality `gt/*.npy` files. See
[`docs/EVALUATION.md`](docs/EVALUATION.md) for the full protocol.

The generation and trajectory-range evaluation commands support only FPSGen on
SemanticKITTI and KITTI-360: `fpsgen.utils.eval_generation` and
`fpsgen.utils.eval_path_multirange`.
No checkpoint or dataset is bundled in this repository.

For a quick real-data end-to-end check after the smoke checkpoints exist:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/smoke_inference.py \
  --input /path/to/KITTI_Odometry/00/input_/000000.npy \
  --bev /path/to/smoke_bev.ckpt --student /path/to/smoke_student.ckpt
```

See [evaluation details](docs/EVALUATION.md) for arguments, expected inputs,
and the local CUDA extension required by generation evaluation.

To generate samples for all eight LiDAR/vehicle/road condition tuples, use the
standalone test script. It writes PLY files by default and a `manifest.json`
containing seeds and inference settings:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_eight_conditions.py \
  --input /path/to/KITTI_Odometry/00/input_/000000.npy \
  --bev /path/to/bev.ckpt \
  --student /path/to/student.ckpt \
  --output outputs/eight_conditions/000000
```

Use `--output-format pcd` only when a legacy PCD consumer requires it.

## Repository layout

```text
fpsgen/                 model, data, training, inference, and metrics code
configs/                public full-training and smoke-test YAML templates
scripts/                environment checks and reproducible utility entry points
docs/                   installation, dataset, method, evaluation, and release notes
third_party/licenses/   notices for bundled third-party source
```

Checkpoints and generated outputs under `checkpoints/`, `experiments/`, and
`outputs/` are local artifacts and are intentionally not part of a source
release.
