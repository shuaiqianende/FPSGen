# FPSGen

<p align="center">
  <img src="assets/FPSGen_pipeline.png" width="100%" alt="FPSGen pipeline">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.26645">
    <img src="https://img.shields.io/badge/arXiv-2607.26645-b31b1b.svg" alt="arXiv">
  </a>
  <img src="https://img.shields.io/badge/Python-3.9-3776AB.svg" alt="Python 3.9">
  <img src="https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C.svg" alt="PyTorch 1.13.0">
  <img src="https://img.shields.io/badge/CUDA-11.7-76B900.svg" alt="CUDA 11.7">
  <img src="https://img.shields.io/badge/MinkowskiEngine-0.5.4-6F42C1.svg" alt="MinkowskiEngine 0.5.4">
  <img src="https://img.shields.io/badge/Status-Official%20Implementation-success.svg" alt="Official implementation">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.26645"><strong>📄 Paper</strong></a>
  &nbsp;•&nbsp;
  <a href="docs/INSTALL.md"><strong>🛠 Installation</strong></a>
  &nbsp;•&nbsp;
  <a href="docs/DATASET.md"><strong>🗂 Dataset</strong></a>
  &nbsp;•&nbsp;
  <a href="docs/METHOD.md"><strong>🧠 Method</strong></a>
  &nbsp;•&nbsp;
  <a href="docs/EVALUATION.md"><strong>📊 Evaluation</strong></a>
  &nbsp;•&nbsp;
  <a href="docs/REPRODUCIBILITY.md"><strong>🔁 Reproducibility</strong></a>
</p>

<p align="center">
  Official implementation of<br>
  <strong>FPSGen: Flexible Point Cloud Scene Generation with BEV-Supported Transport Flows</strong>
</p>

FPSGen is a flexible point-cloud scene generation framework that supports different combinations of LiDAR, vehicle, and road conditions through BEV-supported transport flows.

This repository provides the complete three-stage training pipeline, unified inference, evaluation utilities, dataset loaders, reproducible configuration templates, and pretrained checkpoints.

## 🚀 Highlights

- **🎛 Flexible condition generation:** supports all eight LiDAR, vehicle, and road condition tuples.
- **🗺 BEV-supported structural prior:** models normalized density, maximum height, and occupancy before point-level generation.
- **🔄 Three-stage transport learning:** consists of BEV Flow, Teacher Transport, and Student Point Flow.
- **⚡ Efficient point transport:** uses an approximate optimal-transport trajectory for direct point-cloud generation.
- **📊 Unified evaluation:** supports both scene-generation and trajectory-range completion protocols.
- **✅ Reproducible release:** includes pretrained checkpoints, full-training configurations, smoke tests, and environment-validation scripts.

> [!NOTE]
> This repository contains source code and configuration templates only. Dataset files, experiment logs, local checkpoints, generated point clouds, and machine-specific server paths are excluded by `.gitignore`.
>
> See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for the complete source-release checklist.

## 🧠 Method Overview

FPSGen decomposes large-scale point-cloud scene generation into three stages.

### 1️⃣ Flexible Condition BEV Flow Prior

A conditional velocity field generates the structural BEV prior $B=[D,H,M]$, where $D$, $H$, and $M$ represent normalized density, maximum height, and occupancy, respectively.

### 2️⃣ Teacher Transport Mapping

A Teacher model learns a source-indexed clean endpoint $\mathcal{P}_1^\dagger$ from an independently BEV-sampled source point cloud $\mathcal{P}_0$.

### 3️⃣ Approximate-OT Point Flow

A Student model learns the straight transport velocity from $\mathcal{P}_0$ to $\mathcal{P}_1^\dagger$, conditioned on the generated BEV prior and the active condition tuple.

During inference, BEV Flow and Point Flow are integrated sequentially using classifier-free guidance and forward Euler updates.

For mathematical notation, tensor conventions, and method-to-code correspondence, see [`docs/METHOD.md`](docs/METHOD.md).

## 🛠️ Installation

### Recommended Environment

| Dependency | Version |
| --- | --- |
| Python | 3.9 |
| PyTorch | 1.13.0 |
| CUDA | 11.7 |
| MinkowskiEngine | 0.5.4 |

Follow [`docs/INSTALL.md`](docs/INSTALL.md) in the following order:

1. Install PyTorch and the matching CUDA toolkit.
2. Install MinkowskiEngine.
3. Install the required Python packages.
4. Build the local CUDA extensions.
5. Run the environment and CUDA-kernel checks.

> [!IMPORTANT]
> PyTorch, CUDA, MinkowskiEngine, the local CUDA extensions, and the C++ compiler must use mutually compatible versions.

## ⚙️ Local CUDA Operators

The Chamfer distance implementation is included under:

```text
fpsgen/models/ChamferDistancePytorch/
```

It is distributed under its upstream MIT license and is required for Teacher training and point-cloud evaluation.

Install it in the active Python environment:

```bash
pip install -e fpsgen/models/ChamferDistancePytorch/chamfer3D
```

The installation exposes the following Python interface:

```python
chamfer3D.dist_chamfer_3D.chamfer_3DDist
```

Compile the operator against the same Python environment, PyTorch version, CUDA toolkit, and C++ compiler used to run FPSGen.

Rebuild the operator after changing any of these components.

### Environment Verification

Run the fail-fast environment check before training:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/check_environment.py
```

Verify the Chamfer distance forward and backward kernels:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/verify_cuda_ops.py
```

The GPU index above is only an example. Select any available physical GPU through `CUDA_VISIBLE_DEVICES`; inside the process, the selected device is addressed as `cuda:0`.

## 🗂️ Dataset Preparation

FPSGen uses SemanticKITTI/KITTI-Odometry sequences containing calibration data, poses, raw LiDAR scans, and preprocessed point-cloud arrays.

A prepared sequence should follow a structure similar to:

```text
KITTI_Odometry/
└── 00/
    ├── calib.txt
    ├── poses.txt
    ├── map_clean.npy
    ├── velodyne/
    ├── input_/
    └── gt_/
```

FPSGen training reads the preprocessed `input_/` and `gt_/` NumPy arrays directly.

### Configure the Dataset Path

Set the training dataset root without modifying the YAML files:

```bash
export TRAIN_DATABASE=/path/to/KITTI_Odometry
```

The checked-in configurations intentionally contain no machine-specific paths.

### Prepare Data from Raw SemanticKITTI

Dataset preprocessing consists of two stages.

#### Step 1: Aggregate the Sequence Map

```bash
python scripts/aggregate_semantickitti_map.py
```

This script aggregates the sequence-level point-cloud map and generates:

```text
map_clean.npy
```

#### Step 2: Generate Training Arrays

```bash
python scripts/prepare_semantickitti.py
```

This script performs pose-based cropping and writes the fixed-cardinality arrays used during training and evaluation:

```text
input_/*.npy
gt_/*.npy
```

Detailed arguments, required files, and directory conventions are documented in [`docs/DATASET.md`](docs/DATASET.md).

For a complete from-scratch validation workflow, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 🏋️ Training

FPSGen uses a three-stage training pipeline.

Run the stages in the following order:

| Stage | Component | Default Epochs | Batch Size | Configuration |
| :---: | --- | :---: | :---: | --- |
| 1️⃣ | BEV Flow | 500 | 8 | `configs/train_bev.yaml` |
| 2️⃣ | Teacher Transport | 5 | 2 | `configs/train_teacher.yaml` |
| 3️⃣ | Student Point Flow | 10 | 2 | `configs/train_student.yaml` |

### 1️⃣ Stage 1: BEV Flow

```bash
python -m fpsgen.train_bev \
  --config configs/train_bev.yaml
```

### 2️⃣ Stage 2: Teacher Transport

```bash
python -m fpsgen.train_teacher \
  --config configs/train_teacher.yaml
```

### 3️⃣ Stage 3: Student Point Flow

Before starting Stage 3, set `train.teacher_checkpoint` in `configs/train_student.yaml` to the checkpoint produced by Stage 2.

```bash
python -m fpsgen.train_student \
  --config configs/train_student.yaml
```

### Resume or Evaluate a Model

Use `--checkpoint` to resume a training stage:

```bash
python -m fpsgen.train_student \
  --config configs/train_student.yaml \
  --checkpoint /path/to/model.ckpt
```

Use `--weights` to evaluate an existing model:

```bash
python -m fpsgen.train_student \
  --config configs/train_student.yaml \
  --weights /path/to/model.ckpt
```

The complete full-training configurations are provided under `configs/` and use the complete training-sequence split.

### Smoke-Test Configurations

For a one-batch, one-epoch validation, use:

```text
configs/smoke_bev.yaml
configs/smoke_teacher.yaml
configs/smoke_student.yaml
```

> [!TIP]
> Run the smoke-test configurations before starting full-scale training. They provide a fast way to verify dataset loading, model construction, checkpoint writing, and CUDA-operator compatibility.

## 🔍 Inference

The BEV Flow checkpoint must be specified separately because it is produced during Stage 1.

```bash
export FPSGEN_BEV_CHECKPOINT=/path/to/bev.ckpt
```

Run inference with the Student Point Flow checkpoint:

```bash
python -m fpsgen.inference \
  --diff /path/to/student.ckpt \
  --refine ""
```

The BEV Flow and Student Point Flow models are integrated sequentially during inference.

## 📊 Evaluation

FPSGen provides separate evaluation protocols for point-cloud generation and pose-cropped trajectory completion.

### Pose-Cropped Completion Evaluation

The following command evaluates the SemanticKITTI test sequence using raw `velodyne` scans and a pose-cropped `map_clean.npy` reference:

```bash
export FPSGEN_OUTPUT_DIR=outputs/seq08_completion

PYTHONPATH="$PWD/fpsgen/models/ChamferDistancePytorch:$PYTHONPATH" \
python -m fpsgen.utils.eval_path_multirange \
  --diff /path/to/student.ckpt \
  --path /path/to/KITTI_Odometry \
  --dataset SemanticKITTI \
  --sequences 08 \
  --img-steps 10 \
  --point-steps 1 \
  --cond-weight 2 \
  --cond-mode 100 \
  --select-mode frame \
  --interval-frames 100 \
  --no-save-pcd \
  --save-ply
```

This evaluator uses:

- Raw `velodyne` scans as input.
- Pose-cropped `map_clean.npy` point clouds as references.
- Frame- or trajectory-based sample selection.
- Configurable BEV Flow and Point Flow integration steps.

### Generation Evaluation

Generation evaluation uses prepared fixed-cardinality arrays:

```text
input_/*.npy
gt_/*.npy
```

The supported evaluation entry points are:

```text
fpsgen.utils.eval_generation
fpsgen.utils.eval_path_multirange
```

These evaluators currently support FPSGen on:

- SemanticKITTI
- KITTI-360

No dataset or checkpoint is bundled directly in this repository.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for:

- Complete command-line arguments
- Expected input formats
- Dataset-specific conventions
- Metric definitions
- Output directory structure
- Local CUDA-extension requirements

### Smoke Inference

After producing the smoke-test checkpoints, run a real-data end-to-end check:

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/smoke_inference.py \
  --input /path/to/KITTI_Odometry/00/input_/000000.npy \
  --bev /path/to/smoke_bev.ckpt \
  --student /path/to/smoke_student.ckpt
```

## 🎛️ Eight-Condition Generation

Use the standalone generation script to produce samples for all eight LiDAR, vehicle, and road condition tuples:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_eight_conditions.py \
  --input /path/to/KITTI_Odometry/00/input_/000000.npy \
  --bev /path/to/bev.ckpt \
  --student /path/to/student.ckpt \
  --output outputs/eight_conditions/000000
```

The script writes PLY files by default and creates a `manifest.json` containing:

- Random seeds
- Active condition tuples
- Checkpoint paths
- Integration settings
- Guidance parameters
- Output file information

Use the following option only when compatibility with a legacy PCD consumer is required:

```bash
--output-format pcd
```

## 📦 Pretrained Checkpoints

Download the released checkpoints and place them under `checkpoints/` using the exact filenames shown below.

| Stage | Component | Training | Filename | Download |
| :---: | --- | :---: | --- | :---: |
| 1️⃣ | BEV Flow | 500 epochs | `bevflow_gen_img_epoch=499.ckpt` | [⬇️ Google Drive](https://drive.google.com/file/d/1HzV4bU40-WPxYNO1tnRh9FkJws32-r2w/view?usp=drive_link) |
| 2️⃣ | Teacher Transport | 5 epochs | `teacher_gen_stg1_bev_epoch=04.ckpt` | [⬇️ Google Drive](https://drive.google.com/file/d/1w6MBUeFB1xRRt1-Mano9Wrwt2EQ_--oN/view?usp=drive_link) |
| 3️⃣ | Student Point Flow | 10 epochs | `student_gen_stg2_epoch=09.ckpt` | [⬇️ Google Drive](https://drive.google.com/file/d/1SiB71t2bHnEMUkg6QSuCUrkgeHaPdPY5/view?usp=drive_link) |

Recommended checkpoint layout:

```text
checkpoints/
├── bevflow_gen_img_epoch=499.ckpt
├── teacher_gen_stg1_bev_epoch=04.ckpt
└── student_gen_stg2_epoch=09.ckpt
```

Checkpoint usage:

- Select the BEV Flow checkpoint through `FPSGEN_BEV_CHECKPOINT`.
- Pass the Student Point Flow checkpoint to `--diff` during inference and evaluation.
- Use the Teacher Transport checkpoint when reproducing Stage 3 Student training.

> [!WARNING]
> Checkpoint files are excluded by `.gitignore` and should not be committed to the repository.

## 🗃️ Repository Layout

```text
fpsgen/
├── models/                 Model definitions and local CUDA operators
├── utils/                  Evaluation, metrics, and utility modules
├── train_bev.py            Stage-1 BEV Flow training
├── train_teacher.py        Stage-2 Teacher Transport training
├── train_student.py        Stage-3 Student Point Flow training
└── inference.py            Unified FPSGen inference

configs/
├── train_bev.yaml          Full BEV Flow training configuration
├── train_teacher.yaml      Full Teacher training configuration
├── train_student.yaml      Full Student training configuration
└── smoke_*.yaml            One-batch, one-epoch validation templates

scripts/
├── check_environment.py
├── verify_cuda_ops.py
├── aggregate_semantickitti_map.py
├── prepare_semantickitti.py
├── smoke_inference.py
└── generate_eight_conditions.py

docs/
├── INSTALL.md
├── DATASET.md
├── METHOD.md
├── EVALUATION.md
├── REPRODUCIBILITY.md
└── PUBLISHING.md

third_party/
└── licenses/               Licenses and notices for bundled third-party code
```

The following directories contain local artifacts and are intentionally excluded from the source release:

```text
checkpoints/
experiments/
outputs/
```

## 📖 Documentation

| Document | Description |
| --- | --- |
| [`docs/INSTALL.md`](docs/INSTALL.md) | Environment setup and CUDA-operator installation |
| [`docs/DATASET.md`](docs/DATASET.md) | Dataset structure and preprocessing workflow |
| [`docs/METHOD.md`](docs/METHOD.md) | Mathematical notation and method-to-code correspondence |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Evaluation protocols, expected inputs, and metrics |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Complete from-scratch reproducibility procedure |
| [`docs/PUBLISHING.md`](docs/PUBLISHING.md) | Source-release and publication checklist |

## 🙏 Acknowledgements

We gratefully acknowledge the authors of [**LiDiff**](https://github.com/PRBonn/LiDiff), [**LiDPM**](https://github.com/astra-vision/LiDPM), [**ScoreLiDAR**](https://github.com/happyw1nd/ScoreLiDAR), [**Distillation-DPO**](https://github.com/happyw1nd/DistillationDPO), and [**LiFlow**](https://github.com/matteandre/LiFlow) for their valuable research contributions and publicly available repositories, which inspired and informed the development of this project.

## 📚 Citation

If you find FPSGen useful in your research or work, please consider citing our paper:

```bibtex
@misc{he2026fpsgen,
  title         = {FPSGen: Flexible Point Cloud Scene Generation with BEV-Supported Transport Flows},
  author        = {Wenzhe He and Meng Wang and Jiawei Qian and Jinfeng Xu and Ying Liu and Ruihui Li},
  year          = {2026},
  eprint        = {2607.26645},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.26645}
}
```
