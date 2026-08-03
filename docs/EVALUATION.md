# Evaluation

FPSGen retains two evaluation entry points.

## Multi-range trajectory evaluation

```bash
export FPSGEN_BEV_CHECKPOINT=/path/to/bev.ckpt
export FPSGEN_OUTPUT_DIR=outputs/seq08_completion
PYTHONPATH="$PWD/fpsgen/models/ChamferDistancePytorch:$PYTHONPATH" \
python -m fpsgen.utils.eval_path_multirange \
  --diff /path/to/student.ckpt \
  --path /path/to/KITTI_Odometry \
  --dataset SemanticKITTI --sequences 08 \
  --img-steps 10 --point-steps 1 --cond-weight 2 --cond-mode 100 \
  --select-mode frame --interval-frames 100 \
  --no-save-pcd --save-ply
```

This is the completion evaluator: it compares each generated frame with its
pose-cropped dense map reference and reports per-frame aggregated Chamfer,
completion IoU, precision/recall/F1, and 3D/BEV JSD. DCD is not part of this
completion entry point. For SemanticKITTI test evaluation, use `--sequences 08`;
for a quick sparse trajectory use `--select-mode frame --interval-frames 100`.
For SemanticKITTI, the reference is reconstructed from `map_clean.npy` using
the frame pose, a 50 m radial limit, the vertical limits, and the original
10 m viewpoint voxel filter. It is not the fixed-cardinality `gt/*.npy`
target used by generation evaluation. The trajectory path also starts from
the raw `velodyne/*.bin` scan before FPSGen's internal point reduction;
generation evaluation uses the prepared `input/*.npy` condition.
The reference inference uses CFG guidance scale `2`; pass another
`--cond-weight` only for an explicit ablation.
The completion entry point leaves EMD disabled (`dist_bins=[]`) for the dense
map-crop protocol. Enable an EMD distance range separately only if you extend
the evaluator for that experiment.
Use `--save-ply` to write the generated point cloud for every selected frame
under `FPSGEN_OUTPUT_DIR`; add `--save-gt-ply` when the dense pose-cropped
trajectory reference is also needed for visualization. The public validation
commands intentionally save PLY only; the legacy `--save-pcd` switch remains
available for backward-compatible local experiments but is not needed.

## Generation evaluation

```bash
export FPSGEN_BEV_CHECKPOINT=/path/to/bev.ckpt
PYTHONPATH="$PWD/fpsgen/models/ChamferDistancePytorch:$PYTHONPATH" \
python -m fpsgen.utils.eval_generation \
  --diff /path/to/student.ckpt \
  --path /path/to/KITTI_Odometry \
  --dataset SemanticKITTI --sequences 08
```

This computes distribution metrics such as CD, EMD, DCD, and JSD. It requires
the StructuralLossesBackend CUDA extension in
fpsgen/utils/metrics_gen/pytorch_structural_losses/. Build/install that local
extension for the active Python, PyTorch, CUDA, and compiler before running:

```bash
pip install -e fpsgen/utils/metrics_gen/pytorch_structural_losses
```

Without this extension, the generation evaluator stops at import time with
ModuleNotFoundError: StructuralLossesBackend. This is the remaining known
runtime prerequisite of the clean extraction.
For SemanticKITTI, generation references are read from the prepared
`gt/*.npy` files (180,000 points per frame; `gt_pcd/*.pcd` is a compatibility
export).

## Eight-condition sample generation

`scripts/generate_eight_conditions.py` runs Unified Inference for the eight
condition tuples in `[LiDAR, vehicle, road]` bit order:

```text
000  001  010  011  100  101  110  111
```

Each tuple is saved as `condition_<bits>_sample_00.ply` by default; the output
directory also receives `manifest.json`. Multiple samples per tuple can be
requested with `--samples-per-condition N`, and `--bev-steps`/`--point-steps`
control the two forward-Euler integrations. Pass `--output-format pcd` only for
legacy consumers.
