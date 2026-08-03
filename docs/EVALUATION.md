# Evaluation

FPSGen provides completion evaluation, generation evaluation, and eight-condition
sample generation.

## Completion evaluation

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

`--cond-mode` is a three-bit tuple in `[LiDAR, vehicle, road]` order. `1`
enables a condition and `0` disables it; `100` is LiDAR-only conditioning.
Use the same tuple for every run that will be compared.

The completion script reports Chamfer distance, completion IoU,
precision/recall/F1, and 3D/BEV JSD. `--save-ply` writes predictions under
`FPSGEN_OUTPUT_DIR`; add `--save-gt-ply` to write the matching reference.

## Generation evaluation

```bash
export FPSGEN_BEV_CHECKPOINT=/path/to/bev.ckpt

PYTHONPATH="$PWD/fpsgen/models/ChamferDistancePytorch:$PYTHONPATH" \
python -m fpsgen.utils.eval_generation \
  --diff /path/to/student.ckpt \
  --path /path/to/KITTI_Odometry \
  --dataset SemanticKITTI --sequences 08 \
  --cond-mode 100
```

Generation evaluation reports CD, EMD, DCD, and JSD. It requires the
StructuralLossesBackend extension:

```bash
pip install -e fpsgen/utils/metrics_gen/pytorch_structural_losses
```

## Eight-condition generation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate_eight_conditions.py \
  --input /path/to/KITTI_Odometry/00/input_/000000.npy \
  --bev /path/to/bev.ckpt \
  --student /path/to/student.ckpt \
  --output outputs/eight_conditions/000000
```

The condition tuples use `[LiDAR, vehicle, road]` bit order:

```text
000  001  010  011  100  101  110  111
```

The script writes PLY files and `manifest.json` by default. Use
`--output-format pcd` only for legacy consumers.
