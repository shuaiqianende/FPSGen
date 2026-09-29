# BEVFlow LiDAR-only three-channel reconstruction evaluation

## Scope and non-goals

This evaluation measures **Stage-1 BEVFlow only**.  It does not instantiate a
Teacher, construct a PointFlow source, or calculate point-cloud metrics.  No
BEVFlow training objective, eight-state condition sampling, channel weight
`[1, 2, 1]`, architecture, or preprocessing code is modified.

The formal condition is fixed to `100`:

```text
LiDAR   : enabled
vehicle : disabled
road    : disabled
```

The evaluator creates `layout_cond = zeros([B,2,256,256])` directly.  It does
not call `get_layout_bev` and therefore cannot leak vehicle/road semantic GT.
The sole conditional signal is
`flow.model.get_raw_pc_bev(batch["pcd_part"])`.

## Representation and sampling protocol

GT is built exclusively by the production implementation:

```python
flow.processor.points_to_bev_target(batch["pcd_full"])
```

It has `B=[D,H,M]`, where `D` is log-normalized capped density, `H` is maximum
height, and `M` is occupancy.  Generated BEV is postprocessed with the exact
production rule: `pred_M > 0` denotes occupied; all other cells receive `-1`
in density and height.

The sampler reproduces `DiffCompletion.predict_full_bev` without initializing
PointFlow:

```text
Euler steps     : 10
CFG scale       : 2.0
unconditional   : zero LiDAR BEV + zero layout
conditional     : LiDAR PointPillar BEV + zero layout
initial state   : N(0,I)
```

All metrics mask the square BEV to cells whose centres are inside the 50 m
disk, matching downstream source reconstruction.  A checkpoint's
`hyper_parameters.data.gt_dir` is used automatically.  An explicit mismatched
`--gt-dir` fails unless `--allow-gt-mismatch` is supplied; mismatched results
are not formal results.

## Metrics

### Density

- **Mass-TV**: total-variation distance between occupancy-gated decoded point
  mass distributions over the valid disk.  This is the main density measure,
  because FPSGen later samples `P0` from the density distribution.
- log-density MAE on occupancy union and intersection;
- decoded raw-density MAE/RMSE on the intersection; and
- total predicted/GT density-mass ratio.

### Height

Height is decoded in metres without clamping generated values, so overshoot is
penalized.  Report GT-occupied MAE (end-to-end), intersection MAE/RMSE, and
bias.

### Occupancy and completion

Occupancy reports precision, recall, F1, and IoU.  Completion is restricted to
GT-occupied cells absent from the input LiDAR BEV.  Completion precision,
recall, F1, density log-MAE, and height MAE prevent an observed-LiDAR copy from
appearing as a successful completion model.

All headline metrics are compared with a deterministic **LiDAR Input BEV**
baseline, constructed by rasterizing `pcd_part` with the same processor.
Metrics are additionally reported in `[0,20)`, `[20,35)`, and `[35,50)` m
annuli.

## Commands

CPU metric tests and syntax validation:

```bash
python -m pytest -q tests/test_bev_eval_metrics.py
python -m py_compile fpsgen/utils/bev_eval_metrics.py scripts/eval_bevflow_lidar_only.py
```

B20 × one-seed smoke (use a free GPU; this is evaluation, not training):

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> \
python scripts/eval_bevflow_lidar_only.py \
  --config configs/research_v2/eval_bevflow_lidar_only.yaml \
  --dataset-root /data-12/M2024-HWZ/KITTI_Odometry \
  --manifest configs/research_v2/gate_seq08_20.txt \
  --samples-per-frame 1 --save-visuals 10 \
  --bev-ckpt <BEV_CHECKPOINT> \
  --output outputs/research_v2/bevflow_lidar_only/b20_smoke
```

Formal B100 × three-seed run:

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> \
python scripts/eval_bevflow_lidar_only.py \
  --config configs/research_v2/eval_bevflow_lidar_only.yaml \
  --dataset-root /data-12/M2024-HWZ/KITTI_Odometry \
  --bev-ckpt <BEV_CHECKPOINT> \
  --output outputs/research_v2/bevflow_lidar_only/b100_formal
```

## Outputs

Each run saves `samples.csv` (frame × stochastic seed), `per_frame.csv`
(seed-aggregated), `range_metrics.csv`, `completion_metrics.csv`,
`summary.json`, and up to ten seed-0 4×3 visualization panels.  Dataset
summary statistics aggregate seeds within each frame before aggregating across
frames; they do not incorrectly treat 300 stochastic samples as 300 scenes.

## Result status

**Implementation prepared; no BEV checkpoint evaluation has been run yet.**
The next authorized operation is B20 × one-seed smoke, followed only after
validation by B100 × three seeds.
