# Feature-guidance Gate 1

Status: implementation complete; real-frame gate not yet run in this sandbox.

The potentially aligned representation is the output of the final decoder,
not `y4.F` itself. Both sparse U-Nets now accept `return_features=False` by
default. With `True`, they return their normal prediction plus
`final_point_feature = y4.slice(x).F`. No parameter, default return value, or
checkpoint key changes, so released checkpoints load unchanged.

`slice(x)` is essential: it evaluates a sparse tensor at every original
`TensorField` row in field order. `scripts/diagnose_feature_correspondence.py`
first attaches unique synthetic source IDs, quantizes them with the same
unweighted-average policy, and verifies each recovered row equals the average
of the IDs in *that row's own batch/voxel*. This tests both the source-row
ordering contract and collision semantics. It then records the real Teacher
`P0` and Student `Pt` feature shapes and collision ratios for one frame.

Run Gate 1 in the FPSGen CUDA environment:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_feature_correspondence.py \
  --teacher-ckpt checkpoints/teacher_gen_stg1_bev_epoch=04.ckpt \
  --student-ckpt checkpoints/student_gen_stg2_epoch=09.ckpt \
  --frame /data-12/M2024-HWZ/KITTI_Odometry/08/gt_/000042.npy
```

The command writes `outputs/research_v2/feature_correspondence/report.json`.

## Interpretation rule

Point-wise feature loss is permitted only if the synthetic test is true, both
real feature tensors have `[B * N, C]` shape, and no row crosses batch IDs.
This establishes **source-row order preservation**, not full independent
source-feature identity. At a
voxel collision, all colliding field points necessarily receive the same
sliced sparse feature. Teacher `P0` and Student `Pt` also have different
quantization/collision patterns, so a feature term is a weak regularizer, not
an equality constraint on pure per-point representations.

For the 20-frame Gate A2, use `scripts/run_feature_gate.py` with
`configs/research_v2/gate_seq08_20.txt`. It loads each frozen model once and
stores `per_frame.jsonl` plus `summary.json`, including collision-group and
both-singleton statistics. Gate A is not decided until that GPU run finishes.

Recommended first formal loss after this gate is a detached cosine loss only
on this layer:

\[
L = L_{FM} + 0.05\left(1-\cos(F_S, \operatorname{stopgrad}(F_T))\right).
\]

Start at weights `0`, `0.01`, `0.05`, and `0.1`; do not enable intermediate
encoder/decoder layers before the one-layer smoke run passes.
