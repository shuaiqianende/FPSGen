# DCD-only Teacher — Five-epoch comparison plan

## 1. Motivation

The current Stage-2 Teacher minimizes a square-root Chamfer objective plus a
local repulsion term. Sparse Sinkhorn refinement substantially improves its
endpoint geometry, but the refinement can still increase local point
accumulation. This experiment tests one clean replacement of the Teacher
objective: **Density-aware Chamfer Distance (DCD) only**.

The question is practical rather than a full loss attribution study:

> Can a DCD-only Teacher replace the established CD+repulsion Teacher and
> improve the final Teacher+Sinkhorn target distribution?

## 2. Existing baseline

The preserved historical branch in `fpsgen/models/gen_teacher.py` is

\[
L_{\mathrm{CD+rep}} = L_{\mathrm{sqrt\text{-}Chamfer}} + 0.5 L_{\mathrm{rep}},
\]

where `r_rep = 0.2 m`. Its config does not need a `loss.type` key: omission
continues to select `chamfer_repulsion` exactly as before.

The B100 baseline artifact is:

```text
experiments/gen_stg1_teacher_bev/lightning_logs/version_0/checkpoints/
  gen_stg1_teacher_bev_epoch=04.ckpt
```

The fixed evaluation manifest is
`configs/research_v2/gate_seq08_100.txt`. The original B100 cache is
`outputs/research_v2/sinkhorn_cache/b100/`.

Its existing aggregate summary, stored in
`outputs/research_v2/sinkhorn_diagnostic/b100_best/summary.json`, reports:

| Method | Chamfer | F-score | Coverage | NN-target coverage |
| --- | ---: | ---: | ---: | ---: |
| CD+rep Teacher | 0.13957 | 0.86005 | 0.88539 | 0.63925 |
| CD+rep Teacher + Sinkhorn | 0.06976 | 0.98696 | 0.99003 | 0.77668 |

The same artifact records near-duplicate rates after Sinkhorn of 1.151% at
1 cm, 2.507% at 2 cm, and 9.696% at 5 cm. These values are historical
reference values, not DCD results.

## 3. Proposed objective

The new config selects only

\[
\boxed{L_{\mathrm{DCD\text{-}Teacher}}=L_{\mathrm{DCD}}}.
\]

Fixed parameters:

```yaml
loss:
  type: dcd
  alpha: 1.0
  lambda: 1.0
  non_reg: false
```

There is **no CD loss, no repulsion loss, no CD+DCD mixture, and no other
auxiliary loss** in the DCD branch. `cd_p` and `cd_t` are only logged metrics
reused from the DCD Chamfer forward pass.

`fpsgen/ops/dcd.py` is training-safe: it mirrors
`fpsgen.utils.metrics.calc_dcd` while importing only the pinned CUDA Chamfer
operator. Its exponential uses the squared distances emitted by that operator:

\[
\exp(-\alpha d^2),
\]

not `exp(-alpha * sqrt(dist))`. Assignment multiplicity weights are detached,
use epsilon `1e-6`, and use `n_lambda = 1.0`.

## 4. Experimental controls

Except for `loss.type`, the DCD run matches the baseline contract:

| Item | Fixed value |
| --- | --- |
| Dataset | Existing formal Teacher GT; **not** `gt_possion` |
| Train split | 00–07, 09, 10 |
| Point count | 180,000 |
| Resolution | 0.05 m |
| Source sampler | Existing BEV-supported `P0` sampler |
| Architecture | Teacher `out_dim=96` |
| Optimizer / LR | Adam / 0.001 |
| Batch size | 2 |
| Epochs | 5 |
| Training seed | Existing `train_teacher.py` deterministic seed 42 |

Formal config: `configs/research_v2/train_teacher_dcd_a1_l1_5ep.yaml`.

## 5. Evaluation protocol

The comparison contains exactly four rows:

```text
B0  CD+rep Teacher
B1  CD+rep Teacher + Sinkhorn
D0  DCD-only Teacher
D1  DCD-only Teacher + Sinkhorn
```

Both checkpoints must cache endpoints over the fixed seq08 B100 manifest.
The evaluator verifies `torch.equal(P0_baseline, P0_dcd)` and the GT tensor
for every frame before any metrics are calculated; all 100 pairs must be
identical. Sinkhorn is fixed for B1 and D1:

```text
K=16, epsilon=0.01, iterations=100, sinkhorn_alpha=1.0
```

`dcd_alpha` and `sinkhorn_alpha` are distinct quantities and must not be
mixed.

## 6. Metrics

### Geometry

Chamfer, endpoint-to-GT NN, GT-to-endpoint NN, F-score (0.2 m threshold),
coverage, and NN-target coverage use the existing KeOps-based definitions.

### Local density and clustering

For each endpoint set, report exact duplicate rows; self-NN mean/p01/p05/p50/
p95; ratios below 1 cm, 2 cm, and 5 cm; and the eighth non-self-NN distance
mean/std/CV. Lower duplicate / close-NN ratios and lower 8NN CV indicate a
less clustered distribution.

### Assignment multiplicity

For the GT-to-pred nearest-neighbour mapping, count how many GT points select
each predicted point. Report the ratios with count equal to one and counts
at least 2/3/5, plus mean/p50/p90/p95/max. This directly diagnoses many-to-one
coverage failures.

### Vertical and range-height mass

Height bins are `[-4,-2)`, `[-2,0)`, `[0,1)`, `[1,2)`, `[2,3)`, `[3,4.4)`.
For each bin report point count, point fraction, prediction/GT ratio, and
absolute fraction error. Aggregate height-histogram L1 and TV are lower-is-
better. Also report prediction/GT mass ratios for `z > 2 m` and `z > 3 m`.

The range-height table uses horizontal range bins `0–20`, `20–35`, `35–50 m`
crossed with the same height bins. It prevents a far-range density change from
being mistaken for a purely vertical effect.

## 7. Prepared commands — do not run while GPUs are occupied

### GPU smoke (10 real steps, 180k points, batch 2)

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> \
TRAIN_DATABASE=/data-12/M2024-HWZ/KITTI_Odometry \
python fpsgen/train_teacher.py \
  --config configs/research_v2/train_teacher_dcd_a1_l1_smoke10.yaml
```

Run the command in a `tmux new -s fpsgen_dcd_teacher_5ep` session after an
explicit free-GPU check. Confirm finite DCD/backward/optimizer values, no OOM,
and record peak memory and step time. No 500-step or 5k-step gate is planned.

### Formal five-epoch training (only after the smoke passes)

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> \
TRAIN_DATABASE=/data-12/M2024-HWZ/KITTI_Odometry \
python fpsgen/train_teacher.py \
  --config configs/research_v2/train_teacher_dcd_a1_l1_5ep.yaml
```

### Cache DCD B100 endpoints after training

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> python scripts/cache_teacher_endpoints.py \
  --dataset-root /data-12/M2024-HWZ/KITTI_Odometry --sequence 08 \
  --manifest configs/research_v2/gate_seq08_100.txt \
  --teacher-ckpt <DCD_EPOCH04_CKPT> \
  --output outputs/research_v2/dcd_teacher/cache_b100 \
  --resolution 0.05 --seed 20260928
```

### Fixed four-way endpoint evaluation

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> python scripts/eval_teacher_distribution.py \
  --method cd_rep=outputs/research_v2/sinkhorn_cache/b100 \
  --method dcd=outputs/research_v2/dcd_teacher/cache_b100 \
  --sinkhorn --sinkhorn-k 16 --sinkhorn-epsilon 0.01 \
  --sinkhorn-iterations 100 --sinkhorn-alpha 1.0 \
  --output outputs/research_v2/dcd_teacher/b100_distribution
```

## 8. Results

**Pending GPU availability / five-epoch DCD-only training not yet executed.**

## 9. Discussion

**TODO after fixed B100 endpoint and Sinkhorn evaluation.**

## 10. Final decision

**TODO.** DCD becomes the preferred Teacher candidate only if DCD+Sinkhorn
improves clustering and high/vertical mass behavior without meaningful loss of
geometry, F-score, or coverage versus CD+rep+Sinkhorn.
