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

Formal config: `configs/research_v2/train_teacher_dcd_a1_l1_5ep.yaml` for
one GPU, or `train_teacher_dcd_a1_l1_5ep_ddp2.yaml` for two-process DDP.
The DDP config uses per-rank batch 1, so its global batch remains 2.

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
identical. The selected Sinkhorn configuration for B1 and D1 is:

```text
K=8, epsilon=0.002, iterations=200, sinkhorn_alpha=1.0
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

For physical GPUs 2 and 3 with the same effective global batch, use:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
TRAIN_DATABASE=/data-12/M2024-HWZ/KITTI_Odometry \
python fpsgen/train_teacher.py \
  --config configs/research_v2/train_teacher_dcd_a1_l1_smoke10_ddp2.yaml
```

### Formal five-epoch training (only after the smoke passes)

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU> \
TRAIN_DATABASE=/data-12/M2024-HWZ/KITTI_Odometry \
python fpsgen/train_teacher.py \
  --config configs/research_v2/train_teacher_dcd_a1_l1_5ep.yaml
```

Two-GPU equivalent:

```bash
CUDA_VISIBLE_DEVICES=2,3 \
TRAIN_DATABASE=/data-12/M2024-HWZ/KITTI_Odometry \
python fpsgen/train_teacher.py \
  --config configs/research_v2/train_teacher_dcd_a1_l1_5ep_ddp2.yaml
```

For the user-authorized throughput run with batch 2 **per GPU** (global batch
4), use `train_teacher_dcd_a1_l1_5ep_ddp2_bs2.yaml`. This is recorded as a
DDP throughput variant because its effective global batch differs from the
single-GPU global-batch-2 comparison config.

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
  --sinkhorn --sinkhorn-k 8 --sinkhorn-epsilon 0.002 \
  --sinkhorn-iterations 200 --sinkhorn-alpha 1.0 \
  --output outputs/research_v2/dcd_teacher/b100_distribution
```

## 8. Results

### B10 Sinkhorn parameter decision — epoch-03 checkpoints

This decision uses the fixed, non-adjacent seq08 B10 manifest:

```text
000000, 000452, 000904, 001357, 001809,
002261, 002713, 003166, 003618, 004070
```

Both methods use the same cached `P0` and `gt_possion` target for every
frame (bit-identical checks pass). The following direct comparison fixes
`K=8`, `iterations=100`, and `sinkhorn_alpha=1.0`, and sets
`epsilon=0.002`:

| Teacher + Sinkhorn | Chamfer ↓ | F-score ↑ | Coverage ↑ | NN-target coverage ↑ | Exact duplicates / frame ↓ | NN<1cm ↓ | NN<2cm ↓ | NN<5cm ↓ | Height-TV ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CD+rep | 0.053370 | 0.987064 | 0.981593 | 0.878427 | 323.5 | 1.466% | 1.874% | 3.668% | 0.033195 |
| **DCD-only** | **0.052840** | **0.990150** | **0.986229** | **0.893680** | **132.8** | **0.549%** | **0.830%** | **2.486%** | **0.017644** |

At this same geometry-favorable low epsilon, DCD improves every primary
geometry, coverage, duplicate, close-neighbour, and height-distribution
metric above. The only local-density statistic favouring CD+rep is 8NN CV
(0.2963 versus DCD 0.3023); it does not offset CD+rep's substantially higher
near-duplicate rates.

The full K=8 epsilon sweep was evaluated at
`epsilon ∈ {0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005}`. Below 0.002,
both Teacher variants show sharply rising exact duplicates and NN<1cm ratios.
Therefore `epsilon=0.002` is selected as the lowest tested geometry-improving
value before the severe collapse regime.

### B10 Sinkhorn iteration ablation — DCD epoch-03

With `K=8`, `epsilon=0.002`, and `alpha=1.0` fixed, the same ten non-adjacent
frames were re-evaluated while changing only Sinkhorn iterations. Runtime is
the isolated Sinkhorn refinement time per frame; it excludes geometry metrics.

| Iterations | Chamfer ↓ | F-score ↑ | Coverage ↑ | Exact duplicates / frame ↓ | NN<1cm ↓ | NN<2cm ↓ | NN<5cm ↓ | Runtime / frame ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 0.051255 | 0.988064 | 0.981064 | 493.6 | 2.671% | 3.406% | 6.213% | 123.5 ms |
| 50 | 0.052346 | 0.989365 | 0.984064 | 269.8 | 1.210% | 1.642% | 3.707% | 136.8 ms |
| 100 | 0.052840 | 0.990150 | 0.986229 | 132.9 | 0.549% | 0.830% | 2.486% | 164.1 ms |
| **200** | **0.052735** | **0.990956** | **0.988255** | **64.8** | **0.260%** | **0.476%** | **1.938%** | **218.1 ms** |
| 500 | 0.051936 | 0.991864 | 0.990295 | 20.6 | 0.129% | 0.315% | 1.701% | 380.8 ms |

`iterations=200` is selected as the current quality/runtime Pareto point. It
improves all reported geometry and close-neighbour measures over 100 iterations
for a 1.33x Sinkhorn-only time cost. `iterations=500` remains the
quality-priority upper bound: it is appropriate only when the extra 1.75x cost
over 200 can be amortized by offline target precomputation.

Artifacts:

```text
outputs/research_v2/dcd_teacher/b10_dcd_e03_keps_sweep/k8_eps0.002/
outputs/research_v2/dcd_teacher/b10_cdrep_keps_full_sweep/k8_eps0.002/
outputs/research_v2/dcd_teacher/visuals_cdrep_vs_dcd_k8_eps0.005/
```

The PLY visual comparison contains GT, P0, raw Teacher endpoints, and
Sinkhorn endpoints for the spatially separated frames `000000` and `002713`.

### Remaining validation

The table above is a B10 selection result from the epoch-03 DCD checkpoint.
The final five-epoch checkpoint and fixed B100 robustness evaluation remain
required before making a paper-level claim about the full training run.

## 9. Discussion

The selected next-stage target is DCD-only Teacher plus Sparse Sinkhorn with
`K=8, epsilon=0.002, iterations=200, alpha=1.0`. This isolates the method
choice from the later B100 robustness measurement.

## 10. Final decision

**PROVISIONAL PROCEED — use DCD-only Teacher with K=8 / epsilon=0.002 /
iterations=200 for the
next endpoint-target experiments.** On the controlled B10 comparison it is
better than CD+rep at the selected low-epsilon operating point while producing
substantially fewer local duplicates. Reconfirm this choice on the final
five-epoch checkpoint and fixed B100 before declaring the result final.
