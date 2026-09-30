# Stage-1 BEV Backbone Training Dynamics

Date: 2026-09-30  
Branch: `research/fpsgen-v2`  
Code base at launch: `c67cccdb94563b613bcccf3f9632c5184bb823d9`

## Purpose

This report compares optimization behavior of the three Stage-1 BEV velocity
backbones under the unchanged FPSGen linear Flow Matching objective. It is a
training-dynamics report only: no BEV generation-quality evaluation has yet
been performed for these checkpoints.

## Shared controls

- SemanticKITTI train sequences: `00–07, 09, 10`.
- GT contract: `gt_dir: gt_possion`, 180,000 points/frame.
- BEV target: direct pixel-space `[Density, Height, Occupancy]` at
  `[B, 3, 256, 256]`.
- Objective: existing linear Flow Matching velocity target with fixed channel
  weights `[1, 2, 1]`.
- Conditions: the existing eight equally sampled `[LiDAR, vehicle, road]`
  states; this was not changed.
- Learning rate: `1e-4`.
- Data workers: 4.
- Scalar source: Lightning `train/loss_mse`, emitted every 100 optimizer
  steps. Every table entry below is the mean of the scalar samples inside a
  500-step interval; it is not a single minibatch value.

## Runs and comparability

| Run | Config | Runtime | Batch | Stop state |
| --- | --- | --- | ---: | --- |
| Legacy | `train_bev_legacy_gt_possion_gpu0_bs8_5ep.yaml` | PyTorch 2.0 eager FP32 | 8 | completed 5 epochs |
| DiC-S | `train_bev_dic_s_gt_possion_gpu0_bs2.yaml` | eager FP32 | 2 | manually stopped at step 8,799 |
| PixelU-S | `train_bev_pixelu_s_gt_possion_gpu0_bs8_inductor.yaml` | FP32 + `torch.compile(..., backend="inductor")` | 8 | manually stopped at step 18,599 after plateau |

The loss values are batch-averaged MSE values, so their *scale* is directly
comparable despite different batch sizes. Batch size still changes gradient
noise and the number of examples consumed per optimizer step; therefore this
is not a strict equal-samples compute comparison. The same-step loss trajectory
is nevertheless sufficient to identify PixelU-S's sustained plateau.

## 500-step loss comparison

| Optimizer steps | Legacy eager B8 | DiC-S eager B2 | PixelU-S FP32 + Inductor B8 |
| --- | ---: | ---: | ---: |
| 0–499 | 1.6711 | 1.3200 | 1.6370 |
| 500–999 | 1.0084 | 0.3052 | 1.3980 |
| 1,000–1,499 | 0.3966 | 0.1820 | 1.3349 |
| 1,500–1,999 | 0.2352 | 0.1833 | 1.3327 |
| 2,000–2,499 | 0.2256 | 0.1612 | 1.3262 |
| 2,500–2,999 | 0.1900 | 0.1334 | 1.3193 |
| 3,000–3,499 | 0.1775 | 0.1649 | 1.3166 |
| 3,500–3,999 | 0.1480 | 0.1144 | 1.3039 |
| 4,000–4,499 | 0.1749 | 0.1212 | 1.3096 |
| 4,500–4,999 | 0.1273 | 0.1355 | 1.3015 |
| 5,000–5,499 | 0.1781 | 0.1518 | 1.2974 |
| 5,500–5,999 | 0.1541 | 0.1193 | 1.2934 |
| 6,000–6,499 | 0.1314 | 0.1179 | 1.2700 |
| 6,500–6,999 | 0.1491 | 0.1325 | 1.2817 |
| 7,000–7,499 | 0.1149 | 0.1121 | 1.2805 |
| 7,500–7,999 | 0.1196 | 0.1055 | 1.2727 |
| 8,000–8,499 | 0.1345 | 0.1657 | 1.2803 |
| 8,500–8,999 | 0.1215 | 0.1239 | 1.2721 |
| 9,000–9,499 | 0.1433 | — | 1.2680 |
| 9,500–9,999 | 0.1161 | — | 1.2719 |
| 10,000–10,499 | 0.1165 | — | 1.2578 |
| 10,500–10,999 | 0.1462 | — | 1.2724 |
| 11,000–11,499 | 0.1253 | — | 1.2597 |
| 11,500–end | 0.1447 | — | 1.2712 |

The final Legacy scalar written at step 11,899 was `0.101357`. Lightning
logged once per 100 steps, while the completed five-epoch run contained 11,960
optimizer steps, so the last row has no scalar at the literal final step.

## Training outcomes

### Legacy

- Completed five epochs; checkpoints `epoch=00` through `epoch=04` exist.
- Final checkpoint (machine-local, deliberately not committed):
  `experiments/bev_legacy_gt_possion_gpu0_bs8_5ep/lightning_logs/version_1/checkpoints/bev_legacy_gt_possion_gpu0_bs8_5ep_epoch=04.ckpt`
- After the initial descent it remained in approximately the `0.11–0.18`
  range, with the final logged point at `0.101357`.

### DiC-S

- Reduced the objective fastest in the first 1k steps and stayed mostly in
  the `0.10–0.16` range through the available short run.
- It is the most promising new backbone by training loss, but it still needs
  a completed run and fixed LiDAR-only BEV quality evaluation before any
  quality claim.

### PixelU-S

- Fell from `1.6370` in steps 0–499 to only `1.3349` in steps 1,000–1,499,
  then remained approximately `1.26–1.33` through 18.6k steps.
- Its large compile-speed benefit does not offset this optimization failure.
- Status: **HOLD**. Do not launch a formal PixelU-S training run without a
  separate initialization, condition-gating, and eager-versus-Inductor
  ablation.

## Decision and next action

1. Keep PixelU-S on hold.
2. Evaluate the completed Legacy `epoch=04` checkpoint using the fixed
   LiDAR-only Stage-1 BEV protocol before interpreting its low training loss
   as generation quality.
3. Resume/complete DiC-S only after selecting the intended precision/runtime
   setting; then evaluate Legacy and DiC-S using the same B20/B100 manifest,
   same noise seeds, condition `100`, Euler 10 steps, and CFG 2.0.

No checkpoint, TensorBoard event file, Inductor cache, point cloud, or dataset
artifact is included in this commit.
