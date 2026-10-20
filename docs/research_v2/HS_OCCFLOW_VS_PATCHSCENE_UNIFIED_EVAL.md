# HS-OccFlow vs PatchScene: unified evaluation

## Current status

The first 8 ordered frames (`000000` through `000070`, stride 10) have been
evaluated for both methods. This is a smoke comparison only. PatchScene's full
408-frame inference is running; its final metrics are not yet available.

The paired evaluators verified byte-identical input-file SHA256 and target-GT
array SHA256 for every one of these eight frames. GT is the FPSGen raw-map
local candidate set with existing viewpoint support and frozen ROI; no
morphology or GT downsampling was used.

## Protocol

- HS-OccFlow: Stage1 2 Euler NFE + Stage2 4 Euler NFE, frozen Stage1 EMA and
  Stage2 online checkpoint.
- PatchScene: official `Ring_Spread_V2`, EMA weights, 15 denoising steps.
- HS native fine grid: 768×768×64, voxel size
  `(0.13020833, 0.13020833, 0.13125)` m.
- PatchScene native grid: 640×640×48, voxel size `(0.15625, 0.15625, 0.15625)` m,
  native Z extent `[-3.75, 3.75]` m.
- Common metric ROI: HS frozen XYZ bounds and open 3D radius-50 m sphere.
  PatchScene native centers outside this common ROI are excluded from its
  center-point metric; native jitter samples from each PatchScene voxel's
  intersection with the common ROI.
- Native-center metrics retain all eligible native centers and therefore do
  not equalize point counts.
- Jitter metrics use exactly 1,080,000 points per frame, sampled within each
  model's own native occupied voxels. Seed is `20261019 + frame`.
- CD: exact GPU KD-tree, symmetric mean Euclidean distance, all GT points.
- IoU: corrected physical-grid support IoU at 0.5/0.2/0.1 m.
- JSD: same sparse implementation of the audited FPSGen histogram metric.

## First 8 frames (macro means)

| Method | Mean points | CD ↓ m | IoU@0.5 ↑ | IoU@0.2 ↑ | IoU@0.1 ↑ | JSD-BEV ↓ | JSD-3D ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| HS-OccFlow native centers | 805,328 | 0.18175 | 0.50831 | 0.40571 | 0.17307 | 0.28460 | 0.39694 |
| HS-OccFlow native jitter | 1,080,000 | 0.17987 | 0.53120 | 0.42466 | 0.18002 | 0.27836 | 0.38786 |
| PatchScene native centers | 408,319 | 0.30585 | 0.38980 | 0.31976 | 0.09662 | 0.36864 | 0.47171 |
| PatchScene native jitter | 1,080,000 | 0.29658 | 0.42936 | 0.35878 | 0.16532 | 0.36051 | 0.45955 |

For IoU, the corresponding micro means were:

| Method | IoU@0.5 micro | IoU@0.2 micro | IoU@0.1 micro |
|---|---:|---:|---:|
| HS-OccFlow centers | 0.50851 | 0.40593 | 0.17302 |
| HS-OccFlow jitter | 0.53136 | 0.42485 | 0.17997 |
| PatchScene centers | 0.39012 | 0.32009 | 0.09653 |
| PatchScene jitter | 0.42967 | 0.35912 | 0.16521 |

The eight-frame smoke indicates that jitter improves the listed metrics for
both methods on this subset. It does not establish a sequence-level ranking;
these are the first eight frames in order, not a random or full validation
sample. The two models also use different native voxel resolutions and
different denoising-call budgets (HS 2+4 versus PatchScene 15), so this is a
quality comparison under the requested protocols, not an equal-compute claim.

## Reproduction artifacts

- HS 8-frame result:
  `experiments/hs_stage2_eval/nativefine_jitter_smoke_seq08_skip10_first8_2x4/`
- PatchScene 8-frame native predictions:
  `experiments/patchscene_eval/native_grid_smoke_seq08_skip10_first8_15steps_gpu3/`
- PatchScene 8-frame shared metrics:
  `experiments/patchscene_eval/native_grid_smoke_seq08_skip10_first8_15steps_gpu3/unified_metrics/`
- Full PatchScene inference is tracked in the separate run path documented in
  `PATCHSCENE_OFFICIAL_REPRODUCTION_AUDIT.md`.

The final 408-frame table must be generated only after both full prediction
sets are evaluated with this protocol. Do not substitute the smoke values for
that table.
