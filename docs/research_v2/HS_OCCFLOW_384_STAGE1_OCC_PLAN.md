# 384-grid Stage1 direct occupancy: implementation status and execution gates

> Historical prototype status at `03ad22d`. Current implemented runner and
> measured progress are in [GPU preflight](HS_OCCFLOW_384_STAGE1_GPU_PREFLIGHT.md),
> [data](HS_OCCFLOW_384_STAGE1_DATA.md), and [pilots](HS_OCCFLOW_384_STAGE1_PILOT.md).
> The original limitations below are retained as the prototype audit trail.

Research branch: `research/hs-occflow-384-stage1occ` (based on `research/hs-occflow`). This is a **prototype only**, not a running training job and not a validated 100K model.

## Frozen geometry proposal

Parent: `384 x 384 x 32` over XYZ `[-50,50] x [-50,50] x [-4,4.4]`, valid 50-m sphere.
Child: `2 x 2 x 2`, yielding `768 x 768 x 64`, approximately `0.130208 x 0.130208 x 0.13125 m`.
Coarse grid elements: 4,718,592 (2.25 times old 256-grid). The Stage1 dense 3D CNN memory must be measured, not inferred from element count.
Convolution/pyramid shapes for 3 downsamplings: XYZ `384x384x32 → 192x192x16 → 96x96x8 → 48x48x4`; check actual VoxDet and U-Net forward tensors, not static divisibility alone.

## Implemented in this commit

- `geometry_384_8way.yaml`: stage-1 parent and stage-2 child geometry.
- `stage1_occupancy.py`: full-resolution sparse-ID -> binary occupancy target, symmetric binary Gaussian CFM target `x_1=2o-1`, masked FM-MSE/BCE, independent Stage1 wrapper built on the existing E2E VoxDet + flow U-Net.
- Prototype FM/BCE configurations and CPU math-test source.

## Important limitations (do not report as trained)

- **No complete Stage1 train/evaluate/checkpoint entrypoint has been added**. Existing Stage1 trainer targets count-density and does not implement this new occupancy objective.
- No 384-grid forward/backward was executed on a CUDA device.
- No 384-grid full raw-map train/development dataset has been prepared; the 16-frame CPU cohort does not substitute for it.
- No matched BCE overfit, FM overfit, 1K pilot, candidate threshold calibration, GPU DDP resume test, or long training has been executed.
- This branch does not change existing Student training jobs. GPU admission must be rechecked before any run.

## Recommended next execution

1. Check old PointFlow Student GPU2/3 occupancy, disk space, no interference.
2. Prepare full pre-Poisson raw-map 384/8 sparse labels using `scripts/research_v2/hier_occ_flow/prepare_dataset.py` with versioned manifest. Verify 17,870 train and 900 development frames and ROI consistency. No seq08 for hyperparameter tuning.
3. Implement a **new** Stage1 binary-occupancy train/eval runner, using real complete data and `Stage1DirectOccupancy`. Do not call existing count-density training scripts unchanged.
4. GPU smoke on representative 384 scenes (including max): VoxDet + flow U-Net forward/backward, actual memory/time, AMP overflow/grad check and distributed resume; reject jobs when GPU2/3 busy.
5. Matched BCE/FM 8–16-frame overfit then 1K pilots. Evaluate parent IoU, precision, recall, candidate budget and downstream fine occupancy recall ceiling on full development frames.
6. Freeze geometry/threshold/split/config and only then admit 100K FM/BCE runs if model and memory gates pass. Use stage1 prediction at 384 directly as Stage2 384-parent candidate scores.
7. Archive config SHA, source hashes, development metrics and checkpoint paths.

## Methodology notes

Binary occupied targets derived from raw pre-Poisson local map candidates are *surface-hit support*, not verified free space. Empty cells are unlabeled geometry rather than definitely free; a dense binary baseline treats them as zeros by convention. Report that limitation.

FM maps empty/occupied to -1/+1 and starts from `N(0,1)`; a zero velocity predictor has expected MSE about 2. A falling FM loss alone is not a quality result. Dense all-empty dominance and the distinction between conditional uncertainty and deterministic per-frame support need explicit BCE baseline and recall analysis.

***Never*** claim a formal experiment started without actual cluster logs/checkpoint evidence.
