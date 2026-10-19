# HS-OccFlow PatchScene-style voxel-guided resampling

## Scope

This change is a post-processing/evaluation option for the fixed HS-OccFlow 384+8-way model. It does not change Stage1/Stage2 parameters, checkpoints, source noise, candidate selection, NFE, or the original fine-voxel-center metric path.

The measurement grid is independent of the model's 768x768x64 fine grid. The default is `--upsampling-voxel-size 0.15625` m. Raw predicted fine-voxel centers are re-voxelized from the frozen geometry origin using `floor((p - origin) / voxel_size)`.

## Sampling rule

For each occupied resampling cell, points are sampled uniformly from its box intersection with the frozen HS-OccFlow ROI. The ROI is the clipped XYZ range `[-50,50] x [-50,50] x [-4,4.4]` and the existing open 3D sphere `||xyz|| < 50 m`. Uniform draws outside the sphere are rejected and redrawn; points are never clipped to the sphere boundary. The source cell ID is checked after float32 conversion.

The default count policy, `exact_target`, emits exactly 1,080,000 points. When target count is at least the number of viable occupied cells, every cell gets `floor(target/cells)` points and a seeded, without-replacement subset gets one extra. If the target is smaller than the occupied-cell count, a seeded uniform subset of cells is used, one point per selected cell; dropped cells and coverage are reported.

`patchscene_ceil` implements the paper-style uniform count `ceil(target/cells)` per occupied cell and therefore emits a multiple of the number of cells, possibly above the requested target.

The per-frame random seed is `upsampling_seed_base + frame`, independent from Stage1 and Stage2 sampling seeds. The resampler uses NumPy's `default_rng` and vectorized rejection sampling.

## Evaluation behavior

Without `--enable-voxel-upsampling`, the evaluator continues to score the original fine-voxel-center predictions. No upsampled prediction or metric is computed. With the flag, it retains all center-point metrics and adds a parallel set of CD, IoU@0.5/0.2/0.1 m, JSD-BEV, and JSD-3D metrics, using the existing metric implementations.

The new protocol records the toggle, target, resampling grid size, count policy, seed, ROI rule, and resampler source hash. Resume requires an exact protocol match. Optional `--save-point-samples` exports per-frame center and resampled arrays; it is off by default because a full run can create several gigabytes.

## Command

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/research_v2/hier_occ_inference/eval_seq08_skip10_joint_2x4_gpu3.py \
  --config CONFIG --checkpoint CHECKPOINT --output NEW_OUTPUT_DIR \
  --enable-voxel-upsampling
```

Optional overrides are `--target-points`, `--upsampling-voxel-size`, `--upsampling-count-policy`, and `--upsampling-seed-base`.
