# HS-OccFlow PatchScene-style resampling: implementation and validation status

Date: 2026-10-19

## Implementation

- Added `voxel_guided_resampling.py` with independent re-voxelization at 0.15625 m, deterministic exact-target and PatchScene ceil count policies, vectorized uniform draws, and ROI rejection sampling.
- Added `--enable-voxel-upsampling` (off by default), target count (1,080,000), grid size, count policy, seed base, and optional point-sample export to the existing seq08/skip10 joint 2x4 evaluator.
- The original center-point metrics retain their existing field names and computation. Upsampled metrics use separate `_upsampled` fields; both are written to the same new run's JSONL/CSV/summary outputs.
- The evaluator protocol fingerprints the resampler and rejects resume under a changed sampling protocol.
- Existing historical result directory was read only.

## CPU validation

Passed:

- `tests/test_voxel_guided_resampling.py`: 7 passed, including exactly 1,080,000 points, deterministic seeds, both count policies, voxel membership, open spherical ROI, dropped-cell reporting, empty/invalid input, and a uniformity mean check.
- `tests/test_hier_occ_inference.py tests/test_hier_occ_inference_batch.py tests/test_voxel_guided_resampling.py`: 19 passed.
- Python bytecode compilation for the modified evaluator and sampler.
- Evaluator `--help` exposes all requested options.

## GPU validation and full evaluation

Not run yet. At the first resource check, physical GPU3 had an active external Python process using about 7.45 GiB, with 16.6 GiB free. A later check still found a foreign compute process using about 6.6 GiB, utilization 77%, and 17.3 GiB free. The evaluator requires at least 18 GiB free at startup and rejects external GPU compute processes. No process was interrupted and no GPU smoke or 408-frame upsampled evaluation was started.

Therefore there are no new CD/IoU/JSD metrics, no GPU runtime/peak-memory result, and no new full-evaluation output directory yet. The historical center-point baseline remains:

| Metric | 408-frame result |
|---|---:|
| CD macro | 0.2018256 m |
| IoU@0.5 m macro | 0.510541 |
| IoU@0.2 m macro | 0.402980 |
| IoU@0.1 m macro | 0.167258 |
| JSD-BEV macro | 0.307744 |
| JSD-3D macro | 0.410065 |
| Mean predicted fine voxels | 607,422 |

This baseline was read from the existing `morph_step025000_seq08_skip10_joint_2x4_gpu3_batch4/summary.csv` and was not overwritten.

## Next GPU gates

1. Wait for GPU3 to have no foreign compute processes and at least 18 GiB free.
2. Run a fresh one-frame 2x4 smoke into a new output directory; verify exactly 1,080,000 resampled points and compare both metric sets.
3. Run an 8-16 frame resume test with fixed frame-derived seeds.
4. Only after those pass, run all 408 seq08/skip10 frames into a separate output directory.
