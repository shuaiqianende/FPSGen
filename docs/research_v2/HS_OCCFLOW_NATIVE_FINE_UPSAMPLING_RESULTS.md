# HS-OccFlow native fine voxel point sampling

This report covers the new native-grid output path. It does not replace the
historical 0.15625 m revoxelized-resampling run in
`HS_OCCFLOW_PATCHSCENE_UPSAMPLING_RESULTS.md`.

## Protocol

- Model: frozen Stage1 EMA + Stage2 online checkpoint, joint inference.
- Stage1 / Stage2: 2 / 4 Euler NFE.
- Sequence 08, frames `000000, 000010, ... 004070` (408 total).
- Modes: unchanged `voxel_center`; `native_fine_jitter` directly in occupied
  Stage2 fine coordinates, without revoxelizing the prediction.
- Native Stage2 voxel size: `(0.1302083333, 0.1302083333, 0.13125)` m.
- Jitter target: exactly 1,080,000 float32 XYZ points; seed `20261019 + frame`.
- Jitter is uniform within the intersection of the source native voxel and the
  frozen box / open 3D sphere ROI. Near-tangent cells use a deterministic
  hit-and-run fallback after 256 failed direct rejection rounds. Cells for
  which no valid float32 point exists are counted as dropped; no coordinates
  are clipped to ROI boundaries.
- CD: exact GPU KD-tree against all raw-map GT points. IoU and JSD use the
  existing corrected FPSGen evaluation functions.

## Engineering checks

`tests/test_native_voxel_point_sampling.py`: **6 passed**. Coverage includes
native-center equality, fixed-seed determinism, anisotropic voxel membership,
uniform-sampling moments, exact target count, deterministic voxel subsampling,
ROI boundary behavior and near-tangent fallback.

GPU3 single-frame end-to-end smoke (frame 000000) passed with the frozen model
checkpoints. The output had exactly 1,080,000 jitter points from 818,124
occupied native fine voxels. 2,527 cells (0.31%) had no valid float32 point in
their open-ROI intersection; these were recorded, not silently clipped.
Fourteen points used the rare hit-and-run fallback. Sampling took 3.60 s;
shared metric evaluation took 5.95 s.

| Frame 000000, Stage1/Stage2 2/4 | CD m ↓ | IoU .5 m ↑ | IoU .2 m ↑ | IoU .1 m ↑ | JSD-BEV ↓ | JSD-3D ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Native voxel centers | 0.25131 | 0.47660 | 0.36112 | 0.16852 | 0.32474 | 0.45315 |
| Native fine jitter, 1.08M | 0.24942 | 0.48572 | 0.37832 | 0.17657 | 0.32082 | 0.45058 |

The 8-frame consistency smoke (frames `000000` through `000070`, stride 10)
also completed successfully. This is a small smoke cohort, not a representative
408-frame result.

| First 8 seq08/skip10 frames, Stage1/Stage2 2/4 | CD m ↓ | IoU .5 m ↑ | IoU .2 m ↑ | IoU .1 m ↑ | JSD-BEV ↓ | JSD-3D ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Native voxel centers | 0.18175 | 0.50831 | 0.40571 | 0.17307 | 0.28460 | 0.39694 |
| Native fine jitter, 1.08M | 0.17987 | 0.53120 | 0.42466 | 0.18002 | 0.27836 | 0.38786 |

Each jitter frame produced exactly 1,080,000 points. Mean source occupied
voxels were 805,328; mean retained source-voxel coverage was 0.99627. Mean
sampling time was 1.64 s and mean jitter metric evaluation time was 3.80 s per
frame. These early eight frames are ordered and contiguous; do not interpret
the apparent improvement as a full-sequence estimate. The full 408-frame
HS-OccFlow evaluation is now running on physical GPU2 in tmux session
`hs_native_408_gpu2`, output
`experiments/hs_stage2_eval/nativefine_jitter_seq08_skip10_joint2x4_gpu2/`.
The matching full PatchScene official inference is running on GPU3 in
`patchscene_native_408_gpu3`; both outputs are independently resumable and
will be evaluated with the shared raw-map GT / metric code.

## Artifacts

- One-frame smoke: `experiments/hs_stage2_eval/nativefine_jitter_smoke_seq08_frame000000_2x4_diagnostic/`
- 8-frame smoke: `experiments/hs_stage2_eval/nativefine_jitter_smoke_seq08_skip10_first8_2x4/`
- 8-frame machine summary: `experiments/hs_stage2_eval/nativefine_jitter_smoke_seq08_skip10_first8_2x4/results.json`
- 408-frame all-scan input provenance audit:
  `experiments/hs_stage2_eval/nativefine_jitter_input_audit_408.json`
- 408-frame HS run progress:
  `experiments/hs_stage2_eval/nativefine_jitter_seq08_skip10_joint2x4_gpu2/progress.json`
- 408-frame PatchScene run progress:
  `experiments/patchscene_eval/official_seq08_skip10_native_15steps_gpu3_run2/progress.json`
- Failed initial boundary sampler trials were preserved in separate
  `*_retry1/` outputs; they are not used as results.
