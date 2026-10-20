# HS-OccFlow vs PatchScene — Seq08/skip100 trend evaluation

## Scope

This is a 41-frame trend check, using SemanticKITTI sequence 08 frames `000000, 000100, ..., 004000`. Per the user's direction, the longer PatchScene skip10 run was paused after 135/408 frames; its completed outputs are retained but are not used in this comparison. The full HS-OccFlow 2+4 skip10 evaluation completed, and its same 41 frames were selected for the paired comparison.

This is an interim reference, not the final full-sequence benchmark.

## Frozen models and protocol

- HS-OccFlow: Stage1 2 NFE + Stage2 4 NFE; Stage1 SHA256 `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`; Stage2 SHA256 `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`.
- PatchScene: upstream commit `49de6d848663956e57cdd2f95d3b364e7a7afbe6`; official SemanticKITTI `epoch=99.ckpt`, EMA weights, SHA256 `f50bf65f419c32bb40afa14e30440e1c4ea839dd92c678f8b4487e0d2b36140a`; `Ring_Spread_V2`, 15 denoising steps.
- HS native voxel size: `(0.13020833, 0.13020833, 0.13125)` m. PatchScene native voxel size: `(0.15625, 0.15625, 0.15625)` m. Jitter samples inside each model's own native occupied voxel and the common evaluation ROI; no re-voxelization is used.
- `voxel_center` preserves all native occupied centers inside the common ROI. `native_fine_jitter` uses the exact 1,080,000-point budget for every frame, with independent per-frame seed `20261019 + frame`.
- Both models use the same 18K XYZ input scans and the same raw-map GT loader, frozen ROI, and per-frame GT arrays. Input and GT array hashes match for all 41 paired frames.
- Metrics call the shared evaluator: exact GPU KD-tree symmetric mean Euclidean CD (metres), corrected physical-grid support IoU at 0.5/0.2/0.1 m, and the shared FPSGen JSD-BEV/JSD-3D implementation. IoU micro values aggregate TP/FP/FN across frames; the other reported values are macro means.

## Results

| Model | Output | Mean points | CD ↓ (m) | IoU@0.5m ↑ macro / micro | IoU@0.2m ↑ macro / micro | IoU@0.1m ↑ macro / micro | JSD-3D ↓ | JSD-BEV ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HS-OccFlow | voxel center | 613,818 | 0.20300 | 50.69% / 50.12% | 39.54% / 39.04% | 16.42% / 16.30% | 0.41618 | 0.30767 |
| HS-OccFlow | native fine jitter | 1,080,000 | 0.19798 | 51.96% / 51.64% | 40.44% / 40.19% | 19.09% / 18.70% | 0.40895 | 0.30282 |
| PatchScene | voxel center | 291,419 | 0.35702 | 39.29% / 38.21% | 32.45% / 31.60% | 9.65% / 9.44% | 0.47546 | 0.39243 |
| PatchScene | native fine jitter | 1,080,000 | 0.34360 | 42.96% / 42.14% | 35.76% / 35.17% | 18.56% / 18.14% | 0.46331 | 0.38423 |

## Trend

On this 41-frame subset, HS-OccFlow is better on CD, all three IoU thresholds, and both JSD measures for both output modes. Native jitter improves the point-budget metrics for each model relative to its native centers. PatchScene jitter substantially raises IoU@0.1m compared with its center output, while its CD and coarser IoUs remain behind HS-OccFlow. Treat these as trend observations; the subset is not the complete 408-frame evaluation.

## GPU-policy provenance correction

The first 14 PatchScene frames were initially launched on GPU1 before the GPU restriction was reiterated. That process was interrupted; its partial directory is retained as `experiments/patchscene_eval/official_seq08_skip100_native_15steps_gpu1/` and is not the final source. Those same 14 frames were regenerated on allowed physical GPU2. All 14 native occupancy-coordinate arrays were bitwise identical to the earlier arrays. The final 41-frame prediction set was then re-evaluated on GPU2; the table above uses that verified result. All ongoing compute after the correction used only GPUs2/3.

## Artifacts

- HS 408-frame result, source rows: `experiments/hs_stage2_eval/nativefine_jitter_seq08_skip10_joint2x4_gpu2/`
- PatchScene 41-frame predictions: `experiments/patchscene_eval/official_seq08_skip100_native_15steps_gpu2/`
- Verified unified metrics and paired CSV/JSON: `experiments/patchscene_eval/official_seq08_skip100_unified_gpu2_verified/`
- Recheck evidence: `experiments/patchscene_eval/official_seq08_skip100_native_15steps_gpu2/gpu2_recheck_audit.json`
- Paused PatchScene skip10 run: `experiments/patchscene_eval/official_seq08_skip10_native_15steps_gpu3_run2/` (`PAUSED_SKIP100_ONLY`, 135/408 frames retained).
- FPSGen worktree HEAD used for these evaluation scripts: `ecd4a4dfc6acc699280b1094cbb0ceec5a010820`.
