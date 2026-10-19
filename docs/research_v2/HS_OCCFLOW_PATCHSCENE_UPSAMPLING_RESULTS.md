# HS-OccFlow seq08/skip10 voxel-guided upsampling results

Date: 2026-10-19

Status: **COMPLETE — 408/408 frames**

## Protocol

- Dataset: SemanticKITTI sequence 08, every 10th frame (frames 0–4070; 408 scenes).
- Joint inference: Stage1 **2 NFE**, Stage2 **4 NFE**.
- Stage1: EMA weights; Stage2: online weights.
- Stage2 checkpoint: 25,000 updates / 100,000 frame exposures.
- Resampling: enabled; re-voxelize predicted fine-voxel centers at **0.15625 m**, then draw uniformly within occupied resampling voxels subject to the frozen XYZ and open 3D-sphere ROI.
- Count policy: `exact_target`; each frame has exactly **1,080,000 points**.
- Per-frame resampling seed: `20261019 + frame_id`.
- CD compares against all original pre-Poisson raw-map GT candidates (no GT downsampling); exact GPU KD-tree, symmetric half-sum of directional mean Euclidean distances in meters.
- IoU uses physical support grids at 0.5 m, 0.2 m, and 0.1 m; missed GT support is counted as false negative. JSD uses the existing FPSGen histogram semantics.

The evaluation computes the original fine-voxel-center metrics and the upsampled-point metrics side by side. The upsampling postprocess does not change Stage1/Stage2 inference, checkpoints, candidate parents, or NFE.

## Results

Values are macro averages across all 408 frames. IoU micro averages are also included because they weight occupied support globally rather than weighting each frame equally.

| Metric | Fine voxel centers | 1.08M resampled points | Change |
|---|---:|---:|---:|
| CD ↓ (m) | 0.201826 | **0.199313** | −0.002513 (−1.25%) |
| IoU@0.5m ↑ macro | 0.510541 | **0.520835** | +1.029 pp |
| IoU@0.5m ↑ micro | 0.506182 | **0.518522** | +1.234 pp |
| IoU@0.2m ↑ macro | 0.402980 | **0.405787** | +0.281 pp |
| IoU@0.2m ↑ micro | 0.399608 | **0.404773** | +0.516 pp |
| IoU@0.1m ↑ macro | 0.167258 | **0.181231** | +1.397 pp |
| IoU@0.1m ↑ micro | 0.166600 | **0.178041** | +1.144 pp |
| JSD-BEV ↓ | 0.307744 | **0.303237** | −0.004506 |
| JSD-3D ↓ | 0.410065 | **0.405468** | −0.004598 |

All reported metrics moved modestly in the favorable direction. The largest IoU gain is at 0.1 m, consistent with filling the quantization gaps between fine voxel centers; this is a postprocessing effect and does not establish improved learned occupancy.

## Point counts and runtime

| Quantity | Result |
|---|---:|
| Output points per frame | exactly 1,080,000 (min = max) |
| Mean occupied 0.15625 m resampling voxels | 394,536 |
| Mean samples per occupied resampling voxel | 2.891 |
| Occupied voxel coverage | 100% |
| Dropped occupied voxels | 0 |
| Mean resampling time | 1.029 s/frame |
| Mean upsampled CD time | 0.218 s/frame |
| Total evaluation time | 2,947.6 s (49 min 8 s) |

## Provenance and artifacts

The frozen Stage1 and Stage2 checkpoint SHA256 values are recorded in the run protocol. The run used the original raw-map local candidate GT with the existing viewpoint support and the frozen 50 m sphere / `z ∈ [-4, 4.4]` evaluation region. No metric filtering was applied.

Server-side run directory (large per-frame artifacts remain local and are not committed):

```text
experiments/hs_stage2_eval/morph_step025000_seq08_skip10_joint_2x4_gpu3_batch4_upsample1080k/
```

Key files:

```text
summary.csv
per_frame.csv
per_frame.jsonl
results.json
progress.json
protocol.json
admission.json
```

`progress.json` and the top-level `results.json` report `COMPLETE`, with 408 rows. The embedded `protocol.status` field was initialized as `RUNNING` and was not finalized by the evaluator; the completion state is taken from the run progress and result row count. This metadata inconsistency does not affect the metrics, but should be corrected in the evaluator for future runs.

## Interpretation

Voxel-guided uniform resampling improves the reported point-based CD, support IoU, and histogram distances slightly while increasing each prediction to the requested point budget. It samples within occupied 0.15625 m cells around the original voxel-center support; it does not infer new surface geometry. The gains should therefore be interpreted as denser sampling and reduced center-quantization effects, not as a change to the learned Stage2 model.
