# HS-OccFlow seq08 skip10: Stage1 2-step + Stage2 4-step

## Result

Evaluation completed on 408 frames (`08/input_[0000..4070]`, stride 10) using joint Stage1 + Stage2 inference on physical GPU3. Stage1 uses its EMA weights; Stage2 uses the online weights from the completed 100k-frame-exposure checkpoint.

| Metric | Result |
|---|---:|
| CD, macro mean (m), lower is better | 0.2018256 |
| IoU @ 0.5 m, macro / micro | 0.510541 / 0.506182 |
| IoU @ 0.2 m, macro / micro | 0.402980 / 0.399608 |
| IoU @ 0.1 m, macro / micro | 0.167258 / 0.166600 |
| JSD-BEV, macro mean | 0.307744 |
| JSD-3D, macro mean | 0.410065 |
| Mean predicted occupied fine voxels / output points | 607,422 |
| Mean raw-map GT points after the frozen ROI filter | 1,303,456 |

CD is the exact GPU KD-tree symmetric Chamfer distance: half the sum of the two directional mean Euclidean nearest-neighbor distances, evaluated against all valid raw-map GT points. IoU uses occupied support on physical grids at the indicated cell size; missed GT support counts as false negatives. JSD follows FPSGen's `compute_hist_metrics_multirange` semantics and reports Jensen-Shannon distance for BEV and 3D histograms.

## Morphology and data protocol

No morphological operation was applied during evaluation to either prediction or GT (`metric_filter=NONE`). GT is extracted from the original pre-Poisson local map candidates and filtered only by the frozen ROI/sphere/height geometry. The Stage2 model was trained with a `morph-close26` target, so the reported metrics compare that trained model against unclosed raw-map GT.

Inference uses Stage1 threshold 0.5 to select predicted parents, Stage2 threshold 0.0 for child occupancy, and the full parent-indexed Gaussian source. Ragged Stage2 batching uses four complete scenes, with no token padding or truncation; FP16 batch results are approximate to single-scene execution.

## Provenance

- Stage1 checkpoint SHA256: `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2` (EMA).
- Stage2 checkpoint SHA256: `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe` (25,000 updates; 100,000 frame exposures; online weights).
- Geometry SHA256: `c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b`.
- Stage2 config SHA256: `53ed033318615b8abb502f14d7ae80cb3f5bec338342319958e87b4040828069`.
- Full run time recorded by the evaluator: 1,550.95 seconds.

## Artifacts

- Evaluator: `scripts/research_v2/hier_occ_inference/eval_seq08_skip10_joint_2x4_gpu3.py`
- Aggregate JSON: `experiments/hs_stage2_eval/morph_step025000_seq08_skip10_joint_2x4_gpu3_batch4/results.json`
- Aggregate CSV: `experiments/hs_stage2_eval/morph_step025000_seq08_skip10_joint_2x4_gpu3_batch4/summary.csv`
- Per-frame CSV: `experiments/hs_stage2_eval/morph_step025000_seq08_skip10_joint_2x4_gpu3_batch4/per_frame.csv`

The source point output is one point at each predicted fine occupied voxel center, so it is not constrained to a fixed point count.
