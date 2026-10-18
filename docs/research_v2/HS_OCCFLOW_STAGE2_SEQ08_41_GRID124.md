# Stage1/Stage2 NFE grid on seq08 skip100

User-requested interim evaluation on physicalGPU1; GPU2/3 training continues.
The evaluated checkpoint is frozen at10000 successful Stage2 updates/39520
successful frame exposures, SHA256
`cc1174fa7ab28ed54adb5cd2a654904b8217c250e0e554ce37608f4a27d90e51`.
Stage1 uses the final100K-frame EMA snapshot, SHA256
`b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`.
Stage2 uses online weights, fixed before reading test results, consistently with
the main PLY exports. It remains an Oracle-trained model evaluated with actual
predicted Stage1 parent candidates; this is not a predicted-parent fine-tune.

Frames are exactly08/000000,000100,...,004000:41 scenes, nine combinations of
Stage1 NFE1/2/4 and Stage2 NFE1/2/4. No GT parent correction, candidate truncation,
support dilation, point repetition, outlier filtering or Poisson sampling is
performed. Stage1 threshold is the existing0.5 endpoint-score proxy; Stage2
threshold iszero in the symmetric occupancy state. Neither is tuned on seq08.

## Metrics and GT

CD is `0.5*(mean Euclidean NN P->G +mean Euclidean NN G->P)` in meters.
P consists of generated fine occupancy cell centers on768×768×64; G consists
of ALL valid raw pre-Poisson local map candidate points, reconstructed with the
original pose/crop/viewpoint policy. There is no point subsampling or dense
pairwise cost. Exact CPU `scipy.spatial.cKDTree` queries use eps0 and four
workers. Stage1 missed regions participate fully in the G->P term.

IoU0.5 means a0.5m physical grid, NOT a predicted-probability threshold.
The historical41-frame protocol is preserved: origin[-50,-50,-4], grid200×200×17,
last Z cell clipped at4.4. Prediction rasterizes native fine occupied centers;
GT rasterizes real raw-map points directly. Supports use the same sphere/height
ROI intersection rules, with full missed-region FN accounting. Primary table
values are equal-weight41-frame macro means; microIoU is retained separately.

Every rebuilt raw frame is checked against the existing geometry, input SHA,
valid/filtered count, full packed fine labels and raw0.5m support cache. The
map/pose/calibration, all41 viewpoint scans and raw-generation code are hashed
against the historical raw-source manifest. Raw points are kept only for the
current frame; no new large GT dataset or all-frame point-cloud cache is saved.

## Paired sampling and execution

The LiDAR Encoder executes once per scene and its actual pyramid is reused.
Each Stage1 NFE starts from the same seeded noise. Stage2 draws one full
parent-indexed Gaussian table per frame and gathers rows by actual parent IDs;
shared parent voxels therefore receive identical source noise even when Stage1
candidate sets differ. All Stage2 NFEs start from the exact same gathered source
without resampling between Euler steps. Conditions are derived solely from
the partial XYZ, actual Stage1 scores/features and input-only observed hits.
GT is used only by metric/data-validation code.

The new entry is outside the source-hashed training directory:
`scripts/research_v2/hier_occ_inference/eval_seq08_combinations.py`.
Eight CPU checks passed for CD units/factor/full-target coverage,0.5m indexing,
clipped top/ROI boundary cells, Euler source preservation/NFE and macro/micro
aggregation/invalid-CD reporting. Actual first-scene GPU execution and raw
target identity checks passed; final results require all369 scene/combination
rows. No partial average is presented as a completed41-scene evaluation.

Output:
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/step010000_seq08_skip100_grid124`.
The journal is append-only, progress is atomically updated, and `--resume`
requires exact protocol/code/checkpoint identity. Full per-frame rows and nine
aggregates are exported as JSON/CSV at completion. Checkpoints, GT and PLYs are
not committed. This run does not select checkpoints/hyperparameters on seq08.

## Completed41-frame results

Status:COMPLETE. All369 distinct scene/combination rows are present; zero
duplicates, missing rows or invalid CDs. Macro means were independently
recomputed from individual rows, and each CD's directional terms and each
IoU's TP/FP/FN identity were checked. The evaluation loop took893.46seconds.
Actual GPU1 peak allocated/reserved memory was3.009/3.840GiB. GPU1 returned to
idle afterward; GPU2/3 training continued, observed at update10416 with its
source hash unchanged. This completed budget is evaluation, not formal training
completion or a claim that the Oracle learning gate passed.

| Stage1 NFE | Stage2 NFE | CD(m), lower better | IoU0.5m macro, higher better |
|---:|---:|---:|---:|
|1|1|0.3366|40.69%|
|1|2|0.3187|42.96%|
|1|4|0.3136|43.87%|
|2|1|0.3242|39.29%|
|2|2|0.2540|45.68%|
|2|4|0.2426|47.16%|
|4|1|0.2793|40.28%|
|4|2|0.2382|45.94%|
|4|4|0.2325|47.09%|

For this intermediate checkpoint,4+4 gives the lowest observed CD;2+4 gives
the highest macroIoU, only0.067 percentage points above4+4. Stage2's1->2 NFE
gain exceeds its2->4 gain for all three Stage1 settings. These are descriptive
results for this frozen snapshot; no statistical superiority or test-based
selection of a new training protocol is claimed.

Committed small artifacts:

- `stage2_384_audit/step010000_seq08_grid124_complete.json`: actual completion,
  identities, units, memory, integrity and all nine aggregates.
- `stage2_384_audit/step010000_seq08_grid124_summary.csv`: nine macro/micro rows.
- `stage2_384_audit/step010000_seq08_grid124_per_frame.csv`: all369 point/IoU/CD
  records, including directional CD and complete GT counts.

The full results/protocol and timing journal remain at the output path above.

## Matched Stage1-only control: confirmed IoU regression

A subsequent GPU1 control repeats all41 inputs with the identical final Stage1
EMA weights, scene_seed noise, NFE and threshold used by the nine-combination
run. It uses the same cached raw-map0.5m GT, geometry and macro averaging. All123
frame/NFE candidate counts match all369 original evaluation rows. One encoder
forward per frame;246 control rows;131.21seconds after initialization. Eight
existing metric/inference CPU tests pass. Training sources remain unchanged.

| Stage1 NFE | Stage1 only, parent centers | +Stage2 NFE1 | +Stage2 NFE2 | +Stage2 NFE4 |
|---:|---:|---:|---:|---:|
|1|46.28%|40.69%|42.96%|43.87%|
|2|48.81%|39.29%|45.68%|47.16%|
|4|47.55%|40.28%|45.94%|47.09%|

Thus Stage2 currently lowers0.5m IoU in all nine matched settings. For2+4,
the decrease is48.81%->47.16%, or1.65percentage points; for4+4 it is
47.55%->47.09%, or0.46points. The earlier historical Stage1 checkpoint24000
EMA gave48.76% at NFE2, but this matched comparison removes checkpoint/seed
confounding. No conclusion that Stage2 improves the system is supported.

A no-network control fills all eight geometrically valid children of each
selected parent before rasterization. Its0.5m IoUs are50.78%,51.40%,49.76%
for Stage1 NFE1,2,4. This records the effect of retained coarse spatial
coverage; it does not establish fine-resolution occupancy accuracy, and is
not a learned model or GT-assisted prediction. Stage2 also fails to beat this
simple coverage control at0.5m. Fine/native accuracy must be evaluated before
claiming high-resolution benefit.

The evaluated Stage2 remains Oracle-trained, with neutral parent scores, but
deployment uses predicted parents and actual Stage1 scores. This verified
protocol mismatch is a risk, not a demonstrated causal explanation for the
regression. No training loss, thresholds or candidate protocol were changed
on seq08, and GPU2/3 training was not interrupted. Predicted-parent adaptation
and error diagnosis must use training/development data. The result is an
intermediate10000-update snapshot, not a finished100K-frame experiment.

For the matched Stage1 NFE2 -> Stage2 NFE4 comparison, macro precision rises
from75.33% to86.27%, while macro recall falls from58.57% to51.22%.
The measured regression therefore accompanies loss of true covered support
despite fewer false positives; this describes the error tradeoff, without
asserting a proven architectural or training cause.

Control results and per-frame data:
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/step010000_matched_stage1_control`.
Small audited aggregates are committed in
`stage2_384_audit/step010000_matched_stage1_control.json`.
