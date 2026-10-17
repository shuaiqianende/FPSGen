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

Final results are populated only after all41 frames finish.
