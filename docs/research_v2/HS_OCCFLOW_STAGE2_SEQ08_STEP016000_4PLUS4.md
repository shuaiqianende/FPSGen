# Stage2 update16000, Stage1/Stage2 four-step evaluation

Status: WAITING_GPU0,37/41 actual frames complete. This is NOT a completed
41-frame result. No partial mean is reported as the requested full metric.

Stage2 online update16000/63520 training-frame exposures, checkpoint SHA256
`6891ea7139b428b7c17882e3d206fba5dc19a336a08ce404200dd9e269c4ec46`.
Stage1 remains the final100K-frame EMA snapshot. Actual predicted parent
support and input-only conditioning are used. Both stages use four Euler
velocity evaluations. Seeds, thresholds, raw-map target definitions and41
seq08/skip100 frames match the previous update10000 and13000 evaluations.
CD uses every valid raw-map point, not fine voxel GT centers. IoU is the
previous0.5m physical grid and includes missed support.

The evaluator now supports explicit --stage1-nfes4 --stage2-nfes4 so only
the requested combination is computed. Thirteen CPU checks pass. All37
saved rows are unique and finite, with Stage1 candidate/GT counts matching
the previous evaluation. Active training source SHA is unchanged.

A new, unrelated GPU0 process appeared after frame003600. The evaluator's
existing background-workload protection stopped at the scene boundary.
GPU2/3 training and all other jobs were preserved; no reclamation or
termination was attempted. A persistent idle-only resume task is running in
tmux fpsgen_hs_stage2_eval16000_4x4_gpu0, checking every30seconds and resuming
only when GPU0 is again safely available. Pending frames:003700,003800,
003900,004000. Resume enforces the original protocol/checkpoint/code identity.

Output directory:
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/step016000_seq08_skip100_s1nfe4_s2nfe4_gpu0`.
The evaluator writes results.json, summary.csv and per_frame.csv only after
all41 frames complete. Check the live progress.json/status and completion
files; this waiting report is a point-in-time resource record.
