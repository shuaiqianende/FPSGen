> Budget clarification: the user now requires100K successful frame exposures, ending at34,275 cumulative optimizer updates. Earlier100K-update statements below are historical. See[corrected budget](HS_OCCFLOW_384_STAGE1_100K_FRAME_BUDGET.md).

# Stage1 384 Occupancy: measured30-minute observation

Formal training remains **RUNNING**. Observed 1823.3s after actual successful updates began; separate interactive agent observations: 34, spanning 1985.0s. Preparation/Pilot time is excluded.

Observation source: `/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4/first30min.json`. This window covers the authorized batch4 continuation after update18550. The original scratch-run background record is preserved at `/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k/first30min.json`; it is not claimed as personal observation of that initial window. Long gaps between agent interactions are excluded rather than counted toward the continuous observation gate.

Updates observed: 18551 → 19494; current snapshot: 19590 /100,000. Attempts: 19603. This is an ongoing100K budget, not completed training.

Loss 0.052898 → 0.044435; last100 mean 0.052628, P50 0.052523, P95 0.076405. Loss uses global mean over both ranks; random frame/time variation remains.

Gradient before clipping 0.528141; cumulative clipping events 8384 (this observation window 138); cumulative AMP overflows 13, skipped attempts 13 (window skips 0), nonfinite loss incidents 0. Latest AMP scale 1024.0.

Mean compute/update time 1.896s; last100 host data preparation 0.0038s. Wall-clock speed including checkpoint/validation pauses 0.517 successful updates/s. The user-authorized global2→4 batch transition occurred at18550 updates before this observation window; architecture and batch remained fixed throughout this window.

Encoder, Flow, output head and time embedding all have measured gradients and changed state tensors. Exact module gradient statistics, parameter differences from step0, rank peak allocated/reserved memory, GPU minute snapshots, checkpointSHA and development metrics are in [measured JSON](hs_occ384_stage1_reports/formal_first30_summary.json) and [timeline](hs_occ384_stage1_reports/formal_monitor_timeline.csv).

Checkpoint readback passed with model/EMA/optimizer/scheduler/scaler/rankRNG/sampler/identity fields; rank hashes synchronized: True. Same-origin resume was tested before formal launch; the running formal process was not interrupted merely to repeat that test.

Physical GPUs2/3 only. tmux `fpsgen_hs_occ384_stage1_fm100k_gb4_gpu23`; launch commit `aab49956f3a9ccce6e7006dfedae03730585e17b`. Latest checkpoint `/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4/checkpoints/resume_latest.pt` (snapshot update 19000). Output `/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4`.

Primary validation uses online weights. EMA0.999 is retained and evaluated separately; its cold-start lag is not conflated with online performance. Fixed development thresholds0.25/0.5/0.75 are diagnostics; no seq08 tuning. Stage2 is not trained. Pilot BCE is slightly stronger than FM and no FM superiority is claimed.
