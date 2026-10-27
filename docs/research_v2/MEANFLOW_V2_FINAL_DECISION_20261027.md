# MeanFlow-V2 final/stop decision2026-10-27

Traditional FM is the selected mainline. No more MeanFlow training or automatic resume is authorized.

A completed naturally before the user-stop action:10000 successful updates/20000 exposures/2AMP skips; full41-frame NFE1/2/4/8 evaluation, interval analysis and standaloneStage1 timing completed. NFE1 Parent IoU38.3500%, Recall42.7359%, Ceiling51.6572%, CD0.326994m. OriginalFM20K NFE1 IoU41.6001%, Recall47.8583%, Ceiling56.5898%, CD0.339895m. A improves originalMF20K IoU37.1116% but still does not meet the FM support/recall requirement; its lowerCD is retained rather than calling every metric worse. No long-training GO.

B was user-stopped cleanly at3332 successful updates/6664 exposures/0AMP skips. The CPU waiter/pipeline was retired before the trainer received its existing SIGTERM handler. Full model/FP32 masters/Adam/EMA/scheduler/scaler/source and intervalRNG/counters were saved atomically and verified onCPU. SHA is in `meanflow_v2_final_20261027/b_user_stop.json`. No kill-9, no deletion of history, no numerical failure, no automatic followup. B quality is NOT TESTED; it is not a completed20K negative-quality result.

An initial stop-state read found A had already completed; it sent no signal. Only the liveB training and its own CPU orchestration were stopped after a freshPID/owner/command/CUDA check. GPU2/3 subsequently had no compute processes (about20MiB display use each); other users'GPU0/1 jobs were untouched.

Final A results, interval statistics, paired comparisons and timing are preserved under `meanflow_v2_final_20261027/`. This supersedes the earlier A15K interim status without rewriting that snapshot. A/B code/worktrees, Checkpoints and logs remain local and intact. No extra seed,100K,Stage2 training or newGPU evaluation was launched for this report.
