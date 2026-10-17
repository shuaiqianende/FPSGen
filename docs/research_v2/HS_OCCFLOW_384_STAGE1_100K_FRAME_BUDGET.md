# Corrected budget:100,000 successful training-frame exposures

The user clarified the intended budget is100K successful frame exposures,
not100K optimizer updates. Prior100K-update plans are superseded; their code,
checkpoints, logs and measured results remain preserved.

| Phase | Successful updates | Global batch | Successful frame exposures |
|---|---:|---:|---:|
| Original phase |0→18,550|2|37,100|
| Batch4 phase before revision |18,550→20,000|4|5,800|
| Remaining continuation |20,000→34,275|4|57,100|
| Total |34,275|mixed|100,000|

At source checkpoint20,000, actual saved successful_frame_exposures=42,900.
100,000−42,900=57,100;57,100/4=14,275 remaining successful updates.
Pilot, validation and overflow/skipped optimizer attempts are not counted.
The13 historical overflow attempts consumed26 extra training frame attempts,
which remain separately represented by the global sampler cursor42,926.
This is approximately5.60 equivalent traversals of17,870 unique training frames.

New config:`configs/research_v2/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4.yaml`.
New output:`experiments/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4`.
New tmux:`fpsgen_hs_occ384_stage1_100kframes_gb4_gpu23`.
Launcher:`scripts/research_v2/hier_occ_flow/resume_stage1_100k_frames.sh`.
Only physicalGPU2/3; trueB2/rank; pure binary occupancy FM and architecture,
geometry, raw-map target and frozen splits remain unchanged.

The source is this experiment's immutable20K checkpoint. Its own model, EMA,
AdamW moments, successful steps, sampler cursor, per-rank RNG, scheduler and
AMP scaler are restored. No pilot or external pretrained parameter is loaded.
The checkpoint/config/data/geometry/world-size and fileSHA are validated;
only the user-authorized budget and its scheduler registration may differ.

Learning rates continue without a jump from encoder9.190440524459204e-5 and
flow1.8380881048918408e-4. Warmup is not repeated; a new cosine continuation
starts at the existing multiplier0.9190440524459204 and reaches zero at34,275.
This shortens the remaining schedule consistently with the corrected budget.

CPU tests:34 passed. Mixed-batch accounting, AMP skipped attempts, exact last
optimizer step99,996→100,000, checkpoint output, LR continuity/monotonicity,
and rejection of unrelated source/config changes are tested. Existing actual
batch2 GPU/DDP/resume and maximum-scene gates are retained: architecture and
batch have not changed. Formal stop update34,275 is checked again against
actual counters after restore. The final checkpoint and full900-frame occupancy evaluation run at that stop even though it is not a
multiple of1,000 or5,000.

Registration initially waited for the20K development900 validation and safe
source-rank exit. The actual resumed-running evidence is recorded below.

## Actual continuation:RUNNING

The source completed its full900-frame20K evaluation and exited withPAUSED
at20,000 successful updates. A conservative waiting helper briefly rejected
an empty cmdline from an exiting process; no competing GPU task was started.
After verifying both source PIDs had exited andGPU2/3 were free, continuation
was launched in the new tmux's active training window1. The diagnostic remains
in console.log; no checkpoint or result was removed.

Actual runtime checks all passed: exact source online/EMA hashes, optimizer
step20,001 after first new update, scaler/scheduler restored, unchanged LR at
the boundary, next four frameIDs and sampler cursor correct, runtime LR matches
the new continuous cosine, and actual stop metadata is34,275 /100,000frames.
At least50 successful resumed updates were observed; update20080,
frames43220/100,000 at this snapshot, no newAMPskip.
The latest full module audit at20050 has finite nonzero
gradients and measured parameter changes in encoder/flow/head/time embedding/EMA.
Source launch commit:`6f6361f2c1f2100dac6428973801d7a44101dea9`. Rank0PID:1396126.
Training remains running; this does not claim the100K frame budget is finished.

20K full900 development, online1NFE,threshold0.5,native384grid:IoU47.9201%,
F1=64.2340%,recall55.6807%. These metrics do not enter theFM loss.
