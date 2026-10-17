# Global batch4 continuation

User requested true global batch4 and continuation from the latest own checkpoint.
Original formal run safely paused at18,550 successful updates /18,563 attempts.
Both ranks saved synchronized online/EMA state and exited via SIGTERM at the
optimizer boundary. No unrelated process was signaled. Original output remains
`experiments/hier_occ_flow/stage1_occ384_fm_100k`.

Pinned source: `checkpoints/step018550_batch2_transition.pt` in that output,
hardlinked to its safely committed latest checkpoint. SHA256:
`1411b9cd3d1be81c1fad98ec0ca1094ad75b8ccb9f10a0a1bbb6fbcc55a710fe`.

New configuration: `stage1_occ384_fm_gb4.yaml`. Two physicalGPUs2/3, two ragged
LiDAR scenes per rank in one actual dense encoder/U-Net forward; no gradient
accumulation. Each scene has independent Gaussian noise and time. ROI reduction
normalizes across all valid cells in both scenes. Existing weights, optimizer,
scheduler, scaler, RNG and EMA are restored; encoder1e-4/flow2e-4 nominal learning
rates and100K successful-update schedule remain unchanged.

The new explicit batch-transition validator accepts only this own synchronized
scratch occupancy checkpoint, with matching sourceconfigSHA/data/geometry/model/
world size. Only batch_per_gpu and engineering-gate path may change. It rejects
changes in learning rate, target, architecture or data. The historical global
sample cursor is18,563*2=37,126 and continues from there, without applying the
new batch to all past attempts. Successful frame exposures resume at37,100.
Remaining81,450 updates use4 scenes/update; total expected successful exposures
are362,900. This is still100K optimizer updates, not100K frame exposures.

CPU tests:50 passed,1 CUDA-only parity test skipped in that CPU invocation;
actual CUDA parity is separately measured by both full-scene GPU audits.
B2 DDP:32 successful updates,33 attempts, one recoverable startup overflow.
Same-origin31→32 restart passed atatol1e-5/rtol1e-3: maximum model difference
8.2e-8; RNG/scaler/cursor/frame order match exactly. BothGPU2/3 batch2 forwards
and backwards passed the full training-set maximum319,738-parent scene and
P50/P90/P99 scenes, output[2,1,32,384,384], peakallocation~9.82GiB (no EMA in
that isolated benchmark). DDP optimizer/EMA smoke peaked~10.13GiB allocated,
~11.59GiB reserved. Steady smoke step~1.89s gives~2.12frames/s versusB1~1.87.
These are measured smoke rates, not a guarantee about later formal throughput.

Formal continuation target output:
`experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4`.
Launcher: `scripts/research_v2/hier_occ_flow/resume_stage1_gb4.sh`.
Tmux: `fpsgen_hs_occ384_stage1_fm100k_gb4_gpu23`.
Actual resumed-running evidence will be appended after successful updates.

## Actual resumed-running evidence

Formal continuation RUNNING at snapshot update18605;
starting update18,550; actual global_batch4, output[2,1,32,384,384].
Launch commit `aab49956f3a9ccce6e7006dfedae03730585e17b`; rank0PID1140376.
All runtime restoration checks passed: original online/EMA hashes, unchanged
LR/scheduler/scaler, optimizer step18,551 at first new update, and exact next
four frame IDs from preserved global cursor. See runtime auditJSON.
Original source and old experiment remain preserved. No pilot weight was used.

## Completed observation

Actual post-transition monitor:1,823.3seconds; initial/source preparation and
Pilot time excluded. Separate active-agent checks span more than30minutes, with
maximum gap under93seconds; long earlier interaction gaps are excluded.
Observed updates18,551→19,494. Window skips0, nonfinite loss0, clipping138;
cumulative skips13 were inherited from the original phase. Mean training step
1.896s; GPU2/3 peakallocated10.126/10.125GiB andreserved11.729/11.723GiB.
19K checkpoint readback passed, model/EMA rank hashes identical, optimizer steps
19,000, global data cursor38,926, successful frame exposures38,900.
Latest running snapshot:19578 /100,000. Training is left running.

19K online dev12,1NFE,score>0.5:IoU0.454275,F1=0.620402,recall0.506963.
Separate EMA dev12:IoU0.484319,F1=0.648658,recall0.553774.
These are native384-grid diagnostics, not0.5m evaluation and not dev900.
Full900-frame15K online1NFE:F1=0.623493,IoU0.458741,recall0.531449.
The registered0.25 threshold gives recall0.689149 andfine recall ceiling0.75089
with161,171 mean parents at15K, versus92,547 at0.5. No deployment threshold is
selected on seq08. Stage2 high-recall requirements remain under investigation.
