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
