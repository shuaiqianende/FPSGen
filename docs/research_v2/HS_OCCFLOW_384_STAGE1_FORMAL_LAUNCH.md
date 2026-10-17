# Stage1 384 direct occupancy formal training

Current status: **RUNNING, global batch4**.
Latest actual snapshot update:19608 /100,000.
Original B1/rank phase was safely paused at18,550 updates; it is preserved.
The same model/EMA/optimizer/scheduler/scaler/RNG was restored into a separate
continuation directory after explicit user authorization to change batch.

Active tmux:`fpsgen_hs_occ384_stage1_fm100k_gb4_gpu23`.
Active output:`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4`.
Latest recoverable checkpoint:`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4/checkpoints/resume_latest.pt`.
PhysicalGPU2/3; B2/rank, global4, no gradient accumulation.
Rank0PID:1140376; launch commit`aab49956f3a9ccce6e7006dfedae03730585e17b`.

See[batch transition](HS_OCCFLOW_384_STAGE1_BATCH4_RESUME.md) and
[observation](HS_OCCFLOW_384_STAGE1_FIRST30MIN.md). Initial formal start remains
a true scratch initialization atstep0 with zero pretrained parameters. This
continuation is not a fresh initialization and does not reset successful steps.
The100K budget is ongoing; no completion claim is made.
