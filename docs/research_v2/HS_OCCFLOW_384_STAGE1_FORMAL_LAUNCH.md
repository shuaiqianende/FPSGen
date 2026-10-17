# Stage1 384 occupancy:100K training-frame budget

**RUNNING**, global batch4 on physicalGPU2/3. The user clarified that100K
means successful training-frame exposures. Final optimizer update:**34,275**.
Actual snapshot:20080 updates,43220/100,000frames.

Original scratch phase0→18,550 used global2, contributing37,100frames.
All later successful updates use global4. Full20K checkpoint has42,900frames.
New phase resumes its own20K checkpoint, preserving online/EMA/optimizer/scaler/
scheduler and sampler/RNG. No external pretrained weights or pilot state loaded.
Learning rate continues smoothly from the current value to zero at the new end.

Active tmux:`fpsgen_hs_occ384_stage1_100kframes_gb4_gpu23`, training window1.
Output:`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4`.
Latest own-phase checkpoint:`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4/checkpoints/resume_latest.pt`.
Rank0PID:1396126; training launch commit`6f6361f2c1f2100dac6428973801d7a44101dea9`.

See[exact accounting and runtime proof](HS_OCCFLOW_384_STAGE1_100K_FRAME_BUDGET.md).
The earlier100K-optimizer-update plan is superseded. Historical raw checkpoints,
20K900-frame evaluation, batch-switch audit, and completed30-minute observation
remain preserved. Stage2 is not trained. System environment is unchanged.
