# 384 Stage1 engineering preflight

CPU suite: 85 passed,1 CUDA test skipped in that
CPU invocation. The skipped CPU/CUDA geometry parity test was separately run
on admitted physicalGPU2:1 passed. No system environment was changed.
Environment: `/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39` for full model,
`env/pwm_ptv3_pointflow` for CPU/data tools. PyTorch2.0.1+cu117.

Encoder19,056,568 parameters. FM U-Net524,073; BCE U-Net503,465. Both reuse the
same VoxDet trunk and convolution widths8/16/32/64. BCE uses a learned constant
FiLM embedding; FM uses time embedding. No density/detection/regression neck head
is optimized. All parameters start randomly; constructor checkpoint reads are
forbidden by the construction audit. EMA initially equals the online model.

The prototype imported a missing E2E module and passed unsupported uint32 NumPy
arrays to Torch2.0. Both are fixed in independent Stage1 code. Tokenizer coarse
coordinates now use configured FP64 sizes, matching hierarchy discretization;
the old FP32-size helper remains untouched. The feature version is explicitly
`coarse_f64_v1` in new configurations. CPU/CUDA indices match exactly, with
feature absolute differences<=1.2e-7 on three real scenes.

Actual GPU2FM/GPU3BCE forwards and backwards passed at384 resolution:
condition ZYX16x192x192,8x96x96,4x48x48,128 channels each;
output[1,1,32,384,384]. Peak allocation~5GiB; sample-scene backward~1.05s after
warmup. Full-population follow-up GPU2 FM and GPU3 BCE tests also passed, including
02/000925 with319,738 active parents, the complete training-set maximum.
Reports retain both cohort and full-population test scope.
Initial unscaled FP16 diagnostics underflowed; GradScaler65536 overflowed the
FM output layer initially. Scale16384 is registered; occasional recoverable
startup overflow occurred during warmup and remains in logs.

Final coordinate version completed32 successful DDP updates in33 attempts.
A true same-origin31-save/32-continuous vs restart31/32 test passes with
atol1e-5,rtol1e-3; maximum model difference1.2e-7. RNG, scaler, sampler,
optimizer steps and frame order match exactly. The earlier independent-run32-step
comparison had divergent BN buffers and remains archived asFAIL; it is not
re-labeled as a successful exact-resume result. The same-origin test isolates
checkpoint correctness from independent-run CUDA variability.

Original Student PIDs1294615/1294616 exited externally before GPU admission.
This task sent them no signals. Their last observed update was26,469; exit
reason was not established and they are not labeled complete. All GPU compute
for this work is restricted to physical2/3. CPU preprocessor interruptions
affected only this task's own verified processes and retained committed shards.
