# T1: U1+A depth-moment residual experiment

Hypothesis: plain plane lift loses axial position; first and second feature moments along each collapsed axis can restore some depth-dependent information without full volumetric attention. This is an experimental hypothesis, not a claim of novelty or demonstrated quality improvement.

Four A sites retain their original paths. Additional per-plane first/second centered coordinate moments are linearly mixed in 2D and lifted with matching axial basis functions; a zero initialized projection preserves exact initial baseline output and permits immediate projection gradients. Full U1+A shared scratch initialization is unchanged. No B/S1/S2, extra loss, GT condition, or Stage2 changes.

GPU2 shared admission is explicitly authorized by the user; no other process is stopped. Require >=14.5 GiB free, real Batch2 forward/backward/checkpoint gate and 32 successful-update smoke, then fresh 100K exposure scratch training, original FM, seed20261032, AMP FP16/FP32 parameters, warmup1000 and existing scheduler. Failed admission waits; failed gate blocks. All work runs in independent tmux. Final evaluation: seq08 skip100 41 frames NFE2 plus fixed K and timing. No 900 frames.

CPU: two tests pass (initial identity, projection gradient, depth-dependent output and shared encoder initialization); config validation passes. GPU feasibility/quality NOT TESTED before gate completion. Need compare against U1+A, and if promising a parameter-matched residual control before attributing mechanism benefit.
