# Seq08 408-frame evaluation and U1 continuation

Requested order on physical GPU2:

1. Evaluate Stage1 U1-Adapted (20K checkpoint), U1+A (100K), and
   U1+A+depth moment (100K) using Parent-centre output and the original
   raw-map points. The checkpoint budgets are deliberately not described as
   matched. `desescene_stage1_408_gpu2` is waiting for the existing GPU2
   training/evaluation queue; it performs a one-frame smoke before each run.
2. Evaluate completed Stage1+Stage2 models on the same seq08/skip10 frames,
   with exactly 1,080,000 output points and CD, physical IoUs and JSD.
   The four existing Stage2 variants are now registered serially in tmux
   `desescene_joint_408_gpu2`, after the Stage1 queue. The evaluator uses the
   historical 0.15625m occupied-cell resampling, but has not yet passed its
   real-GPU one-frame smoke. Failure stops the queue instead of reporting a
   result.
3. After all requested evaluations complete, start U1-Adapted scratch training
   for 100,000 successful frame exposures, batch 2, seed 20261032. Network
   weights and feature computation should use explicit FP16 with FP32 master
   weights, optimizer states and EMA; geometry and numerical control retain
   necessary precision. Use Inductor for a specifically verified submodule,
   and record the exact compile scope and any failure. A 32-update smoke and
   checkpoint-resume check precede a fresh formal run.
   Registered in tmux `desescene_u1_explicit_100k_gpu2` after successful joint
   evaluation. A new runner uses the full 19,130-frame index and its unchanged
   original surface-hit Parent IDs; the stored morphology child masks are not
   used by Stage1. It creates new FP16 weights and FP32 master/EMA from scratch.
   Its first down block is the only Inductor target. GPU gate, 32-update smoke
   and 16+16 checkpoint resume are mandatory and have not run yet.

GPU3 next training is a separate scratch ablation combining child-position
encoding with late LiDAR residual injection. Its tmux
`desescene_stage2_position_late_gpu3` waits for the existing 100K GPU3 training
and an idle GPU, then runs the real largest-four-scene batch-4 gate, independent
32-update smoke, and 100K formal run. It does not automatically evaluate on
GPU3. `desescene_gpu3_eval_handoff` monitors the pre-existing GPU3 training;
near its successful completion it will hold only the old scheduler, never its
training child, then retire that scheduler after the 100K checkpoint is safely
complete. Its obsolete 41-frame GPU3 evaluation is replaced by the registered
GPU2 408-frame late-encoder evaluation. Handoff is currently waiting, and must
not be claimed complete before its status records success.

The Stage1/joint evaluation's Python syntax and seven CPU protocol/precision
tests were checked. GPU execution, numerical parity and final metrics have not
yet been established. No 900-frame
evaluation is authorized. Stage1 U1 20K and U1+A/depth-moment 100K results
must not be represented as equal-training-budget architecture comparisons.

On 2026-11-05, the 19,130 full-training entries were checked against the raw
train, former development and guard-frame indexes: all 19,130 Parent ID slices
match the same filesystem inode, offset and length as their raw references.
The three Stage1 EMA checkpoints and the four Stage2 checkpoint structures
also loaded strictly on CPU; the two running Stage2 checkpoints were still at
86K exposures at that check. All 408 expected seq08 input files were present.
These checks do not replace the planned GPU smoke.

The new scratch Flow initialization was regenerated on CPU with seed
`20261032 + 104729`; its SHA256 is
`6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd`, matching
the earlier U1+A 100K initialization audit. The new runner checks this hash.

For continuity with prior measurements, the Stage1-only ablation uses Stage1
source seed 20261031; the joint evaluation uses the Stage2 study's registered
Stage1 seed 20261018. Each comparison within a cohort remains paired, but a
frame's Stage1-only and joint supports are not asserted to be identical.
