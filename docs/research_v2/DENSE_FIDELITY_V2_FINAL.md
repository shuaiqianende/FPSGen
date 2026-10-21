# HS-OccFlow Dense Fidelity V2 — Current Status

## Result

The V2 branch is created from the requested OMD V1 commit and contains the
failure audit, three isolated dense Stage0 candidates, registered 20K
configs, and CPU tests. The study is **PARTIAL / GPU-BLOCKED**. No candidate has
been trained or benchmarked on GPU, so no new speed or quality claim is made.

## R1/R2 evidence

The 900-frame saved curves show that R1 Bottleneck-64 loses ranking quality:
at the same K=84,963 its IoU is 0.28637 versus R0's 0.37135, and precision is
0.58024 versus 0.70837. At threshold 0.5, R1 predicts about 25K more parents
per frame while macro precision falls by about 19 percentage points. R2 is
weaker still (fixed-K IoU 0.22953).

The static R2 EMA audit finds `context_scale` magnitudes no larger than 0.00560
and gate-bias sigmoid values between 0.475 and 0.525. This rules out only a
large scalar gain or strong bias-only axis preference. It does not isolate the
context branch's causal effect; actual branch activation norms and a
same-checkpoint context ablation remain unmeasured.

## Candidate status

| Variant | Code/config | CPU shape + backward | GPU timing | 20K training | 900-frame quality | TensorRT FP16 |
|---|---|---|---|---|---|---|
| R0 Original | available reference | prior V1 | historical only | prior V1 | prior V1 | not part of this run |
| E1 Bottleneck-96 | implemented | PASS | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| E2 Asymmetric 3D rank-64 | implemented | PASS | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| E3 Lightweight Skip | implemented | PASS | NOT RUN | NOT RUN | NOT RUN | NOT RUN |

CPU tests confirm the candidate blocks preserve tested feature shapes and
backpropagate finite gradients. All three registered configs pass the fixed
20K exposure/5K update, GPU2+3 global-batch-4 and Stage1 NFE/threshold contract.
The targeted suite reports 8 passed. Python compilation passes.

## Blocker

The current execution environment has no `/dev/nvidia*`; `nvidia-smi` cannot
connect to the driver and the Stage1 environment reports CUDA unavailable.
The installed Stage1 environment also lacks `torch_scatter`, preventing an
actual VoxDet construction smoke. Consequently the mandated RTX 3090 block
microbenchmark, full Encoder test, 20K training, 900-frame PR/fixed-K
validation, and TensorRT FP16 comparison were not started. No process was
preempted.

## Next valid gate

Restore the existing Stage1 CUDA runtime and its pinned `torch_scatter` build,
then run the real-shape block and full-Encoder tests on admitted GPU2/3. Train
only candidates with measured compute benefit and a successful full Encoder
smoke. Evaluate the resulting EMA checkpoints against the same 900-frame
manifest and fixed K values `78,694 / 84,963 / 94,087`. Only quality-qualified
candidates proceed to same-model PyTorch FP16 versus TensorRT FP16 comparison.
100K training remains outside this study's automatic scope.
