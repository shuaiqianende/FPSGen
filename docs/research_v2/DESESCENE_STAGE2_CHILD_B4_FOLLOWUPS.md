# Stage2 Encoder + local Child state: two queued scratch experiments

The present GPU2 baseline has T1 U1+A depth-moment LiDAR features injected into Stage2; GPU3 removes that feature branch. Both are scratch Stage2 FM models trained on the full registered 00–07/09/10 morphology data. These new arms both retain the same frozen T1 Encoder injection, observed mask, GT-only training Parent support without extra empty parents, no score branch, loss, parent-token PTv3 and Stage2 output. Their common baseline is the current **with-encoder** B4 run, not the no-encoder run.

| Arm | Queue | Additional dynamic state branch |
|---|---|---|
| Position | GPU2 after existing with-encoder 100K training **and** 41-frame evaluation | Encode the eight scalar states with shared 1→16 projection, XYZ child centers and a small per-child FFN; ordered 8×16→64 residual, zero output initialization. |
| Attention | GPU3 after existing no-encoder 100K training **and** 41-frame evaluation | Same position branch plus 16D, two-head attention inside each Parent's eight children. Invalid children are masked; FP32 softmax. No extra PTv3 tokens. |

These are matching scratch 100K successful-exposure, real Batch4, 25K successful-update, seed 20261032 experiments with FP16 weights/activations, FP32 masters/Adam/EMA, the same optimizer and 100K schedule as the active B4 pair. Common tensors have identical seeded initialization; new residual outputs initialize to zero. Attention adds parameters, so position vs attention is not perfectly parameter matched. Both encode the same partial LiDAR with the frozen T1 model and use no ground truth for deploy-time support.

Both queues run an actual untruncated four-largest-scenes Forward/Backward gate, separate 32 successful-update smoke, and only then fresh formal training; no silent smaller batch, token crop, fallback, or restart after numerical failure. GPU availability is checked after the prior evaluator exits. Queued 41-frame seq08 evaluations use final EMA, predicted Stage1 support and NFE2/4, native fine centers, physical IoU, exact GPU CD, JSD-3D/BEV. They first run a one-frame evaluation smoke. A failed earlier evaluation or gate blocks the dependent queue and is recorded.

Existing earlier L0/L1 were 20K additional fine-tuning of different models and data, not these matched 100K scratch arms. No quality or speed improvement is implied. CPU tests verify shared initialization, zero residual, mask and gradients; a small CUDA FP16 test validates both child branches. The full Batch4 memory and 32-update integration tests are pending until the queued GPUs are available.
