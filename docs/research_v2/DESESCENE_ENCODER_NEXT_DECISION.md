# Next research decision: no automatic P3 formal training

Current decision: complete existing L0/L1 evidence, then gated EF1/EF2 and S2-E1/S2-E2. P3 remains **PREPARATION ONLY / NO FORMAL GO**. No extra seed,100K, new Child Attention or unregistered architecture enters the queue.

## ER selective E2 design

Historical matched scratch20K results: R0 IoU.420339/Encoder180.05ms; E1 .425641/177.72ms; E2 .422217/162.00ms; E3 .420464/167.26ms. These are PyTorch explicitFP16 Encoder timings, not generation/TRT latency.

ER-Local retains original Voxel Backbone, replaces only three Local Stage0 blocks with existing `AsymmetricFactorized3D(rank=64)`. ER-Voxel replaces only two Voxel Stage0 blocks and retains original Local Backbone. Reuse existing non-vendor block/forward wrappers; three128-channel pyramid shapes, necks, tokenizer, original FM unchanged. Source-trained E2 weights are not loaded into scratch candidates.

The independent engineering-only adapter is `encoder_retraining_preparation.py`; it reuses the existing E2 blocks without changing vendor code or old runners. CPU small-shape B2 tests cover gradients and exact preservation of the other backbone. No formal P3 training entry point is provided.

Before any training GO: full real Encoder B2 forward/backward, same-backend module/full-Encoder timing, actual parameters/memory, ONNX/TRT export feasibility. These real-GPU/deployment checks are currently **NOT TESTED for these selective structures**. Original scratch cohort used seed20261021 whereas this registration uses20261032, so historical R0/E2 are evidence for prioritization, not automatically a strictly matched new-seed control.

## S2-Scratch design

Conditional pair: mature Encoder+scratch Stage2 versus random Encoder+the identical scratch Stage2 initialization. Same frozen Teacher Support, original Stage2 FM,20K exposures/batch2 and data schedule. Never substitute Teacher-Support success for deployable shared-Encoder Support. New objectives/multitask losses require a separate registered study.

## Required GO evidence

Use actual P1/P2 paired900 geometry, stage1 Support drift, grouped CI/worst scenes, repeated runtime and costs. An A-only(Teacher-Support) gain with B deployment degradation is a research result, not adoption. Tiny/no gains are archived rather than extending training. A formal P3 GO requires an explicit evidence-based decision and matching control identity; GPU availability alone is insufficient. The supervisor ends after P2 and cannot launch ER/S2-Scratch.
