> Latest decision2026-10-27: traditionalFM retained; A has completed20K/41 frames; B explicitly saved/stopped by user at6664 exposures, no automatic resume. See [decision](MEANFLOW_V2_FINAL_DECISION_20261027.md). Earlier snapshots below remain historical.

> Historical launch snapshot below. Latest A15K interim evaluation: [report](MEANFLOW_V2_A_INTERIM_15K_RESULTS.md). B was reassigned to GPU3 in branch `research/desescene-meanflow-v2-b-gpu3`; this evaluator does not launch training.

# MeanFlow-V2 execution status

Snapshot: 2026-10-27T15:11:50; not a completed20K quality claim.

## Completed evidence checks

Original MF:10,005 attempts /10,000 successful updates /20,000 exposures /5 AMP skips. Last200 successful raw loss mean20.21329893, median0.13158702, p95=106.6668743, p99=295.0447479, max1239.574707. Across all attempts:

| Batch interval type | Attempts | Raw mean | Raw median | Raw max | AMP skips |
|---|---:|---:|---:|---:|---:|
| Diagonal only |5641|0.175084|0.119761|2.003332|0|
| Mixed |3713|23.786955|2.125189|1509.359619|1|
| Finite only |651|47.181551|11.763179|2179.813721|4|

All loss>10 attempts involve finite intervals. This establishes association, not causality. Old logs lack per-scene times/gaps/JVP norms; long-gap concentration, JVP amplification and causal direction remain NOT TESTED from those logs. Final quality remains the original41-frame data in snapshotd6a3a60.

## Engineering / execution

-46 CPU mathematical/protocol/regression tests passed.
-A maximum trueBatch2 GPU_PASS: full dense grid; global-largest positive-support scenes02/925(319738) and02/4192(319737), no input reduction.
-Initial shared FP32 and rounded FP16 model hashes exactly match original scratch MeanFlow. No new parameters in A/B.
-Encoder and Flow finite nonzero gradients; parameters change; FP32 master/Adam/EMA policy audited.
-Disk checkpoint/optimizer/RNG/EMA roundtrip exact. Native resume is **NUMERICAL_NATIVE_REPLAY_WITHIN_FIXED_BOUNDS**, global gradient relativeL2=6.15345558e-05; not bitwise reproducibility.
-Peak allocated=8.4038GiB, reserved=9.8750GiB.
-A discarded Smoke completed32 successful updates /64 exposures /0AMP skips in79.5054s. A formal run was created from the same fresh seed, no pretrained or Smoke weights.
-B GPU engineering,32-update Smoke and formal training:QUEUED_AFTER_A_EVALUATION, not yet passed or run.

## Quality and cost

| Arm | Formal status | NFE1 Parent IoU | Recall | Ceiling | CD | Full41 quality |
|---|---|---:|---:|---:|---:|---|
| Original Scratch FM | Completed |0.416000731|0.478583095|0.565898293|0.339894727|Available in d6a3a60|
| Original Scratch MF | Completed |0.371116145|0.416729588|0.503969665|0.333522275|Available in d6a3a60|
| MFV2-A | RUNNING_FORMAL |NOT TESTED|NOT TESTED|NOT TESTED|NOT TESTED|NOT TESTED|
| MFV2-B | Queued after A |NOT TESTED|NOT TESTED|NOT TESTED|NOT TESTED|NOT TESTED|

No new result is marked GO before final quality. No900-frame evaluation or long training authorized. Per-arm final interval summaries, paired deltas, trajectory/sweep diagnostics and timing are generated automatically in the experiment directory after completion. Frozen0.5 is primary; sweep is exploratory.

Environment: Python3.9.19, PyTorch2.0.1+cu117, driver580.159.03. A standalone `torch.backends.cudnn.version()` query reports compiled8.5/runtime8.4 incompatibility; the actual passed training process maps the existing CUDA11.7 cuDNN8.4 library. This warning is recorded in `meanflow_v2/environment.json`, not silently corrected by changing the shared environment. Actual native Conv/JVP/backward tests and32-update Smoke passed under the original import/runtime stack.

Reports/source evidence: [Protocol](MEANFLOW_V2_PROTOCOL.md), [baseline audit](meanflow_v2/original_loss_audit.json), [A GPU gate](meanflow_v2/mfv2_a_engineering.json), [A Smoke](meanflow_v2/mfv2_a_smoke.json), [environment](meanflow_v2/environment.json).

Formal live snapshot (not final quality): `{"updates": 106, "exposures": 212, "attempts": 106, "amp_skips": 0, "raw_unweighted_mse": 1.1585087776184082, "weighted_loss": 0.9914416074752808, "gradient_norm": 2.050696611404419, "seconds": 2.2426232676953077}`. See timestamp in `meanflow_v2/execution_status.json`.
