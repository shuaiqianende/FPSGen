# Stage2-guided Encoder: engineering and pending experiments

Status: **CPU gradient path verified; real GPU/memory/learning/quality NOT TESTED. P2 waits for P1 completion.**

The new path is `stage2_differentiable_batch()` in `encoder_adaptation.py`. Original frozen inference functions are preserved. Frozen mature Stage1 chooses Parent IDs and scores without GT. Labels are looked up afterward. Trainable Encoder reprocesses both original LiDAR scans in one realB2 forward; differentiable `sample_pyramid()` produces384D half features; packed PTv3 consumes them. Neither sampling nor the trainable feature path is detached/cached. Source noise and Euler integration stayFP32.

S2-E1: Encoder trainable, Stage2 weights/states frozen; existing non-vendor block checkpoint wrappers retain input gradients. S2-E2: Encoder+original Dynamic PTv3 trainable. Stage1 Flow is frozen in both. S2-E0: Encoder frozen/eval, Stage2 only trainable, matching the new runner's feature/forward representation.

Strict CPU source loading passed for all three arms. Trainable tensor counts are169(S2-E0),198(S2-E1),367(S2-E2); shared initialization hashes match EF1/EF2. CPU surrogate tests demonstrate `FM loss -> frozen Stage2 -> grid_sample -> Encoder` and frozen-state preservation. They do not substitute for real SpConv/FlashAttention Batch2 backward tests.

Before training, the verifier uses the two largest observed Support scenes; logs pyramid/parameter/gradient bytes, peak allocated/reserved memory and real gradients. Replay bounds are pre-registered(max absolute1e-7, relativeL2 1e-6); any difference is labeled numerical replay. Failure/OOM stops formal launch without truncation. An additional original-B0 versus new-B2 forward/gradient comparison is recorded for S2-E0. A two-scene result alone cannot justify900-frame or all-training equivalence; hence the registered new20K S2-E0 control is retained where equivalence is unproven.

Full900 evaluation will separately publish Teacher-Support adapted-feature results and the single shared Encoder's deployable Stage1 Support results. Teacher-only improvements do not qualify as deployment success. Original mature Stage2 online weights and surface-hit GT are used; no L0/L1/M1/INT8/auxiliary loss is combined.

| Arm | GPU Batch2 |32 success smoke |20K extra exposures |900 geometry | Decision |
|---|---|---|---|---|---|
| S2-E0 matched control | NOT TESTED | NOT TESTED | NOT STARTED | NOT TESTED | Wait P1 |
| S2-E1 Encoder only | NOT TESTED | NOT TESTED | NOT STARTED | NOT TESTED | Wait P1 |
| S2-E2 joint | NOT TESTED | NOT TESTED | NOT STARTED | NOT TESTED | Wait P1 |
