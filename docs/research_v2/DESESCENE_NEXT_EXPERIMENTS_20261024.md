# Registered next experiments: local Child state and Encoder adaptation

User authorization, 2026-10-24: upload completed research to **FPSGen-private**; implement local positional encoding and lightweight attention for Stage2's2×2×2 state; also validate Encoder fine-tuning or retraining. Continue separate single-card experiments, Batch2, one shared training seed and20K successful frame exposures each. No automatic100K or extra training seeds. Existing jobs and historical artifacts remain intact.

## First pair: encode the eight local state values

Keep the original eight-state MLP,240D early fusion,64D PTv3, time conditioning, eight-velocity decoder, Gaussian source and FP32 Euler integration. One PTv3 token remains per Parent. Add a residual **local-state encoder before early fusion**, with a zero-initialized output projection for exact initial compatibility with the original source weights.

- **L0 Position:** shared scalar state embedding + fixed XYZ Child-center coordinates + a small per-Child feed-forward block; flatten the ordered eight features and project to a64D residual.
- **L1 Position+Attention:** the same branch with one lightweight self-attention block over only the eight Children of each Parent. Initial setting: width16, two heads, no dropout, no cross-Parent attention.
- Existing B0 is the original no-branch reference, trained for the same additional20K budget from the same source. L0 versusL1 isolates the incremental attention block; it does not perfectly match parameter count. A parameter-matched or attention-without-position ablation is conditional on a real first-round gain.

Child ordering is `k = x + 2*y + 4*z`. XYZ offsets are derived from the geometry's actual Child-center positions, not an assumed ZYX reshape. State and position are kept distinguishable before ordered pooling. Invalid Children are masked as attention keys and queries. An all-invalid Parent must produce finite zero branch output. The source noise and geometry retain their original precision.

Use the same raw-GT Stage2 source as B0/M1 (SHA `eaf70f8e2cbc068add3b744b1988299197c787b8f15567fd19fd06b0d5f3f74c`), frozen mature Stage1, predicted Stage1NFE2 candidates and threshold0.5. Each arm starts fresh optimizer/EMA/scaler/scheduler, same training seed20261032,10K successful updates /20K exposures. This is additional matched fine-tuning, not20K total scratch learning.

Before training: CPU shape/gradient/Child-order tests; real maximum-scene packedBatch2, AMP, zero-output parity, sorting/cached-path parity, checkpoint resume and module runtime/memory checks;32-update smoke; then100→500→10,000 successful-update health stages. Final evaluation is900 Development scenes, primaryNFE4 plus NFE1/2/4/8 diagnostics and same-backend runtime. Preserve negative results. Do not select a winner from training loss alone.

Initial card preference was GPU1 andGPU3, one experiment per card. The user subsequently explicitly authorized GPU2: the waiting position arm now trains onGPU2, while attention continues onGPU3. Both passed engineering checks and are training; see [the current progress report](desescene_child_local_progress_20261024_2140/README.md). Other users' processes are not stopped or preempted.

## Encoder studies, separate from the Stage2 pair

Keep Encoder frozen during L0/L1 so the local-state change is attributable. Register these later candidates rather than combining all variables:

1. **EF1 Encoder-only fine-tuning:** initialize the original Encoder and Stage1 Flow from the same mature Stage1 EMA; freeze Flow parameters and optimize Encoder on the original Stage1 FM objective. Verify gradients pass through the frozen Flow to the Encoder. Retain a frozen-source evaluation reference.
2. **EF2 Encoder+Stage1 Flow fine-tuning:** same initialization, manifests, exposure budget, source-noise policy and optimization schedule; jointly adapt Encoder and Flow. EF1/EF2 are a matched additional20K cohort, not scratch architecture pilots.
3. **ER retraining:** use the original Dense Encoder or a specifically selected fidelity variant, from scratch for20K, with a matching R0 control. Existing valid R0/E1/E2/E3 single-GPU20K results may be reused when identities match. Do not retrain them merely to keep a card occupied. A new retraining arm needs a distinct hypothesis/configuration.

Encoder adaptation needs its own explicit pretrained-initialization runner and source/config identity validation; the original scratch-only trainer must not be bypassed or relabeled. Engineering readiness is pending at this registration. Low fine-tuning learning rates, exact frozen-parameter checks and Batch2 memory checks must precede formal launch.

Evaluate Stage1 scores, Parent/fixed-K IoU, Fine-GT ceiling and dedicated Encoder latency. Then test frozen Stage2 compatibility: changed Pyramid features can shift its conditions. Joint Encoder/Stage2 adaptation is a separate future experiment, not an automatic consequence of an Encoder-only gain.

## Existing evidence and continuation limits

- B0/M1 additional20K and900-frame tests are complete; M1 has no demonstrated benefit and is not scheduled for expansion.
- C1 Static FM/BCE have a metric tradeoff and belong to a different scratch/NFE4 cohort.
- D3 fixed-K heuristic ranking gains are below the registered2pp threshold; no Ranker training is queued.
- The completed trained-checkpoint Encoder timing study and all compact latest evidence are in [the completed results bundle](desescene_completed_20261024/README.md).

An automatic continuation queue may run only registered, engineering-verified candidates. A failed check stops that arm and retains the log. GPU availability alone is not research evidence. Further formal Encoder training remains a candidate until its runner and checks exist; it must not be described as already running.
