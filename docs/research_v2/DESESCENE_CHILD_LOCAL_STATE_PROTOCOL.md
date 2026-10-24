# Local XYZ state encoding and lightweight Child attention

This pilot implements the user's October24 request. Branch: `research/desescene-child-local-state-v1`, based on the completed evidence commit `e6d6875` in **FPSGen-private**. The previous M1 state–condition reinjection did not improve the matched900-frame comparison; it is preserved as a negative result and not combined with this experiment.

## Hypothesis and exact scope

The current dynamic state is eight scalar Child occupancies. The original8→64→64 MLP can learn their fixed order, but has no explicit geometric sharing or local attention. Test whether a small shared Child representation with XYZ positions helps. No new information is introduced by fixed positions, so a gain is not guaranteed.

The two new arms are:

| Arm | Additional local computation | GPU preference |
|---|---|---|
| L0 / position | Shared1→16 state projection + XYZ3→16 positional projection; per-Child LayerNorm/FFN; ordered8×16→64 output | GPU2, newly authorized after GPU1 wait |
| L1 / attention | Same, plus one16D/two-head self-attention block over exactly eight Children | GPU3 |

Add the64D result to the original state MLP output before the unchanged240D fusion. The new output projection is initialized to zero. The original source weights load strictly, and every old tensor remains present under its original key. The output projection can receive gradients immediately; upstream branch parameters begin receiving gradients after it opens. Do not interpret zero-initialized compatibility as evidence of learned quality.

The original Parent position/score/observed/LiDAR embeddings, PTv3 width/depth, time AdaLN,8D velocity decoder and one-token-per-Parent contract remain intact. Stage1 is frozen. Gaussian source, FP32 state integration, NFE, candidate threshold and ROI are unchanged. This is not a Child-as-PTv3-token expansion.

## Geometry and masking

Use the geometry's actual X-fastest Child order `k=x+2*y+4*z`. Local normalized Child-center coordinates are the eight combinations ofXYZ±0.5. These are relative coordinates, not world-coordinate quantization; geometry and indices are not converted to FP16.

Attention has no cross-Parent or cross-scene interaction. Invalid Child values are replaced before feature construction. They cannot be keys and are zeroed as queries after each block. The all-invalid case uses a finite masked softmax and yields zero residual. Ordered flattening retains the identity of every Child slot. Attention probabilities and their renormalization use FP32; network projections follow the active AMP/weight dtype. No dropout is added.

## Matched learning protocol

- Same original raw-GT source Stage2 online checkpoint as the completed B0/M1 pilot: SHA `eaf70f8e2cbc068add3b744b1988299197c787b8f15567fd19fd06b0d5f3f74c`.
- Same mature Frozen Stage1 EMA, predicted Stage1NFE2 support, threshold0.5, no GT candidate addition or truncation.
- Both start fresh optimizer, EMA, scaler and scheduler. Training seed20261032; Batch2;10,000 successful updates /20,000 successful frame exposures.
- This is additional matched fine-tuning from a source with99,520 prior exposures, not20K total scratch training. The completed B0 is the no-extra-branch reference under the same data/initialization schedule.
- Same AdamW2e−4, weight decay0.01, warmup500/cosine, clip1, EMA0.999 and source/time sampling as B0. Skipped AMP attempts never count toward successful exposures.

L0 versusL1 isolates the extra attention block, with a small parameter-count difference. This is not a perfect parameter-matched ablation. Only a promising result warrants a later equal-capacity control or attention-without-position arm.

## Checks and execution

1. CPU: exact initial compatibility; XYZ order; invalid and all-invalid masking; no cross-Parent leakage; permutation; gradients after the output projection opens; state-dict roundtrip; immutable paired configuration.
2. GPU3: real maximum-support Batch2 using the two largest scenes from the fixed900-frame NFE2 census. Compare initial velocities, four Euler steps and Occupancy exactly; verify sorting/cached paths, AMP gradients, memory and diagnostic one-step time.
3. Checkpoint replay: report bitwise exactness separately. The original SpConv embedding weight-gradient is nondeterministic; the existing bounded numerical replay check uses an independent unmodified B0 control. Do not claim bitwise training reproducibility.
4. Both arms receive a separate32-successful-update smoke on GPU3. These engineering weights are discarded. The real evidence gate is tied to the final source/config/checkpoint identities.
5. Formal L0 originally waited for GPU1. After explicit user authorization, the never-started L0 arm moved to idleGPU2; L1 continues its original GPU3 process. Neither arm waits for the other's long training to finish.100→500→10,000-update health stages preserve optimizer/EMA/scheduler/scaler/RNG and the original full scheduler horizon.
6. Each completed arm automatically runs900-scene primaryNFE4 native/continuous evaluation and64-scene NFE1/2/4/8 and condition diagnostics. GPU admission is repeated between tasks. No existing process is stopped.

Main evaluation: full-GT native Fine IoU/F1, support-only IoU, CD, corrected physical IoU0.5/0.2/0.1m, JSD, occupancy counts and paired per-frame changes. Check Parent-ID hashes and Fine-GT ceiling as invariants. Use native voxel centers for all models. Dedicated same-backend runtime is required before claiming a deployable quality/efficiency gain; isolated one-step timing is diagnostic.

The first screening target follows the prior pilot: approximately+1pp native Fine IoU with paired grouped95% interval above zero, no meaningful CD/fine-scale regression and modest measured runtime overhead. Small/no gains are archived; no automatic100K or additional training seed is scheduled.

## Encoder follow-ups requested by the user

The separate [registered plan](DESESCENE_NEXT_EXPERIMENTS_20261024.md) includes Encoder-only fine-tuning with frozen Flow, matched Encoder+Flow fine-tuning and hypothesis-driven scratch retraining. Each uses20K successful exposures and explicit initialization identity. Keep these changes separate from L0/L1. Their runner and engineering checks must exist before they can auto-start; registration is not a claim that Encoder training is already running.

The persistent backlog is `configs/research_v2/hier_occ_flow/desescene_research_backlog.yaml`. Existing valid scratch20K R0/E1/E2/E3 results can serve as identity-matched references. The current PyTorch Encoder wall means on GPU3 are R0 180.05ms, E1 177.72ms, E2 162.00ms and E3 167.26ms; these are not full-generation or TensorRT results.

## Initial engineering evidence

The actual largest-scene pair has165,919+165,513 Parent tokens. Both local modes pass exact zero-initialized velocity/endpoint/Occupancy, sorting and cached-path tests, and finite nonzero branch gradients. Checkpoint replay uses the disclosed numerical SpConv-embedding control. Peak allocated memory is8.45GiB. Added parameters are9,376 forPosition and10,496 forPosition+Attention.

Diagnostic maximum-Batch2 one-step CUDA times (three warmed repetitions, physicalGPU3) are198.15ms baseline,211.23ms position and229.69ms position+attention. These are not formal end-to-end timings or evidence of a speed gain. Quality is NOT TESTED until the20K pilot and900-frame evaluation complete. See the retained [GPU report](DESESCENE_CHILD_LOCAL_STATE_GPU_VALIDATION.json).

## Current execution and future readiness

Both32-update smokes and the100/500-update health checks have passed. L0 is now training onGPU2, L1 onGPU3. The replacement supervisor preserves the running L1 PID and original pinned training source hash; only the waiting L0 assignment changed. The actual continuation script and transfer audit are retained with [the latest progress snapshot](desescene_child_local_progress_20261024_2140/README.md).

That snapshot includes200-successful-update mean losses, B0 comparisons, saved checkpoint identities and the actual automatic evaluation schedule. It distinguishes active training from Encoder candidates whose engineering checks and automatic training continuation are not yet implemented. No learned quality gain is claimed while the pilots are incomplete.
