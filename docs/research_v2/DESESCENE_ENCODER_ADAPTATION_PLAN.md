# DeSeScene Encoder adaptation: registered short pilots

Research branch: `research/desescene-encoder-adaptation-v1`, isolated from running L0/L1 worktree; based on `601d115`. Private publication destination is **FPSGen-private**, never the root checkout's public `origin`.

## Sources and scope

Both checkpoint files were read and SHA256 verified on this server:

| Source | SHA256 | Selection |
|---|---|---|
| Stage1 `stage1_final_100kframes.pt` | `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2` | Strict EMA |
| Original Dynamic Stage2 `step025000.pt` | `eaf70f8e2cbc068add3b744b1988299197c787b8f15567fd19fd06b0d5f3f74c` | Strict online, original surface-hit labels |

No source checkpoint is overwritten. Stage1 old scratch validation, original frozen adapters and vendor code remain unchanged. This cohort is **mature initialization +20K additional successful frame exposures**, not scratch20K.

Each formal arm has one GPU, true batch2, seed20261032,10K successful updates/20K exposures. Dataset manifests are the original disjoint17870 Train/900 Development, excluding Sequence08. AdamW, weight decay.01,500-update warmup,10K-horizon cosine, clip1, EMA.999, FP16 AMP. Encoder LR2e-5 in every applicable arm; EF2 Flow1e-5; trainable Stage2 LR2e-4. Scheduler is never reset at100/500 updates. No100K or second seed.

## Phase dependencies

1. **P0:** preserve existing L0/L1 supervisor. Wait for both10K updates, both900-frame evaluations and64-frame NFE/condition diagnostics. Independently audit final identities, EMA/optimizer/scaler/counters, matched B0 per-frame geometry, branch output RMS versus64D state MLP and actual L1 attention. Measure three alternating5+20 timing rounds on one idle card. Do not infer a geometry win from loss.
2. **P1:** EF0 no-training frozen reference; EF1 trains Encoder only through frozen differentiable Flow; EF2 trains Encoder+Flow. Each must pass largest observed real pair forward/backward, frozen-state and strict resume checks, then32 successful smoke updates. Both arms must pass engineering before formal pairing. Continue100→500→10000 on the same optimizer/scaler/EMA/RNG. Complete900-frame evaluation before P2.
3. **P2:** same original sources, independent from EF1/EF2 results; freeze Teacher's Stage1 support atNFE2/.5. Recompute trainable Encoder features in a realB2 forward, sample384D conditions with autograd, then original packed PTv3 FM. S2-E1 freezes Stage2 parameters but not its input derivative; S2-E2 trains both. No reusable feature cache. The maximum-pair test cannot establish B0 equivalence over all training data; a separate registered S2-E0 Stage2-only control is used if complete equivalence has not been established.
4. **P3:** design/engineering/decision only; formal training never queued automatically.

The maximum-pair selector scans both completed L0/L1 training token records plus the full900 Development support census, and selects the two largest distinct scenes. No token cap or smaller replacement is used. OOM or replay/gradient failures stop the affected phase and retain logs.

## Evaluation and pre-registered decisions

All900 frames, Stage1 and Stage2 NFE1/2/4/8; primary deploymentNFE2/4. Stage1 Parent IoU/F1/Precision/Recall, counts, fixedK84963/120000 and Fine-GT ceiling. Fine metrics include every GT child, including parents missed by Stage1, support-only diagnostics, distance/observed partitions, physical0.5/0.2/0.1m IoU, exact CUDA KD-tree CD and original JSD.

Report **A Teacher-Support+Adapted Features** and **B shared Adapted-Encoder predicted Support+Features** separately. A is diagnostic, not a two-encoder deployment proposal. Original noise policy and Child order `k=x+2y+4z` are preserved; changing support can change row-indexed RNG consumption, so deploymentA/B is not labeled identical per-Parent noise.

Stage1 screening: ParentIoU gain≥.003 or fixedK ceiling≥.005, grouped paired CI lower bound>0. Stage2 screening: deployment NativeFineIoU gain≥.01 and CI lower bound>0. Additional conservative compatibility guards: deployment FineIoU drop≤.005, CD increase≤.5%, physical IoU drops≤.002. JSD increases and worst-frame regressions trigger review; no automatic model adoption. All numerical failures/negative results retained. Confidence intervals resample sequence/time blocks, not independent training seeds.

## Timing identity

Three alternating rounds,5 warmup+20 timed Development inputs preloaded on GPU; same physicalGPU and same PyTorch originalAMP backend for all arms. Main wall covers GPU XYZ→GPU Fine Coordinates, including the unchanged CPU selection/observed operations. Encoder/Flow/PTv3 CUDA Events are auxiliary. Metrics/disk/checkpoint loads excluded. Jitter and historical V7 TensorRT330ms are outside this comparison; no cross-backend speedup claim.

## Reproduction and supervision

Use `/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python` in this worktree. All task artifacts live under `experiments/desescene_encoder_adaptation_v1/` and stay out of Git. Configurations are `configs/research_v2/hier_occ_flow/encoder_adaptation_*_20k.yaml`; experiment matrix is `desescene_encoder_experiment_matrix.yaml`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -m pytest -q tests/test_encoder_adaptation.py
/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python scripts/research_v2/hier_occ_flow/run_encoder_adaptation_phases.py --publish-phases
```

The supervisor takes an exclusive own-run lock, pins sourceSHA, waits for P0's existing completion file, checks idleGPU2/3 before each child, and records a task PID/log. It never kills an existing process. Training sources must not be edited after registration. Failed/partial artifacts are preserved for manual review; there is no silent eager fallback, batch reduction or automatic retry with changed semantics. Phase publication allowlists compact result documents and pushes explicitly to private repository/branch without force.
