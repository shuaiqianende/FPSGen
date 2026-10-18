# Stage2 FM: user-authorized100K successful training-frame exposures

Latest measured progress, loss windows and evaluation results are in
[HS_OCCFLOW_STAGE2_LATEST_STATUS.md](HS_OCCFLOW_STAGE2_LATEST_STATUS.md).
This document preserves budget/launch history; its early throughput/ETA values
are observations at launch, not current estimates.

On2026-10-18 the user requested direct training on100K samples instead of
waiting for the overfit gate. This explicitly changes the earlier QUALITY
prerequisites to monitoring. It does not waive GPU admission, full-scene memory,
gradient/finite checks, parameter synchronization, exact own-checkpoint resume,
frozen Stage1 identity or data isolation. The400-update overfit result remains
`FAIL_ORACLE_OVERFIT`; no pass flag or F1 threshold was edited.

## Actual budget and continuation

The diagnostic run was safely paused at successful update572, after1808
successful scene exposures. The earlier lineage contained160 B1 exposures,
followed by412 GB4 updates. Both rank hashes matched and the checkpoint was
readable; AMP skips werezero.

```text
past successful frames =160 +412*4 =1808
remaining successful frames =100000 -1808 =98192
new full-split updates =98192/4 =24548
absolute final successful update =572 +24548 =25120
```

These are exposures, not100K distinct frames or100K optimizer updates. The
remaining phase samples all17,870 training frames in frozen shuffled order,
roughly5.49 additional equivalent passes. Development900 remains disjoint;
only8 fixed frames are used for periodic monitoring, and seq08 is excluded.
No small-cohort index list exists in the new training configuration.

The source checkpoint is preserved separately:
`experiments/hier_occ_flow/stage2_384_preflight/fm_before_full100kframes.pt`.
SHA256: `aa8d6174f0c04ea9fd777faf5077b29c9770c71adf865675bc6a00a9dc2bbd5e`.
Online, joint Stage2 EMA,169 AdamW states, scaler and both source RNG streams
continue; the full-split sampler starts atcursor0. This is an explicit phase
transition, not a claim of identical data order. The cosine horizon changes
from1000 to25120 absolute updates at baseLR2e-4, and the boundary LR change is
recorded in `launch.json`. The model capacity and objective are unchanged.

The runner persists the actual successful-frame counter and phase counter,
validates that the budget is exactly reachable with global batch4, and requires
the frozen optimizer horizon to match that count. AMP-skipped attempts consume
data but do not increase the successful exposure counter. Exact resumption
restores counters instead of restarting a new100K budget.

## Experiment identity

- Geometry384×384×32 parents,2×2×2 children, fine768×768×64, sphere50m.
- One parent token has8 local occupancy states, pureFM MSE only.
- Physical GPU2/3 DDP, two full variable-length scenes/rank, global batch4.
- Stage2 network remains2,677,128 parameters, width64 PTv3, FP16 outer AMP,
  FlashAttention and activation checkpointing. No parent truncation/padding.
- Final frozen Stage1 EMA SHA:
  `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`.
- **Oracle support**: all GT occupied parents plus25% nearby empty parents.
  GT membership never enters a feature; score is neutral for both classes.
  Actual frozen Stage1 LiDAR pyramid and input-only observed masks condition
  Stage2. No full Stage1 Flow sampling is needed with Oracle neutral scores.
- This establishes Stage2 subdivision learning, not predicted-parent deployment.
  Predicted-parent end-to-end quality remains an independent evaluation/task.
- Sampled-condition disk caching is disabled for the complete split, preventing
  a huge M×384 feature cache. Original sparse labels/maps remain unchanged.

## Checks, execution and artifacts

CPU tests after the counter/policy changes:72 passed,1 GPU-only test skipped.
The skipped test is not claimed as a GPU pass. Existing full-scene/max-token
GPU2/3, packedB2,32-update DDP and21-vs20+restore1 evidence remains preserved.
New tests verify historical batch accounting, exact budget/restoration,
rejection of unreachable budgets, unchanged failed-quality evidence and
mandatory engineering checks even with the user-requested quality policy.

The launcher first continues20 full-split updates to592 (1888 total successful
frames). It checks actual parameter changes, both rank hashes, finite gradients,
zero AMP skips, more than12 distinct scenes/rank and the saved frame counter.
It then **automatically resumes the same frozen configuration and budget**.
No new approval, different model, fresh optimizer or additional pilot budget is
inserted. A failed numerical/checkpoint check prevents the continuation.

Config: `configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_100kframes.yaml`.
Launcher: `scripts/research_v2/hier_occ_flow/run_stage2_384_fm_100kframes.sh`.
Output: `experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes`.
Log: `experiments/hier_occ_flow/stage2_384_preflight/fm_gb4_100kframes.log`.
tmux: `fpsgen_hs_stage2_384_fm_gb4_gpu23`, window `fm100kframes`.

Resume latest is saved every500 successful updates; milestones every2500.
At the first20-update boundary a latest checkpoint is saved regardless of the
regular interval. Monitoring every1000 updates includes1/2/4/8 NFE and the same
two fixed source seeds. A numerical safe pause is distinct from completion.
The tmux shell remains open after either outcome. No BCE, PointFlow or Teacher
long-running job is queued. Existing Stage1 and older Stage2 artifacts remain.

Actual launch/first20/resume evidence is recorded under this experiment and
`docs/research_v2/stage2_384_audit`; only small JSON and code enter Git.
Training is complete only at100000 successful exposures, never at launch.

## Actual startup evidence

The20-update full-split check passed at592 updates/1888 successful exposures:
all169 model tensors changed,169 AdamW states continued, both online/EMA rank
hashes matched, gradients were finite and nonzero, and AMP skippedzero attempts.
The launcher then actually resumed the same output/config/source identity from
592 and continued beyond600 updates. This is not merely a queued continuation.
See `stage2_384_audit/full100kframes_launch.json` for the timestamped live
snapshot; the experiment's `full_split20_audit.json` contains the real check.

The first20-update median was2.59seconds after startup. Early post-resume scenes
instead took about6.27seconds/update (8 warm observations), so the initial19–22h
estimate was optimistic; roughly40–45h is the current early estimate, subject
to actual throughput across the complete split. All evaluation history remains.
