# DeSeScene Phase0: completed evidence and the next matched pilot

Snapshot: 2026-10-24. This report supersedes the pending C1 status in the October23 audit. Numbers below are from saved evaluations, not extrapolated training milestones. All point-cloud metrics use native Fine Voxel centers, the same raw-map GT/ROI, exact CUDA KD-tree CD, corrected physical-grid IoU and the shared FPSGen JSD implementation.

## Completed C1: neither objective dominates

Both independent single-GPU models completed 10,000 successful updates / 20,000 successful frame exposures, Batch2, training seed20261032. Both use frozen mature Stage1 EMA, **predicted NFE4 support**, threshold0.5 and scratch Stage2 initialization. The EMA primary evaluation covers the same900 Development scenes and sampling seed20261031. Paired input and Parent-ID hashes match on all900 scenes. Native metrics include missed-Parent Fine false negatives.

| Metric | Static FM, NFE4 | BCE, one pass |
|---|---:|---:|
| Native Fine IoU | 0.300555 | 0.353279 |
| Native Fine F1 | 0.458546 | 0.516108 |
| CD, m ↓ | 0.198340 | 0.345239 |
| Physical IoU0.5m | 0.534942 | 0.428682 |
| Physical IoU0.2m | 0.380646 | 0.372079 |
| Physical IoU0.1m | 0.141546 | 0.162320 |
| JSD3D ↓ | 0.385266 | 0.422834 |
| JSDBEV ↓ | 0.306451 | 0.390911 |

BCE improves native IoU by5.2724 percentage points, grouped paired95% CI[3.7484,6.6417]pp. Its CD increases by0.146899m, CI[0.097671,0.200242]m; its0.5m IoU drops10.6259pp, CI[−11.9724,−9.3842]pp. These intervals resample sequence/100-frame groups; only nine such groups are available. They do not describe uncertainty across independent training seeds.

This supports testing the quality/efficiency tradeoff further. It does not establish that static context is equivalent to dynamic PTv3, nor that FM should be removed. The strictly matched original Dynamic FM20K arm was absent from the previous cohort. The early FM/BCE scalar losses optimize different objectives and cannot be ranked against each other.

[All900 paired per-frame metrics](DESESCENE_C1_900_PAIRED_METRICS.csv) and [compact machine-readable evidence](DESESCENE_STATE_CONDITION_PHASE0_RESULTS.json) accompany this report. The older generic paired-summary helper labels the lower delta tail as `worst` even for CD/JSD; those fields are not interpreted as direction-aware worst regressions. The CSV contains every delta for independent direction-aware analysis.

The C1 jobs were restored from their saved9000-step states after unexplained process disappearance. The registered successful trajectory remains20K effective exposures; replayed physical computation is disclosed separately in [the recovery audit](DESESCENE_C1_RECOVERY_AND_SEED_POLICY.md). The root cause of process disappearance remains unknown.

## Other completed diagnostics

**B1:** the Child Geometry bridge does not show a consistent benefit over its position-only control. NFE4 native Fine IoU is0.308658 versus0.309067. Native IoU declines with increasing NFE for both. These are separate models/protocols and are not the new Dynamic FM control.

**D1/D2:** the mature frozen dynamic checkpoint has completed64-scene NFE1/2/4/8, repeated-sampling, teacher-forced velocity and conditional-perturbation diagnostics. For example, single-sample native IoU changes from0.323651 atNFE1 to0.309578 atNFE2, while CD improves from0.257766m to0.214850m. The checkpoint was originally Oracle-support trained; these are mechanism diagnostics on predicted support, not the new matched20K architecture comparison. Multiple diagnostic sampling seeds do not add any training runs.

**D3:** GT-free score/neighborhood/observed-prior ranking was calibrated on200 scenes and evaluated on700 held-out scenes. Candidate-pool construction never uses GT.

| Fixed K | Held-out baseline Fine-GT ceiling | Ranking gain, pp | Ranker training |
|---|---:|---:|---|
| 84,963 | 0.585480 | +0.5171 | NO-GO |
| 120,000 | 0.677213 | +0.3977 | NO-GO |

Both miss the pre-registered+2pp threshold. No learned Parent Ranker training is scheduled.

**D4 quality:** E1 Bottleneck96 is now a viable candidate in the completed single-GPU20K cohort: Parent IoU0.425641 versusR0 0.420339; Fine-GT ceiling0.598639 versus0.590896. E2 IoU is0.422217; E3 is0.420464. E1's paired IoU gain is+0.5302pp, grouped95% CI[+0.3402,+0.6913]pp. Trained-checkpoint fixed-K, dedicated same-backend timings and downstream compatibility remain unverified. E1 is not yet declared the best deployable Encoder. Three alternating trained-checkpoint Encoder timing rounds are queued after the M1 arm's evaluation.

## M1 engineering: exact initial parity, numerical training resume

The maximum Development support census identifies two scenes with165,919 and165,513 Parent tokens. The real packed Batch2 test passes:

- Exact zero-initialization velocity, four-step endpoint and Occupancy parity against B0.
- Exact Parent permutation and cached-condition results.
- Finite nonzero gate gradients, followed by gradients into the new interaction branch.
- Identical repeated inference for identical weights.

The original all-parameter bitwise resume gate failed. The discrepancy was measured rather than ignored: only `trunk.net.embedding.stem.conv.weight` differs after the replayed optimizer step. Its gradient difference is about2e−9; parameter differences are about1e−8. The unmodified B0 control exhibits the same phenomenon. Losses match exactly.

The final source-hash-matched maximum-scene check is `m1_gpu_verification_resume_control_v5`. Replay is therefore classified **NUMERICAL_EMBEDDING_WGRAD**, not EXACT. Four-step endpoints can differ after those tiny weight changes: the final M1/B0 probes have163/152 sign flips among approximately2.65 million Child slots. The numerical gate restricts parameter/gradient differences to that single tensor, bounds absolute differences at1e−7 and relativeL2 at1e−6, and requires M1 endpoint errors to stay within twice the observed B0 control. Exact initial B0/M1 parity is still mandatory. Full quality validation is not waived.

The failed exact attempts and their raw logs remain server-local under `phase0/m1_gpu_verification*`. The independent B0 control explains the location and scale of nondeterminism; it does not prove that long resumed trajectories are bitwise reproducible.

Both32-successful-update engineering smokes encountered one identical AMP scaler overflow:33 attempts,64 successful exposures per arm. A skipped attempt is retained and excluded from successful exposure counts. The gate now checks32 actual successful updates, a stable final eight-attempt window and at most two isolated scaler skips, instead of incorrectly requiring32 total attempts. Larger or persistent failures still block training.

## Authorized next experiments and automatic continuation

Following the user's latest GPU authorization:

| Card | Arm | Research variable | New budget |
|---|---|---|---|
| GPU1 | B0 Dynamic FM | Original dynamic PTv3 | Batch2,10K updates /20K exposures |
| GPU3 | M1 Interaction FM | Two zero-initialized state–condition residuals | Same |

Both use training seed20261032, the same raw-GT source checkpoint, frozen Stage1, predicted NFE2 support, data order, optimizer, fresh EMA and schedule. This is **additional20K matched fine-tuning**, not20K total scratch training. It is not pooled with C1's NFE4 scratch cohort. The source checkpoint contains99,520 prior successful exposures; its SHA is `eaf70f8e2cbc068add3b744b1988299197c787b8f15567fd19fd06b0d5f3f74c`.

The queue runs independent32-update engineering smokes, registers the evidence-backed C1 review decision, then executes100→500→10,000 successful-update stages with saved optimizer/EMA/scheduler/scaler/RNG. Health checks inspect real counters, gradients and AMP stability. Stage100/500 are health checks, not architecture quality selection. The full10K scheduler horizon is frozen from the first update.

Each completed arm automatically proceeds to900-scene primary NFE4 continuous/native evaluation and64-scene NFE/condition diagnostics. The M1 card then benchmarks the trained R0/E1/E2/E3 Encoders in three alternating5+20 rounds. GPU admission is repeated between processes; no existing compute task is preempted. GPU2 is occupied by another task and is left untouched.

No extra training seeds, M2/M3/M4 formal training or100K training are scheduled. Source changes stop the queue before another launch. M1 adoption still requires the pre-registered900-frame quality/runtime criteria; no learned improvement is claimed at this snapshot.

Reproduce the authorized queue, using a new output directory and the retained evidence paths:

```bash
/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python \
  scripts/research_v2/hier_occ_flow/run_desescene_m1_matched_queue.py \
  --output experiments/desescene_state_condition_v1/matched20k_gpu1_gpu3_v2 \
  --verification experiments/desescene_state_condition_v1/phase0/m1_gpu_verification_resume_control_v5/result.json \
  --review-decision experiments/desescene_state_condition_v1/phase0/m1_research_go_20261024.json \
  --source-worktree /data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun
```

Model weights, optimizer checkpoints, large predictions and caches remain outside Git. Both engineering smokes have completed and the paired formal training processes have launched on GPU1 and GPU3. Formal progress is read from the server JSONL and queue status files; this document is a dated snapshot.
