# Stage2 384 pilot status

| Phase | Actual status | Successful updates | Quality result |
|---|---|---:|---|
| CPU contracts and local math |58 passed,1 GPU unit skipped| CPU test only | independent real GPU reports attached |
| Actual Stage1 strict CPU load | passed | 0 Stage2 | all parameters frozen |
| Actual PTv3 CPU construction | passed | 0 | FM/BCE random, capacity audited |
| GPU full-scene / maximum | PASS on GPU2/GPU3 |16 optimizer steps/GPU|399,672 tokens, FM7.619 GiB allocated |
| DDP + AMP + CUDA resume | PASS |32 successful updates|21 vs20+restore1 passed |
| Oracle BCE overfit |STARTING|0 /200 at report capture| quality pending |
| Oracle FM overfit |STARTING|0 /200 at report capture| quality pending |
| Matched full-split 1K BCE/FM | not started | 0 / 1,000 each | no result |
| Oracle/Predicted 5K | not started | 0 / 5,000 each | gated |
| Formal long training | not ready | 0 | not launched |

CPU tests include 8-way geometry, native fine indexing, sphere masks,
neutral-score Oracle negatives, strict predicted budgets, analytic world-space
pyramid sampling, input-only conditions, changed-GT invariance, semantic-column
exclusion, genuine small local FM/BCE gradients, constant-velocity Euler1/2/4/8,
complete missed-parent FNs, sparse export, cache corruption/versioning and CPU
model/optimizer/EMA/RNG restore. Parent-wrapper ordering uses a CPU stub;
actual GPU PTv3 ordering has now passed on real scenes, with zero permutation
error. Actual FlashAttention and SpConv execution is verified by runtime hooks.

## Registered learning gates

Overfit uses12 fixed real training frames and the same candidate/condition seeds
for BCE and FM. Validation of the short overfit uses online weights explicitly;
EMA0.999 would substantially lag at200 updates. Development monitoring uses
separately labelled EMA weights. Neither is silently substituted for the other.

The overfit evidence checker requires200 successful updates, finite updates,
last20 mean loss <0.9 × first20 mean loss, and at least one registered NFE with
macro F1 exceeding the all-candidate-children-occupied baseline by0.02. These
are engineering learning gates, not a paper superiority claim. Failed overfit
blocks pilots; it must be investigated, not bypassed by changing a boolean.

The1K comparison checks equal support policy, full training split, frame order,
PTv3 configuration, resources and validation protocol. It reports FM's extra
119,088 parameters and actual timing. Stable learning by both methods is needed
before5K. A gate pass does not establish that FM outperforms BCE.

NFE rows preserve fixed source seeds. Metrics include full-GT FNs from missed
parents, candidate recall ceilings, macro/micro fine scores, empty suppression,
distance and observed-hit breakdowns. Model inference timing excludes CPU metric
computation, which is reported separately.

Engineering checkpoints and fresh overfit checkpoints exist; learned quality
NFE curves and trained-result PLY exports are still pending. Small CPU export tests are not visual evidence
of a trained end-to-end model.

## Active queue

`tmux attach -t fpsgen_hs_stage2_384_pilots_gpu23`

GPU2 runs Oracle BCE; GPU3 runs Oracle FM, each from random initialization,
12 fixed real training frames and200 successful updates. The frozen Stage1 is
the completed100K-frame final EMA. Once both saved learning gates pass, the
queue starts matched full-training-split1K pilots. It stops on a failed gate
and never automatically launches100K Stage2 or PointFlow. Logs are under
`experiments/hier_occ_flow/stage2_384_preflight/`. Stage1 training was preserved.
