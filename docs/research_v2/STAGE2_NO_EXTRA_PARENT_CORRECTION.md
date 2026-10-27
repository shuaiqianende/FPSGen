# Stage2 no-extra-Parent correction and actual joint inference

## Confirmed mistake

The two recent Scratch20K Stage2 runs did not follow the user's requirement of GT occupied Parents without added empty Parents. The older template retained hardcoded `negative_fraction=.25` in both `scratch20k_training.ScratchSystem.loss` and `queued20k_training.QueueSystem.loss`; the older evaluator also explicitly used.25. This is an implementation/protocol error, not an authorized change. Existing GT-only support tests/configs were already present, so the recent runner regressed to an obsolete template.

The trained weights have that history. They cannot be relabelled as no-extra-Parent training by changing inference. Their earlier Oracle tables are retained as historical noncompliant-protocol results, not validation of the requested training recipe. No checkpoint, log or old worktree is deleted or overwritten.

## Enforced correction in this independent branch

- New input-only joint inference API accepts no GT Parent/Child labels, rejects empty support rather than falling back, and selects **exactly** original valid Stage1 score>0.5.
- Zero neighbor expansion, zero extra Parent sampling, zero observed-Parent union, no token truncation or GT filtering. False positives genuinely predicted by Stage1 remain; they are not the artificial25% augmentation.
- The GT helper now defaults to negative_fraction0 and rejects explicit nonzero fractions. The encoder-free assembler and generic oracle condition builder also default to0. Old runners still explicitly passing.25 fail before training updates in this branch rather than silently reintroducing the retired augmentation. A new compliant training cohort would need separate registration; none is started here.
- Original historical branch638e201 remains byte-identical for audit/reproduction. No vendor/model math changes are made. Model/metric equivalence audit lists unchanged inference implementations.

## Current evaluation

OrdinaryFM Stage1 Scratch20K EMA (`3e25ac1962d0208314801f6d56cf2cc5d9ba3203913293ba351d2eee46043c1d`), NFE2, threshold0.5 -> each of the two old Stage2 Scratch20K EMA models. Stage2 primaryNFE4, diagnostics1/2/8. Same predicted Parent tensor, actual Stage1 scores, input-only Observed Mask, source noise and full native geometry for both. Only the384D feature input differs; the Encoder is computed once by Stage1 and its exact frozen-training state is verified for the with-Encoder arm.

Both models finish generation before raw GT is loaded for each frame. Full-GT fine FN includes all missed Stage1 Parents, with Support-only IoU separately reported. Native Fine voxel centers, corrected physical IoU0.5/0.2/0.1m, exact CUDA KD-tree CD, JSD and original41 frames0..4000. No Jitter, no GT candidate assistance, no threshold tuning and no quality-pass latency claim.

These are real predicted-support inference results using **legacy noncompliant training weights**. They do not complete the requested pure-GT/no-extra-Parent training experiment. No new training is authorized or automatically launched.

GPU2 was admitted exclusively with>=22GiB free; other users'GPU0/1 tasks were untouched. GPU3 was separately admitted exclusively for a Stage1 support-fill quality control (every valid Child inside exactly the same predicted Parents). That control is not used for cross-GPU latency comparisons. Its complete41-frame Parent hashes must match the main run before publication.

## Reproduce

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/desescene-joint-fm-no-extra-parents-v1
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python -u \
 scripts/research_v2/joint_fm/evaluate.py \
 --config configs/research_v2/hier_occ_flow/joint_fm_predicted_no_extra.yaml \
 --output <fresh quality41 output>
```

No existing output is overwritten. Unit tests enforce strict thresholding, absence of GT/fallback/neighbor inputs, zero extra count, exact common conditions and rejection of the old.25 fraction. Older real-score GT-only condition tests are also rerun.

## Validation

`tests/test_joint_fm_no_extra_parents.py`, `tests/test_stage2_real_score_gt_only.py` and `tests/test_stage2_384.py`: **44 passed**, one pre-existing timm import deprecation warning. The old test asserting25 automatically added neighboring Parents is replaced by explicit GT-only/rejected-augmentation assertions, matching the user instruction; model shape/gradient/checkpoint tests remain enabled.

After the evaluator finishes, publish compact evidence with:

```bash
python scripts/research_v2/joint_fm/finalize.py \
 --run experiments/desescene_joint_fm_no_extra_v1/quality41 \
 --docs docs/research_v2/joint_fm_no_extra_parents_results
python scripts/research_v2/joint_fm/compare_support_fill.py \
 --docs docs/research_v2/joint_fm_no_extra_parents_results \
 --control experiments/desescene_joint_fm_no_extra_v1/support_fill_geometry/results.json
```

The registered finalizer already waits on the live evaluator; do not run a duplicate publisher against the same destination. The support-fill script accepts a fresh output directory and `--continuous` for CD/physical IoU/JSD. Full checkpoints, raw scans and CUDA caches remain outside Git.
