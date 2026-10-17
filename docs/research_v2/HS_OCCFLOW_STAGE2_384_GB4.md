# FM-only Stage2: true ragged global batch4

Update2026-10-18: the400-update queue completed but **failed the Oracle quality
gate**, so its shell exited and tmux disappeared. No full-split1K pilot ran.
See [the recovery report](HS_OCCFLOW_STAGE2_384_TMUX_RECOVERY.md) for the
unchanged quality thresholds and bounded400→1000 overfit diagnostic. Launch
snapshots below describe the earlier run, not a current live status.

The latest user instruction supersedes the earlier paired BCE/FM queue. BCE's
completed200-update result is preserved. Its coordinator was terminated by
verified PID; the FM process received its own safe-stop signal and saved at
successful update160. No unrelated process was signalled.

## Batch representation

Both physical GPU2/3 train the **same FM model with DDP**. Each rank packs two
complete scenes of different sizes into one sparse PTv3 forward/backward:

```text
GPU2: scene0[M0,8] + scene1[M1,8]
GPU3: scene2[M2,8] + scene3[M3,8]
global batch =2 ranks ×2 scenes =4
```

There is no padding to the largest scene, random token removal, separate
per-scene backward, or gradient accumulation pretending to be batch4.
Local8D states remain one PTv3 token per parent. All2,677,128 FM parameters and
their names/shapes are unchanged.

The previous adapter hardcoded batch=zeros and one scene time. The new path
uses batch IDs in sparse indices, serialization, pooling/unpooling and context.
Canonical keys are batch_id×num_parents+parent_id, permitting equal parent IDs
in different scenes. Serialization depth is fixed to the geometry (9 bits).
Instance normalization computes independent scene means/variances. Time
embeddings are[B,32] and each point selects context[batch].

Loss is mean of **per-scene valid-child FM MSE**, then DDP averages rank losses.
A larger scene does not receive extra weight merely because it has more tokens.
Every scene has its own Gaussian source and sampled t. Only FM is supervised.

## Actual single-GPU B2 validation

Both GPUs passed two real test compositions:

| Composition | Parent tokens | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Unequal real scenes |172,498+399,672|11.323 GiB|12.070 GiB|
| Two complete maximum scenes |399,672+399,672|16.058 GiB|19.130 GiB|

The repeated maximum scene is an explicit worst-case composition, not a claim
that ordinary training repeats frames. Actual frame sampling uses four distinct
positions from the frozen shuffled sampler each update. No maximum scene was
dropped. Full packed backward and AdamW updates have finite nonzero gradients.

Changing scene1 state/time leaves scene0 output exactly unchanged. Permutation
error is zero. With outer AMP disabled, packed versus separate output max error
is9.54e-7 (FlashAttention itself still usesFP16). The original strict cellwise
AMP comparison failed near zero: max error9.77e-4, mean2.74e-6, relativeL2
4.84e-5, sign disagreement about1.1e-6. This is batch-size dependent half-GEMM
rounding, not accepted as bit equality. Failed reports are retained.

After that control, the explicit dtype-based AMP contract requires relativeL2
≤FP16 epsilon, max error≤2 epsilon×max(1,abs(reference)), sign disagreement
≤1e-4 AND the unchanged strict AMP-disabled control. Exact scene isolation and
permutation are separate checks. The final runs execute this complete contract;
an edited boolean never substitutes for a rerun. DDP restore retains the original
atol1e-5,rtol1e-3 and exact RNG/sampler/integer tests.

## Continuation and gates

Source: `experiments/hier_occ_flow/stage2_384_preflight/fm_before_gb4.pt`.
SHA256: `0c793bec14f3fce91856ab6044fae430aa0dc97b83aee11df55bb62f5332fa1d`.
This is this experiment's own randomly initialized FM lineage atstep160;
it is not a pretrained external model. Online, EMA, AdamW and scaler are loaded
strictly. Geometry, Stage1 SHA, data manifests, model config and source config
hash are checked. Changing batch/world size is explicitly recorded rather
than mislabeled exact resume.

Old steps are not multiplied/divided retrospectively. The160 B1 updates account
for160 successful frame exposures; each new GB4 update adds4. Rank0 restores the
old source RNG; the new rank receives an independent deterministic stream. All
subsequent checkpoints contain both rank RNGs and the actual sampler cursor.

At the old100-update diagnostic, FM overfit macroF1 was0.540 versus the
all-occupied baseline0.564. Therefore small-scene learning is still a real gate:
the GB4 continuation extends the overfit horizon to400 total updates, retaining
baseLR2e-4 and recording the new cosine LR at transition. It does not pretend
the earlier200-update quality gate passed. Once the FM-only learning gate passes,
the queue continues onto the complete17,870-frame train split for1000 additional
updates, preserving trained state and explicitly starting a full-split sampler.
No BCE gate/run is required by this new user-selected mainline.

Active tmux: `fpsgen_hs_stage2_384_fm_gb4_gpu23`.
Log: `experiments/hier_occ_flow/stage2_384_preflight/fm_gb4_pipeline.log`.
The queue runs actual GB4 DDP restore smoke before continuation. No100K formal
Stage2 training or PointFlow task is automatically started.

## Actual launch snapshot

GB4 DDP32-update smoke completed, and21 versus20+restore1 passed. The real
continuation loaded169 AdamW parameter states and the original online/EMA
weights exactly. Atupdate180, both rank hashes matched, all169 parameter
tensors changed, and successful frame exposures were240 =160 +20×4.
Latest observed update at this report capture: 200;
AMP skips:0. This is the real continuation, not the
independent smoke weights. PhysicalGPU rank PIDs are3557816 and3557817.
