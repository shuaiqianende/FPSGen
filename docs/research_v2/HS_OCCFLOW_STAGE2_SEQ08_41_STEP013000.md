# Stage2 update13000: true two-stage seq08 evaluation

This evaluation was explicitly requested on physical GPU0. GPU2/3 training is
preserved. Stage1 and Stage2 execute in sequence using only partial XYZ and
seeded noise; Stage1 uses its actual encoder pyramid and generated predicted
parent support, without GT candidate correction. GT is used only for metrics.

Frozen Stage2 snapshot: update13000,51520 successful training-frame exposures,
online weights, SHA256
`0713ee0f6139a2dff505424517026c1767d7623cb74510a1bd629026d14c4525`.
Frozen Stage1: final100K-frame EMA snapshot, SHA256
`b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`.
The Stage2 model remains Oracle-trained; this evaluation uses predicted parents
and actual Stage1 scores. It does not constitute predicted-parent fine-tuning.

The protocol matches the completed update10000 run:41 frames08/000000 through
004000 in increments of100; Stage1 NFE1/2/4 x Stage2 NFE1/2/4. Stage1 threshold
is0.5, Stage2 symmetric endpoint threshold is0. Noise is fixed per scene and
shared across NFEs, including parent-ID-indexed Stage2 source noise across
different Stage1 candidate sets. No test-set threshold/model selection,
outlier filter, candidate expansion, truncation, GT support correction or
180K point resampling is used.

CD is half the sum of both mean Euclidean nearest-neighbor distances, in meters,
between predicted fine occupied centers and every valid original raw-map GT
point. Exact cKDTree uses eps0; no point subsampling or dense pairwise matrix.
IoU0.5m uses the previous origin[-50,-50,-4],200x200x17 grid, clipped final Z
cell at4.4, and full raw-map GT support including Stage1 missed regions.
Macro metrics average all41 scenes equally; micro results are retained.
Raw map/pose/crop/viewpoint identities and reconstructed fine/0.5m GT are
checked against the existing audited caches.

GPU0 has an existing root-owned uvicorn service, observed at346MiB and zero
GPU load before admission. It is preserved. The evaluation allocator is capped
at25% of GPU memory, with at least16GiB free required before admission. The
service PID/command/memory are checked before each scene, with termination of
evaluation if the background workload changes beyond the registered bound.
No other process is stopped or modified. GPU0 and GPU1 evaluations therefore
share quality definitions; elapsed wall times include CPU GT/CD work and are
not a controlled comparison of inference speed.

Twelve CPU checks pass for metric definitions, source-preserving Euler steps,
snapshot policy and GPU admission. Active training source SHA remains
`01ef37195ef50b412016dd5bae481afb2421d4986b64618c153ed5cf5929a3f7`.

Output:
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_eval/step013000_seq08_skip100_grid124_gpu0`.

Completion and measured aggregates are recorded below only after all369
distinct rows have been produced and independently audited.

## Completed results

Status:COMPLETE. All369 unique combinations are present, each with41 scenes,
zero invalid CDs and zero missing rows. Independent checks confirm macro means,
CD directional formula, IoU TP/FP/FN formula, and matching candidate/GT counts
against the update10000 run. Runtime was861.09seconds, excluding
initialization. Peak allocated/reserved GPU memory:3.009/
3.840GiB. These are actual evaluation results.

| Stage1 NFE | Stage2 NFE | CD(m) | IoU0.5m macro | IoU change vs10K(pp) |
|---:|---:|---:|---:|---:|
|1|1|0.3344|40.95%|+0.264|
|1|2|0.3012|43.28%|+0.321|
|1|4|0.3010|43.90%|+0.035|
|2|1|0.3315|39.43%|+0.133|
|2|2|0.2427|45.44%|-0.244|
|2|4|0.2321|46.93%|-0.234|
|4|1|0.2776|40.48%|+0.196|
|4|2|0.2333|45.66%|-0.288|
|4|4|0.2277|46.82%|-0.273|

For4+4, CD improves0.2325m->0.2277m(-2.04%), while IoU falls
47.09%->46.82%(-0.273pp). For2+4, CD improves0.2426m->0.2321m(-4.36%),
while IoU falls47.16%->46.93%(-0.234pp). These observed changes are not
statistical significance claims. There is no evidence of simultaneous voxel
coverage and geometry improvement from this interim checkpoint.

The unchanged matched Stage1-only parent-center IoUs are46.28%,48.81%,47.55%
for NFE1,2,4. Stage2 NFE4 still falls below these respective baselines. The
Oracle-to-predicted candidate/score distribution gap remains a verified
protocol limitation, without a demonstrated causal attribution. Test results
do not trigger any threshold, source, loss or training-protocol modification.

Small artifacts in stage2_384_audit retain369 per-frame rows, nine macro/micro
aggregates, comparison to10K, and the independent completion audit. Checkpoints
and raw maps/point clouds are not committed.
