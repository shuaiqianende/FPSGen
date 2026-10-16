# Hierarchical Sparse Occupancy Flow — implementation and execution state

This is an independent research branch, `research/hs-occflow`, based on
`612cc976cc7f632eb1e1e1e3a765ff546fcf1acc`. Its worktree is
`/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow`. Historical code, data, experiments,
and the running PointFlow Student job are preserved. There is no new long-run queue.

## Implemented representation and supervision

`SparseHierarchyGeometry` supports `(2,2,2)` and `(3,3,2)` subdivisions without a
hard-coded K. World coordinates and grid coordinates are XYZ; flattened IDs are
X-fastest/Z-major. Child ID is `dx + sx*(dy + sy*dz)`. NumPy and Torch compute
world indexing in float64, interpreting stored XYZ exactly, without epsilon
snapping. Point ROI is the open50m sphere intersected with open Z(-4,4.4).
Child validity means positive-volume cell/sphere intersection. A cell with its
center outside the sphere can retain valid target hits. Boundary round trips
are tested. CUDA indexing parity is pending GPU admission.

New targets are sparse **surface-hit occupancy from pre-Poisson map candidates**.
They are not semantic occupancy, proven free space, density regression or fixed
180k points. Exact source functions, maps, calibrated poses and viewpoint-support
scans are versioned and hashed. The dataset stores uint32 parent IDs plus packed
child masks, not dense fine fields or point clouds.

## Models

All new Stage2 parameters start randomly. The default parent network has width64,
two official PTv3 encoder levels64/128, depths2/2, one decoder level of depth2,
pooling stride2, attention patches64/128 and4/8 heads. It uses the already vendored
official PTv3 code, instance normalization in sparse stem/pool/unpool, LayerNorm
in transformer blocks, FlashAttention, and tensor-only activation checkpointing.
Its true GPU behavior is not yet validated for this adapter.

One input parent produces one PTv3 token. Local state MLP maps[M,K]→[M,64].
LiDAR pyramid, parent XYZ, raw candidate score, input-observed child hits and
32D scene time are fused before PTv3. The velocity decoder receives decoded
features, current K-way state and time, then emits[M,K]. Invalid children have
zero velocity and zero loss. Each scene shares one time; child noise is independent.

The new adapter passes absolute integer parent coordinates directly as
`grid_coord` and fixed geometry-derived sparse shape. It bypasses PTv3's
`coord.min(0)` quantization. IDs are canonically sorted before the backbone and
inverted afterward. CPU tests cover this adapter mapping; CUDA end-to-end
permutation invariance remains pending. Pooling/unpooling retain each parent row.

The parent grid's physical voxel size is the quantization scale, including its
known anisotropy. No0.1m PointFlow point tokenizer collapses parent tokens.

| Model, K18 | Parameters | Training target |
|---|---:|---|
| PTv3+BCE |2,559,010|Masked BCEWithLogits only|
| Independent Local FM |86,566|Masked FM MSE only|
| PTv3+FM |2,679,398|Masked FM MSE only|

The PTv3 capacity/depth is shared by BCE/FM; FM's local/time paths add~4.7% parameters.
BCE does not instantiate time modulation, read a noise state or accept a time.
Local FM is explicitly smaller, so future conclusions must account for capacity.
The CPU width16 observed-only smoke is a separate engineering model, not this table.

FM uses `x0 ~ N(0,I)`, `x1=2*occ-1`, straight interpolation and target `x1-x0`.
It has no occupancy auxiliary, Chamfer, structural mass loss, teacher target,
shortcut, Sinkhorn or PointFlow. EMA is optional parameter averaging for evaluation,
not distillation. Euler1/2/4/8 reuses one encoded condition and the same seeded
source across NFE comparisons. Initial threshold0 is uncalibrated and will be
selected/frozen on development only.

## Frozen Stage1 and conditions

The initial compatible Stage1 is the existing supervised VoxDet-Density25K:

```
/data-12/M2024-HWZ/FPSGen/experiments/voxel_point/
voxdet_density_formal_gpu23_b2_roi_20261013_094356/checkpoint_025000.pt
```

SHA256: `0af3007573297c2f963b3302e35f59d6f2d28c3163a8b8a826509b82462474d6`.
Strict CPU state loading passed. The complete density model has22,082,470 frozen
parameters; its five trunk modules total21,151,780, including both local neck
branches. This is the full existing density model, not a separately pruned E2E
encoder checkpoint. No model weights are included in Git.

The pinned upstream sources remain VoxDet
`e10d8280de96d4f03f54b616e57e7c6754044fa4` and PWM
`c6e809486db3d0b6f8503318dfea246c06c7a50e`, with original manifests under
`third_party/voxel_point`. No vendored source was changed.

Original VoxDet features use internal XYZ dense layout. The new adapter exposes
ZYX volumes for `grid_sample`, sampling world-normalized XYZ with
`align_corners=False`. Partial input uses first3 columns only. Multi-scale features
are computed once and sampled at each parent center. Stage1 density and features
share that one encoder pass in predicted mode. Cross-resolution oracle parents may
sample these256-grid features in world space;256-grid density cannot be
reinterpreted as352-grid predicted candidates.

Existing Stage1 was trained on historical Poisson density. Its recall against new
raw surface support is therefore measured explicitly. Candidate choices support
raw-count threshold, top-K, neighbor expansion and budget limits. Oracle training
uses all GT parents plus sampled nearby-empty parents; all receive the same neutral
score1 to avoid disclosing empty-parent labels via score. Sampling weights are
not treated as calibrated deployment probabilities. Predicted-parent fine-tuning
and full-development threshold selection are pending.

## Storage, runtime and resume

Continuous ID/mask binaries are memory-mapped, with structured frame/shard/offset/
length index. Manifest replacement is the atomic commit point, references carry
hashes, and explicit resume checks version/source/geometry. Interrupted uncommitted
shards are ignored. NPZ is retained only for the small I/O comparison cohort.

The trainer uses AdamW, successful-update warmup/cosine, gradient clipping,
optional AMP, B1 variable-length scenes, DDP scene partitioning and periodic fixed
development evaluation. Atomic checkpoints store model/EMA/optimizer/scheduler/
scaler, per-rank RNG, sampler cursor, geometry/data/config identity, source code
hash and Git revision. Resume rejects identity changes; Stage1 SHA participates
when it conditions the model. Fresh output with checkpoints is refused. A request
above5K updates requires an independent `formal_ready` engineering gate.
CPU exact-resume tests passed; CUDA/DDP/AMP resume is not validated.

GPU admission rejects any pre-existing compute process on physical2/3, regardless
of apparent memory headroom. Existing training is never stopped or squeezed for
this task. Data and temporary/build paths remain on/data-12. Existing isolated
environments were reused; none were modified. Full GPU paths use
`env/voxdet_density_py39/bin/python` (MMCV+PTv3); CPU tests use
`env/pwm_ptv3_pointflow/bin/python`.

## Reproduction and next gate

Run from the new worktree. Raw data remains outside the repository, read-only.

```bash
PY=/data-12/M2024-HWZ/FPSGen/env/pwm_ptv3_pointflow/bin/python
OMP_NUM_THREADS=4 "$PY" scripts/research_v2/hier_occ_flow/audit_geometry.py \
  --output experiments/hier_occ_flow/NEW_audit16 --reference-check
"$PY" -m pytest -q tests/test_hier_occ_geometry.py tests/test_hier_occ_dataset.py \
  tests/test_hier_occ_flow.py tests/test_hier_occ_inference.py \
  tests/test_voxel_point.py tests/test_voxel_shortcut.py
```

When GPU admission becomes available, benchmark actual complete scenes first:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/research_v2/hier_occ_flow/launch.sh benchmark_memory \
  --execute --physical-gpu 2 --config configs/research_v2/hier_occ_flow/proposed_ptv3_fm.yaml \
  --output experiments/hier_occ_flow/ptv3_fm_gpu_preflight.json
CUDA_VISIBLE_DEVICES=2 bash scripts/research_v2/hier_occ_flow/launch.sh train_bce \
  --config configs/research_v2/hier_occ_flow/baseline_bce.yaml \
  --output experiments/hier_occ_flow/NEW_bce_pilot --device cuda --max-updates 1000
```

These are explicit commands, not a running queue. First do matched8–16-frame
overfit with the configured baselines and validate actual maximum-token GPU memory.
Only then lock geometry, prepare full labels using the versioned preprocessing
entry point, calibrate candidates and proceed through the user-specified1K/5K gates.
The current configs are preliminary16-frame pilot configs, not full-data formal
configs. Full test preprocessing requires frozen geometry and method/checkpoint
selection. Seq08 was not read by this research milestone.

Evaluation includes full-GT false negatives for missed parents, native macro/micro,
observed/not-directly-hit and range bins, exact child patterns, empty suppression,
cross-parent contact-edge F1 and sparse connected fragmentation. For fair
cross-resolution quality use the same raw-map15cm physical target, optionally
rebuilt during evaluation; native GT-center proxies are separately labeled.
Exports are packed occupancy and variable-count cell-center PLY, not180k physical
points. Boundary-intersecting cells may have centers outside the sphere; exported
centers are cell representatives. Surface offsets/point allocation are future work.

Measured results and blockers: [data](HS_OCCFLOW_DATA_AUDIT.md),
[resolution/I/O](HS_OCCFLOW_RESOLUTION_AUDIT.md), [smoke/pilots](HS_OCCFLOW_PILOT_RESULTS.md),
[formal status](HS_OCCFLOW_FINAL_RESULTS.md).
