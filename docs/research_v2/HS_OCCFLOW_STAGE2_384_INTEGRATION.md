# Stage1 Occupancy → Stage2 integration

## Actual frozen Stage1 identity

Active runs use final384 binary Occupancy checkpoint stage1_final_100kframes.pt,
34,275 successful optimizer updates and100,000 successful frame exposures.
SHA256: b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2.
Strict loading and actual frozen encoder/flow forwards passed on GPU2/GPU3.
It does not call the legacy Density25K adapter.

The table below preserves the INITIAL27K interface-development snapshot; it is
not the checkpoint used by the active training configs.

| Field | CPU-verified value |
|---|---|
| Snapshot | experiments/hier_occ_flow/stage2_384_preflight/stage1_intermediate.pt |
| Updates | 27,000 |
| Successful frame exposures | 70,900 / 100,000 |
| Weights | EMA |
| Checkpoint SHA256 | b62b33fb285e38ae0c799388a38db7bc034d127d242bba4097b36b334cedf980 |
| Stage1 config SHA256 | 06badff24b884320062c6734509846f480b7bff7076ae1215679f77cc49ddc10 |
| Encoder parameters | 19,056,568 |
| Stage1 Flow parameters | 524,073 |
| Frozen parameters | all 19,580,641 |

The preceding table is historical interface-development evidence. Formal
Stage2 refuses an intermediate checkpoint. Changing the selected Stage1
checkpoint requires a new config/cache/gate identity; it is not silently swapped.

Expected feature shapes are [1,128,16,192,192], [1,128,8,96,96],
[1,128,4,48,48]. Actual GPU shapes passed all16 full-scene rows per GPU. `encode_features` checks every actual forward.

`FrozenStage1OccupancyAdapter` validates checkpoint SHA before loading, then
strict model state, Stage1 experiment/objective, scratch lineage, rank sync,
geometry, original frozen config, manifest checksums and raw-map target version.
Calling `train(True)` keeps the adapter recursively in evaluation mode.

## Data and conditions

Complete sparse data is reused read-only from the Stage1 worktree:

```text
experiments/hier_occ_flow/occ384_full_parallel/train        17,870 frames
experiments/hier_occ_flow/occ384_full_parallel/development     900 frames
guard                                                       360 frames
```

Train/development overlap and seq08 inclusion are rejected. The 900-frame
manifest is a split identity; these pilot configs evaluate only eight fixed
development frames, not 900 frames every time. No seq08 result selects a
threshold, checkpoint, noise scale or training hyperparameter here.

Predicted conditions are constructed before target lookup. Changing GT labels
leaves every predicted condition unchanged in the CPU leakage regression test.
Only input XYZ is read; changing the fourth semantic column has no effect.

Cache identity includes Stage1 SHA/weights, geometry, candidate policy, input
file SHA, frame seed and preprocessing version. Oracle support SHA is cache
metadata only. Sampled feature precision is consistently FP16 for uncached
training, cached training and input-only export. Per-rank caches avoid writers
colliding. Full-dataset pilots disable feature caching by default: approximately
200K × 384 × 2 bytes is already 154 MB per frame, so naive full caching would
cost terabytes. Overfit caching is bounded to8 GiB per run, partitioned by output
namespace and rank to avoid concurrent BCE/FM cache writers colliding.

## Entrypoints

```bash
# CPU validation and actual full-split token audit
env/voxdet_density_py39/bin/python -m pytest -q tests/test_stage2_384.py
env/voxdet_density_py39/bin/python scripts/research_v2/hier_occ_flow/audit_stage2_384.py --help

# GPU execution requires idle physical GPU2/3 AND saved engineering evidence
CUDA_VISIBLE_DEVICES=2,3 bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh \
  --config configs/research_v2/hier_occ_flow/stage2_384_oracle_fm.yaml \
  --output experiments/hier_occ_flow/stage2_384_oracle_fm_overfit
```

The shell resolves the shared project environment from the main Git directory;
the shown relative Python paths assume the main repository. The actual verified
interpreter is `/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python`.

Checkpoint resume is explicit and restricted to the same experiment identity,
Stage1 SHA, config, data, source hash, output and world size. Atomic checkpoints
contain model, full EMA, AdamW, scheduler, AMP scaler, rank RNGs, sampler cursor,
attempt/success counters and rank hashes. Signals request a boundary save;
no launcher stops another task.

Deployment export accepts an input-only `{id,input}` manifest and predicted
support only. It produces sparse fine occupancy NPZ, occupancy-center PLY and
provenance. These points are occupied voxel centers, not an allocated 180K
point cloud. Actual GPU input-only export is still pending.
