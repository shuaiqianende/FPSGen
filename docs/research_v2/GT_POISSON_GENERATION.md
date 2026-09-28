# SemanticKITTI Hard-Poisson GT Generation

The generator `scripts/generate_semantickitti_poisson_gt.py` creates a new
GT variant under each sequence's deliberately named `gt_possion/` directory.
It never reads `gt_` as a candidate source and never modifies `gt_`, `input_`,
`velodyne`, `map_clean.npy`, poses, or calibration.

## Geometry and data contract

For every frame it reuses `prepare_semantickitti.py`'s `load_map`,
`load_poses`, `read_scan`, and `viewpoint_mask` logic:

```text
map_clean.npy → 50 m pose crop → current-LiDAR transform
→ z > -4 m → 10 m viewpoint support → full candidate [N, 4]
→ hard-Poisson original-index selection → [180000, 4] float32
```

The hard-Poisson selector returns only original candidate indices.  Therefore
both XYZ and semantic label are copied from the same candidate rows; it does
not interpolate, move, duplicate, or relabel points.  A deterministic seed is
`20260928 + int(sequence) * 100000 + int(frame)`.

## Safe staged execution

Run a three-frame GPU smoke test first:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/generate_semantickitti_poisson_gt.py \
  --sequences 08 --frames 000000,002035,004070 --device cuda
```

The generator validates exact `[180000,4]` float32 output, finite values,
zero exact XYZ duplicates, index-preserving labels, and GPU PyKeOps self-KNN
spacing.  It writes each output atomically and supports safe resume:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/generate_semantickitti_poisson_gt.py \
  --sequences 08 --skip-existing --device cuda
```

Each sequence writes `gt_possion.meta.json`; repository-local aggregate logs
are in `outputs/research_v2/gt_poisson_generation/`.  Large arrays and logs
remain Git-ignored.
