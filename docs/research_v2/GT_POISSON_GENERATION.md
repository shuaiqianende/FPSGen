# SemanticKITTI Hard-Poisson GT Generation

The generator `scripts/generate_semantickitti_poisson_gt.py` creates a new
GT variant under each sequence's deliberately named `gt_possion/` directory.
It never reads `gt_` as a candidate source and never modifies `gt_`, `input_`,
`velodyne`, `map_clean.npy`, poses, or calibration.

## Geometry and data contract

For every frame it reuses `prepare_semantickitti.py`'s `load_map`,
`load_poses`, `read_scan`, and `viewpoint_mask` logic:

```text
LiDiff map_clean.npy (already moving-filtered, >3.5 m filtered, 0.1 m mapped)
→ 50 m pose crop → current-LiDAR transform → -4.0 m < z < 4.4 m
→ raw LiDAR scan <50 m → 10 m viewpoint support
→ fixed 0.15 m voxel original-index downsample
→ hard-Poisson original-index selection → exactly 180000 points
→ gt_possion/<frame>.ply
```

The hard-Poisson selector returns only original candidate indices.  Therefore
both XYZ and semantic label are copied from the same candidate rows; it does
not interpolate, move, duplicate, or relabel points.  A deterministic seed is
`20260928 + int(sequence) * 100000 + int(frame)`.

Output is binary little-endian PLY with four `float32` properties in order:
`x`, `y`, `z`, and `label`.  Map crop, local-frame transform, and height
filtering run on GPU when `--candidate-device cuda` is selected; this is an
implementation acceleration and does not alter the crop contract.  The fixed
voxel and spatial-hash hard-Poisson selection remain index-preserving.

## Safe staged execution

Run a three-frame GPU smoke test first:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/generate_semantickitti_poisson_gt.py \
  --sequences 08 --frames 000000,002035,004070 --voxel-size 0.15 \
  --min-z -4.0 --max-z 4.4 --max-range 50.0 --device cuda --candidate-device cuda
```

The generator validates exact 180,000-vertex float32 PLY output, finite values,
zero exact XYZ duplicates, index-preserving labels, and GPU PyKeOps self-KNN
spacing.  It writes each output atomically and supports safe resume:

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/generate_semantickitti_poisson_gt.py \
  --sequences 08 --skip-existing --device cuda
```

Each sequence writes `gt_possion.meta.json`; repository-local aggregate logs
are in `outputs/research_v2/gt_poisson_generation/`.  Large arrays and logs
remain Git-ignored.
