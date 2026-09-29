# `gt_possion` training-data contract

All **new** FPSGen training and acceleration experiments must explicitly set:

```yaml
data:
  gt_dir: gt_possion
```

The Dataset pairs each `gt_possion/<frame>.ply` (or legacy `.npy`) with the
immutable sibling `input_/<frame>.npy`; it does not use string replacement and
therefore cannot accidentally look for `input_possion`.

Historical baseline configurations intentionally retain the default
`gt_dir: gt_` for checkpoint reproducibility. The DCD-only Teacher run already
in progress when this contract was introduced remains an old-`gt_` experiment;
it must not be switched mid-run. New BEVFlow speed probes use
`configs/research_v2/train_bev_gt_possion.yaml`.

## LiDAR-scan viewpoint-support contract

`scripts/generate_semantickitti_poisson_gt.py::_full_candidate` rebuilds the
GT candidate from `map_clean.npy` and applies the same coarse LiDiff/FPSGen
viewpoint support as `prepare_semantickitti.py`:

```text
map_clean world crop (<50 m around the frame pose)
  -> transform into the current LiDAR frame
  -> -4.0 m < z < 4.4 m
raw current LiDAR scan, range <50 m
  -> occupied 10 m XYZ voxels
  -> retain only map GT in those occupied voxels
  -> fixed voxel candidate selection + Hard-Poisson
```

This is **scan-support filtering**, not per-ray rendering or an attempt to
copy the raw scan's point-density distribution: a map point is eligible when
its 10 m spatial cell contains at least one raw scan point.

Read-only reconstruction checks over spatially separated generated PLY files
confirmed the contract. In each sample, all 180,000 output rows belonged to a
10 m voxel occupied by the corresponding raw LiDAR scan:

| Sequence / frame | Scan occupied 10 m cells | GT occupied cells | GT rows outside scan support |
| --- | ---: | ---: | ---: |
| 00 / 000000 | 97 | 96 | 0 / 180,000 |
| 01 / 000550 | 75 | 74 | 0 / 180,000 |
| 03 / 000400 | 100 | 100 | 0 / 180,000 |
| 10 / 001000 | 63 | 62 | 0 / 180,000 |

The machine-readable audit is
`outputs/research_v2/gt_poisson_generation/viewpoint_support_audit_samples.json`.
