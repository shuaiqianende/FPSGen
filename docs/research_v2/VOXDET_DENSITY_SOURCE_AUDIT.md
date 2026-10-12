# Source and architecture audit

Repository started on research/fpsgen-v2 at
`dbcf7e581457fa14150aa2b810c3760659cc402f`. The original dirty state is captured
in voxel_point_reports/repository_before.json. Only new experiment files are
staged; existing tracked/untracked work and original checkpoints are retained.

## Fixed upstream

VoxDet: [vita-epfl/VoxDet](https://github.com/vita-epfl/VoxDet/tree/e10d8280de96d4f03f54b616e57e7c6754044fa4),
commit `e10d8280de96d4f03f54b616e57e7c6754044fa4`.
PWM: [shuaiqianende/PWM](https://github.com/shuaiqianende/PWM/tree/c6e809486db3d0b6f8503318dfea246c06c7a50e),
commit `c6e809486db3d0b6f8503318dfea246c06c7a50e`.
Both local clones matched those heads and were left unchanged. Clones are
ignored under env/src. Selected original reference files and their SHA256s
are retained in third_party/voxel_point; active vendoring is reproducible via
vendor_sources.py. MIT license and PTv3 authorship headers are retained.

The configured official LiDAR route includes **two** 3D ResNets: [2,2,2] for
LiDAR and [3,3,3] for local aggregation, both width128. SpatiallyDecoupledFPN
contains learned plane pooling and2D deformable convolutions. These are all
retained, including split[8,8,8], tokenizer6→128, GN neck settings, stride[1,2,2],
and four six-direction guided refinements. Ident/VoxelAggregatorDual only
wrap constructors/forward and have no parameters; their modules are assembled
directly. This is not a substitute U-Net or another Student architecture.

The original-config parameter count is22,084,900. Adapted density count is
22,082,470. Tokenizer, both backbones and both necks have identical per-module
counts. Difference:20→1 class channels in two final convolutions (-2432),
plus two negative trainable scalar biases (+2). The audit constructs the
configured original20-channel modules and adapted modules using the extracted
unmodified constructor bodies; it is not a claim that the complete legacy
MMDetection training stack was installed. See architecture_audit.json.

## Explicit adaptations

- Input: replace intensity with norm(XYZ)/50, preserving six input channels;
  input_ column4 is a semantic label and is discarded by IO.
- Geometry: derive coarse/fine sizes and world centers from one spec, XYZ dense
  layout, Z-major/X-fastest point IDs, cell/disk-intersection validity.
- Head/loss:1-channel main and auxiliary density, FP32 softplus, density losses,
  occupied-run geometry targets. Remove semantic CE/scaling functions and GT
  semantic refinement/car-class filters from the active route.
- Compatibility: remove registry/unused imports while extracting configured
  official class bodies. Use MMCV1.7.2 factories/tokenizer BaseModule and original MMEngine neck
  BaseModule; do not require the
  full camera/MMDetection3D data stack. No network width/depth was reduced.
- Coordinates: official create_grid stacks XYZ despite grid_sample's W,H,D
  order and normalizes by size-1 under align_corners=False. Adapted create_grid
  stacks ZYX and maps voxel centers using2*(index+.5)/size-1. An asymmetric XYZ
  ramp sampling test verifies no transpose/mirror/stretch.
- Memory/precision: whole Stage1 module checkpointing; PTv3 block reentrant
  checkpointing with exact dense/sparse input preservation; Stage2 sparse eval
  FP32 + FlashAttention FP16; process/rank-local KeOps device caches.

The source manifest records class extraction and namespace-only PWM changes.
Runtime checkpoint/precision adapters live outside vendor files. Their numerical
parity against the unmodified PWM reference adapter is separately tested, with
max forward and gradient differences0 on the recorded GPU test.

## PWM audit

The pinned tree does not contain pointworldmodel/core/data. Its training scripts
cannot be assumed directly runnable. FPSGen's strict numeric pairing/count
adapter provides the missing minimal data contract. Only PTv3 reference trunk,
serialization, time conditioning and flow adapter are imported. Teacher mapping
semantics follow mapping_teacher.py: source/context flags0/1, fixed time0,
source-only displacement. Student uses that same PTv3 architecture with6 versus
Teacher7 input channels; counts are14,692,739 and14,708,739 parameters respectively.
Quantization/fine/coarse token sizes0.1/0.1/0.2m are explicit metric-scene
hyperparameters; Transformer depths3/4/2, widths128/256, heads8 and serialized
windows64/128 are retained. Internal grouping never reduces output cardinality.

PWM allocate_counts silently substitutes uniform density for invalid/zero
weights; this is removed. PWM default anchors allow crossing cells; FPSGen uses
true conditional Gaussian cell/disk sampling. PWM sqrt-CD's0.5 convention is
preserved, using differentiable selected KNN distances rather than its dense
CPU fallback. Source rows remain indexed through Teacher, sparse barycenters
and velocity MSE. OT graph/coalescing/solver are reused from existing FPSGen;
coverage/marginal policies are added in the new adapter only.

## GT height and origin evidence

Existing `generate_semantickitti_poisson_gt.py::_full_candidate` explicitly
filters `local_z > min_z` and `< max_z`, defaults -4.0/4.4; its GPU branch uses
the same conditions. Existing GT_POSSSION_TRAINING_CONTRACT.md and
GT_POISSON_GENERATION.md document the current-LiDAR transform,50m crop and
10m viewpoint support. The new grid follows this height contract. Original GT
candidate cropping is a3D range crop; the new explicit cylindrical ROI is not
inferred from an object centroid or AABB.100 actual GT frames pass the new
strict cylinder/height/finite/count checks. Labels remain unused.

For comparison only, LiDiff was inspected at actual upstream HEAD
`570e73c5c8200813adf834822f0151f12dcef105` from
[PRBonn/LiDiff](https://github.com/PRBonn/LiDiff/tree/570e73c5c8200813adf834822f0151f12dcef105).
SemanticKITTITemporal.py filters partial/full z>-4; it does not provide VoxDet's
fixed grid. Its eval_path.py also contains -4<z<4.4. These were not used to guess
GT bounds. Source hashes are in data_source_provenance.json. No LiDiff model
or data-generation code was modified.
