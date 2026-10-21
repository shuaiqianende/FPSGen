# OMD-3D V1 Research Design

Status: V1 implementation, matched 20K training, full-development PR/fixed-K evaluation, and trained-encoder timing are complete. This was an architecture pilot, not a 100K production training run or full-system Stage2 evaluation.

## Fixed pipeline

The V1 pilot preserves the 192×192×16 dense input tensor, the original LiDAR
tokenizer and float64 voxel indexing, both 3-D backbones, the Voxel FPN neck,
the classification TPV neck, the three output feature shapes, and the Stage1
flow-matching objective. Only the five blocks in
`voxel_backbone.layers.0` (2) and `local_backbone.layers.0` (3) are eligible for
replacement. The remaining stages and necks stay on the pinned VoxDet path.

R0 is the original ResNet3D. R1 is a 128→64→128 dense bottleneck with two
64-channel 3×3×3 convolutions. R2 retains that same local branch and adds three
depthwise 3-D axial convolutions with 7×1×1, 1×7×1 and 1×1×3 kernels. Its gate
uses only the current input scan's coarse occupied-cell mask and zero-padded
window coverages. It modulates the axial context contribution; it never masks
the feature field or reads occupancy labels.

The exact original skip-projection module and DropPath object are moved into
each replacement. This preserves the first-block 3×3×3 skip projection even
when stride is one, along with the Local Backbone's GroupNorm skip and the
Voxel Backbone's BatchNorm skip. The pinned Vendor implementation is unchanged.

## Pilot controls

Each registered R0/R1/R2 arm starts from scratch and uses the same dataset
manifest, sample order seed, FM source/time seed, optimizer schedule, global
batch 4 and 20,000 successful frame exposures (5,000 updates). A separately
seeded Flow U-Net state is copied to all arms so differing encoder parameter
counts do not shift the Flow initialization. Invalid updates are logged and do
not count toward the successful exposure budget. The ordinary 100K Formal
Runner contract remains intact; the 20K budget is available only under the
separate `architecture_pilot` scope.

## Current evidence

The original block audit, CPU contract tests, and GPU smoke passed. All three
variants completed the matched 20K exposure budget and 900-record EMA
evaluation; exact checkpoint hashes are documented in
`OMD3D_V1_TRAINING_RESULTS.md`. The paired 900-frame PR/fixed-K comparison
shows no quality gain for R1 or R2 over R0. Explicit-FP16 Encoder-only timing
shows a speed/quality tradeoff for R1 and no meaningful speed gain for R2.
These results do not establish convergence at the formal 100K budget, Stage2
adaptation, or full-system performance.
