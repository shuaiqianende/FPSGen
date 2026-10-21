# OMD-3D V1: Original Stage0 Block Audit

The pinned source is `fpsgen/voxel_point/vendor/voxdet/resnet3d.py`; this file
was not modified. `BasicBlock3D.forward` computes the skip projection first,
then main `ConvModule(3³) → ConvModule(3³)`, applies DropPath to the main branch,
adds the skip and applies ReLU.

For both Stage0s, input and output are `[B,128,192,192,16]` in VoxDet's
`[B,C,X,Y,Z]` layout. The first block in each Stage0 uses stride 1 but still
has a learned 3×3×3 128→128 skip ConvModule. The next blocks use identity skips.
The main branch's two ConvModules are hard-coded to BatchNorm3d in the vendor
block. `CustomResNet3D` passes BatchNorm3d to the Voxel Backbone skip and
GroupNorm(32) to the Local Backbone skip. This distinction is retained.

| Backbone | Stage0 blocks | First-block main | First-block skip | Later-block main | DropPath rates |
|---|---:|---|---|---|---|
| Voxel | 2 | 2× 3³ 128→128 + BN3d | 3³ 128→128 + BN3d | 2× 3³ 128→128 + BN3d | 0, 0.05 |
| Local | 3 | 2× 3³ 128→128 + BN3d | 3³ 128→128 + GN32 | 4× 3³ 128→128 + BN3d | 0, 0.0333, 0.0667 |

The high-resolution grid contains 589,824 spatial positions. One full-width
3×3×3 128→128 convolution costs 260.919 GMAC. Counting only the 3-D
convolutions, the five Stage0 blocks cost about 3.131 TMAC: 10 main-branch
convolutions plus 2 learned skip projections. A width-64 bottleneck is about
1.222 TMAC across those same blocks, a 61.0% theoretical reduction before
normalization, activation, gate, axial branches, memory traffic and kernel
launch overhead. This is an analytical count, not a speed claim.

At the start of V1, detailed per-layer CUDA profile is `NOT TESTED`. The
one-frame block and complete encoder smoke are recorded in
`experiments/omd3d_v1/gpu_profile/`; their tiny sample counts are explicitly
not used as formal latency results.
