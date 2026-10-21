# Dense Fidelity V2 Blocks

The study keeps the original Dense 3D LiDAR path, 384×384×32 Stage1 grid,
three 128-channel condition features, Voxel Neck, TPV/Local Neck, and Flow U-Net.
Only Stage0 blocks in the Voxel and Local 3D backbones change. The pinned
VoxDet vendor source is untouched.

The original Stage0 feature tensor is `[B,128,192,192,16]` in `[B,C,X,Y,Z]`
layout. R0 has two Stage0 blocks in the Voxel Backbone and three in the Local
Backbone. The first block of each backbone has a learned 3×3×3 128→128 skip;
the remaining blocks use identity skips.

Three independent scratch variants are registered:

| ID | Main path | Skip path | Changed Stage0 blocks |
|---|---|---|---:|
| E1 `bottleneck96` | 128→96 1³, two 96→96 3³, 96→128 1³ | Original | 2 Voxel + 3 Local |
| E2 `asymmetric3d` | Original first 3³ 128→128; second 3³ represented as 128→64 1³, 64→64 3³, 64→128 1³, with no intermediate norm/activation | Original | 2 Voxel + 3 Local |
| E3 `lightweight_skip` | Original two full-width 3³ convolutions | First projected skip uses 1³ 128→128 with the original BN3d or GN32; later identity skips unchanged | First block of each backbone |

All variants retain the residual addition, DropPath rate, final ReLU, and
feature shapes. E2 retains the original first ConvModule and the original
second-convolution BatchNorm after the factorized spatial computation. E3
does not build the input observation context, since it has no observation
branch.

The 20K configs use the same seed, optimizer, exposure count, global batch,
data manifests, NFE, validation threshold, and fixed candidate budgets as the
V1 architecture pilot. They are separate scratch runs; none loads the V1 or
V6 encoder weights.

Current validation is limited to CPU shape/gradient tests and config-contract
tests. Real `[1/2,128,192,192,16]` GPU block timing, complete Encoder timing,
900-frame quality, and TensorRT FP16 export are `NOT RUN` because the current
runtime exposes no NVIDIA device nodes and the installed Stage1 environment
is missing `torch_scatter`.
