# OMD-3D V1 GPU Microbenchmark

Hardware: NVIDIA RTX 3090, physical GPU2. Software: PyTorch 2.0.1+cu117,
CUDA 11.7, cuDNN 8.5.0. The run used the isolated VoxDet Python environment
with its bundled cuDNN libraries (`LD_LIBRARY_PATH` unset for that command;
no machine-wide environment was changed).

## Complete randomly initialized Encoder smoke and benchmark

Input: SemanticKITTI sequence 00 frame 000100, 18,000 LiDAR points (the exact
point count is in the JSON record). AMP FP16, batch 1, eval/inference mode,
20 warmup calls and 100 timed calls. Timings include CUDA event and wall-time
measurement of the Encoder call; model construction and input disk read are
outside the timed region.

| Variant | CUDA Event mean | Wall mean | Wall P95 | Peak allocated | Pyramid shapes |
|---|---:|---:|---:|---:|---|
| R0 original | 203.78 ms | 203.88 ms | 204.45 ms | 3.031 GiB | correct |
| R1 bottleneck | 160.14 ms | 160.21 ms | 160.86 ms | 3.018 GiB | correct |
| R2 OMD-3D | 195.16 ms | 195.22 ms | 196.25 ms | 3.018 GiB | correct |

Relative to this same-run R0, R1 is 21.4% faster at the Encoder; R2 is 4.2%
faster. These models are randomly initialized and this is only a speed test;
it says nothing about trained geometric quality. All three returned the exact
expected pyramid shapes. Their pyramid dtypes were FP32, FP16, FP16 under the
original AMP path.

## Stage0 forward/backward microbenchmark

Registered protocol: 20 warmups, 100 timed iterations, synchronized wall-time
per iteration, actual `[B,128,192,192,16]` tensor, forward+backward. AMP FP16
and explicit FP16 were both tested for batch 1 and 2. The full per-case output,
including standard deviation and memory, is in
`experiments/omd3d_v1/gpu_profile/stage0_forward_backward_gpu2.json`.

Selected Batch-1 AMP results:

| Block location | R0 | R1 | R2 |
|---|---:|---:|---:|
| Voxel first block, including learned skip projection | 74.18 ms | 51.30 ms | 73.02 ms |
| Voxel later block, identity skip | 56.35 ms | 32.69 ms | 54.58 ms |
| Local first block, GroupNorm skip projection | 79.06 ms | 55.44 ms | 77.46 ms |

At Batch 2 AMP, the same cases were 149.40/102.43/146.79 ms,
112.46/64.43/108.86 ms and 156.30/108.66/153.16 ms respectively. R1 yields a
clear compute reduction. R2's axial branch largely gives back the bottleneck
speed savings on RTX 3090, despite adding few theoretical MACs; this suggests
memory traffic and extra kernel launches matter. Explicit FP16 reduced peak
allocated memory but did not consistently improve time over AMP.

The observation context builder took about 4.7–6.3 ms in the warm cases. Its
first recorded case includes one-time Python/module initialization and is not
representative; a dedicated repeated context-only timing is still needed.

## Limitations

The block microbenchmark isolates one block and therefore overstates its share
of full-network impact. It uses a repeated real-scan observation mask and
random dense activations. The complete Encoder result is more representative,
but still uses random model weights and only one input frame. Training, a
multi-frame speed profile, Stage1 validation quality and full-generation
latency remain `NOT TESTED`.
