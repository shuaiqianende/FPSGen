# HS-OccFlow V5 Conv3d Kernel Profile

## Scope and protocol

Profiled the frozen Stage1 EMA and Stage2 Online checkpoints on physical GPU2 (RTX 3090), using the existing Stage1 FP16 autocast encoder path, batch 1, and real SemanticKITTI sequence 08 inputs. Checkpoints: Stage1 `b60ab632…e3d2`; Stage2 `4279a1f3…dfafe`. Three frames (2300, 700, 100; each 18,000 points) were profiled after a real-frame warmup. This is diagnostic profiling, not the formal latency benchmark.

Raw data: `experiments/hs_inference_opt_v5/conv3d_profile/gpu2_3frames.json`. Nsight Systems target trace and exported stats: `ncu_top_conv_nsys.nsys-rep`, `ncu_top_conv.csv`, and `gpu2_nsys_target.json` in the same directory.

## Findings

The 48 Conv3d modules account for 138.5–138.9 ms per encoder call across the three frames, versus 204.1–204.7 ms total encoder wall time. The layer-event ranking below is representative (frame medians vary by a few tenths of a millisecond). Layer inputs are reported in PyTorch N,C,D,H,W order; the large tensors have shape `1×128×192×192×16`. Most largest 3×3×3 layers process FP16 activations with FP32 weights under autocast; some neck and later-stage layers receive FP32.

| Rank | Module | Group | Kernel | Event ms |
|---:|---|---|---|---:|
| 1 | `local_backbone.layers.0.2.conv1.conv` | local backbone | 3³, 128→128 | 8.02 |
| 2 | `voxel_neck.fpn_convs.0.conv` | voxel neck | 3³, 128→128 | 8.02 |
| 3 | `local_neck.fpn_convs_cls.0.conv` | classification neck | 3³, 128→128 | 8.02 |
| 4 | `local_backbone.layers.0.0.conv1.conv` | local backbone | 3³, 128→128 | 7.98 |
| 5 | `local_backbone.layers.0.0.downsample.conv` | local backbone | 3³, 128→128 | 7.97 |
| 6 | `local_backbone.layers.0.1.conv1.conv` | local backbone | 3³, 128→128 | 7.97 |
| 7 | `voxel_backbone.layers.0.0.downsample.conv` | voxel backbone | 3³, 128→128 | 7.78 |
| 8 | `voxel_backbone.layers.0.0.conv1.conv` | voxel backbone | 3³, 128→128 | 7.59 |
| 9 | `voxel_backbone.layers.0.0.conv2.conv` | voxel backbone | 3³, 128→128 | 7.55 |
| 10 | `voxel_backbone.layers.0.1.conv1.conv` | voxel backbone | 3³, 128→128 | 7.51 |
| 11 | `local_backbone.layers.0.2.conv2.conv` | local backbone | 3³, 128→128 | 7.50 |
| 12 | `local_backbone.layers.0.1.conv2.conv` | local backbone | 3³, 128→128 | 7.46 |
| 13 | `local_backbone.layers.0.0.conv2.conv` | local backbone | 3³, 128→128 | 7.45 |
| 14 | `voxel_backbone.layers.0.1.conv2.conv` | voxel backbone | 3³, 128→128 | 7.44 |
| 15 | `voxel_neck.lateral_convs.0.conv` | voxel neck | 1³, 256→128 | 2.92 |
| 16 | `local_neck.lateral_convs_cls.0.conv` | classification neck | 1³, 256→128 | 2.92 |
| 17 | `local_backbone.layers.1.0.downsample.conv` | local backbone | 3³, stride 2 | 1.93 |
| 18 | `local_backbone.layers.1.0.conv1.conv` | local backbone | 3³, stride 2 | 1.93 |
| 19 | `local_neck.sp_tpvs_cls.0.fuse_conv` | classification neck | 1³ | 1.73 |
| 20 | `voxel_backbone.layers.1.0.conv1.conv` | voxel backbone | 3³, stride 2 | 1.40 |

Summed by group (per-layer medians): local backbone 66.07 ms (21 convolutions), voxel backbone 44.53 ms (15), local neck 15.70 ms (8), voxel neck 12.38 ms (4). The top 14 are large, full-resolution 3×3×3 convolutions; their combined time makes them the most credible optimization targets.

Nsight Systems on `local_backbone.layers.0.2.conv1.conv` shows cuDNN selecting an `sm86_xmma_fprop_implicit_gemm_f16f16_f16f32_f32...` kernel, consistent with FP16 Tensor Core math and FP32 accumulation. The selected main kernel took about 6.49 ms (about 83.5% of the layer's GPU time); NCHW↔NHWC transform kernels took about 0.75 ms combined, with additional copies around 0.53 ms. The input activation was logically NCDHW and not `channels_last_3d`. This identifies both a large Tensor-Core convolution and measurable layout-conversion overhead.

Nsight Compute was attempted, but hardware performance counters are unavailable to this user (`ERR_NVGPUCTRPERM`). Consequently, SM occupancy, Tensor Core utilization percentage, and DRAM bandwidth are **not measured**; the kernel name is evidence of the selected Tensor Core-capable kernel, not a utilization measurement. No system permissions were changed.

The Python environment is PyTorch 2.0.1+cu117. `torch.backends.cudnn.version()` reports a runtime mismatch: PyTorch was built against cuDNN 8.5.0 but resolves 8.4.0 through the current runtime library path. Inference ran, but this environment mismatch is a reproducibility caveat and was not altered during this experiment.

## Decision

Encoder total Conv3d time is the dominant measured category, but compilation of the dense submodules has not produced a robust end-to-end gain (see `HS_OCCFLOW_V5_ENCODER_COMPILE.md`). The large convolutions are Tensor-Core-capable already; a generic “enable FP16” change would not be new. The observed layout transforms motivate a correctly paired channels-last experiment, but prior V4 whole-encoder channels-last testing regressed substantially, so no layout change is accepted here. TensorRT was unavailable in this environment and remains untested.
