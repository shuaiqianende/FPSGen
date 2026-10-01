# Legacy BEVFlow training-speed audit

This audit is intentionally restricted to the historical `legacy`
`BEVFlowTransNet` contract: 3-channel 256² BEV, 180,000 points, K=16,
DynamicKNNPillarEncoder, the eight uniformly sampled condition states, and the
existing linear flow-matching objective.  No backbone, condition, data split,
target definition, or optimizer hyperparameter was changed.

## Environment and method

- GPU: one RTX 3090 (physical GPU2 or GPU3), batch 8.
- Software: PyTorch 2.0.1+cu117, CUDA 11.7, eager geometry/KeOps.
- Each steady-state microbenchmark uses 50 warm-up and 300 timed real batches.
- Fixed-real-batch results describe the GPU ceiling; end-to-end results also
  include DataLoader wait and host-to-device transfer.  These must not be
  conflated.

## Baseline profile

The FP32 fixed-batch profile was 410.4 ms/step on GPU2 (19.49 samples/s).

| region | ms/step | share |
|---|---:|---:|
| semantic layout | 0.85 | 0.2% |
| GT BEV target | 0.83 | 0.2% |
| KNN + pillar | 32.70 | 8.0% |
| dense forward | 89.61 | 21.8% |
| backward | 267.66 | 65.2% |
| optimizer | 14.07 | 3.4% |

Neither static GT/layout caching nor a KNN-index cache met the predeclared
10%-of-step gate.  They were deliberately **not built**: a two-flip FP32
target/layout cache would also require roughly 35 GB for 19,130 frames, while
an int32 `[180000,16]` KNN cache would be roughly 220 GB before metadata.

## Data path

The old `torch.isin` layout path was replaced, only for the default
SemanticKITTI vehicle/road class sets, with a cached device LUT.  It is exact
to the historical result, including out-of-range IDs.  Dataset conversion can
now opt into contiguous `torch.from_numpy` rather than copying through
`torch.tensor`.  Both are behind performance config options; historical
configs retain their defaults.

The isolated 300-batch loader sweep (batch 8, pinning/prefetch 4) showed:

| workers | mean batch latency | p99 | samples/s |
|---:|---:|---:|---:|
| 2 | 69.97 ms | 265.52 ms | 114.33 |
| 4 | 38.45 ms | 203.32 ms | 208.04 |
| 8 | 25.25 ms | 176.27 ms | 316.86 |

The end-to-end Legacy training probe nevertheless selected four workers: it
reduced data-ready wait from 28.62 ms in the current B0 settings to 1.16 ms;
eight workers did not improve end-to-end GPU cadence and increased its p99.
Pinned non-blocking transfer reduced the measured H2D portion from 7.68 ms to
1.22 ms.

## Dense compute sweep

| configuration | ms/step | samples/s | peak allocated |
|---|---:|---:|---:|
| FP32 eager, fixed | 410.38 | 19.49 | 9.35 GB |
| FP32 TF32 + cuDNN benchmark | 377.65 | 21.18 | 15.53 GB |
| FP16 eager (dense only) | 254.24 | 31.47 | 12.41 GB |
| FP32 dense Inductor | 356.69 | 22.43 | 15.48 GB |
| FP16 + dense Inductor | **212.34** | **37.68** | **11.63 GB** |

BF16 was not selected: the installed PyTorch 2.0 CUDA path raises
`upsample_nearest2d_out_frame not implemented for BFloat16`.  Geometry,
KeOps distances, target/layout rasterization and loss remain FP32 in FP16
mode; only dense convolution/attention uses autocast.

Inductor is restricted to the dense `xt,t,raw_pc,layout -> BEVFlowTransNet`
forward.  It never compiles Dataset logic, PyKeOps KNN, point indexing, or
scatter rasterization.  Its first real Lightning invocation took about one
minute to compile; that startup cost is excluded from the fixed-batch
steady-state table and is reported separately in the short-training result.

## Validation status

- CPU exact-equivalence tests for the LUT and zero-copy conversion: PASS.
- GPU B=8 16-step FP32 Legacy preflight: PASS (finite loss/gradients and
  active PointPillar/condition-FPN gradients).
- GPU B=8 16-step FP16 + dense-Inductor preflight: PASS.
- Paired 1,500-step training completed on GPU2 (FP32 eager) and GPU3 (FP16
  dense Inductor), with seed 42 and the same config-level batch, LR, data split
  and objective. The runs were concurrent, so their wall-clock speeds include
  shared CPU/DataLoader contention.

| measure | FP32 eager baseline | FP16 + dense Inductor |
|---|---:|---:|
| loss, steps 0–499 | 1.7722 | 1.7772 |
| loss, steps 500–999 | 1.0858 | 1.0781 |
| loss, steps 1000–1499 | 0.5076 | 0.4834 |
| final 200-step loss | 0.4144 | 0.3909 |
| mean of 1,500 logged losses | 1.1209 | 1.1120 |
| observed wall time | 13m31s | 12m30s |
| callback median synchronized step | 374.5 ms | 234.6 ms |
| callback median throughput | 21.36 samples/s | 34.11 samples/s |
| peak allocated memory | 11.1 GiB | 6.6 GiB |

The short-run losses track closely; the optimized run is lower late in the
run.  The 1500-step validation did not show a quality regression on the fixed
sequence-08 B20 CFG2 screen:

| metric | FP32 eager | FP16 + dense Inductor |
|---|---:|---:|
| LiDAR Gshuffle, t<0.2 (B100) | 0.1214 | 0.1319 |
| LiDAR Gshuffle, t<0.4 (B100) | 0.0996 | 0.1057 |
| all-condition Gshuffle, t<0.2 (B100) | 0.1416 | 0.1510 |
| occupancy IoU | 0.6656 | 0.6712 |
| occupancy F1 | 0.7976 | 0.8017 |
| completion F1 | 0.7341 | 0.7393 |
| height MAE on GT occupied cells (m) | 2.0903 | 2.0829 |
| density mass TV | 0.5619 | 0.5523 |

These B20 differences are descriptive; they are not a multi-seed quality
claim. The optimized run's end-to-end wall time improved by about 8%, while
the isolated fixed-batch GPU rate improved by about 77%. Data-ready gaps and
the first Inductor compile reduce the realized gain. The callback's
per-interval p95 gap includes occasional multi-second stalls, so the selected
four-worker policy should be re-profiled without concurrent jobs before a
long run.

## Work not run

- `torch.optim.AdamW(fused=True)` and `foreach=True` did not receive a
  dedicated comparison.
- A strict visualization-on/off Lightning pair was not run. The baseline
  used its historical every-100-step rendering; F0 disabled rendering. The
  measured wall-time difference therefore includes both compute and render
  effects.
- GPU2+3 DDP scaling remains unmeasured. The DDP config is prepared, and
  Legacy's active model has no MinkowskiBatchNorm, so conversion is skipped.
  The GPU launch/status operation was blocked by the platform's automatic
  approval usage limit; it must be run once GPU-command approval is available.

`formal 500-epoch training: NOT STARTED`.
