# Resolution and resource preflight

| Setting | Parent tokens P50 / sample max | Packed MB/frame | Est. train GB | Quantization P90 (cm) |
|---|---:|---:|---:|---:|
| 256_18 | 84,404 / 127,703 | 0.651 | 11.63 | 8.52 |
| 320_8 | 118,072 / 178,738 | 0.566 | 10.12 | 9.71 |
| 352_8 | 137,319 / 206,914 | 0.656 | 11.72 | 9.05 |
| 384_8 | 156,792 / 236,151 | 0.747 | 13.36 | 8.52 |

The provisional next configuration is256+18, with352+8 retained as an ablation.
Median parent tokens decrease38.54%; dense Stage1 cells decrease47.11% vs352.
Both settings have similar15cm physical-grid center representational IoU against
the **same raw map target**:0.69585 and0.69338. This is quantization/representation
analysis, **not trained model IoU**. 256+18 and384+8 share exactly the same fine
grid768x768x64; their fine geometry statistics agree, while parent token counts differ.
Geometry is **not locked for formal experiments** until real GPU measurements
and matched BCE/FM overfit/pilot tests pass. See [decision](hs_occflow_reports/geometry_decision.json).

CPU/NumPy/Torch indexing and world-space feature sampling pass tests. CUDA indexing
parity, complete Stage1 numerical forward shapes at352/384, PTv3 CUDA permutation,
and P99/max GPU forward/backward are **not executed**. All four candidates satisfy
divisibility by8. Static XYZ pyramids for352 are352/176/88/44 with Z32/16/8/4;
they are recorded as expected shapes, never as measured forwards.

Stage1's old VoxDet internals are XYZ. The independent condition adapter explicitly
converts feature volumes to ZYX and samples normalized world coordinates with
`align_corners=False`. Oracle352 parents may sample the frozen256 encoder in world
space; a256 density field is explicitly rejected as352 predicted parent occupancy.
The old U-Net uses shape-derived down/up sampling; no historical geometry checks
were relaxed. New-resolution convolution weight compatibility is not a quality result.

GPU2/3 already have compute PIDs1294615/1294616. Admission snapshots report free
memory6,784/1,420MiB. No HS-OccFlow CUDA network was launched. The old PointFlow
Student run remains active. System root had~3.5GiB free; all new data stays on/data-12.
See [GPU2 gate](hs_occflow_reports/gpu2_preflight.json), [GPU3 gate](hs_occflow_reports/gpu3_preflight.json).

## I/O measured on the actual sparse cohort

Warm-cache128 random reads,8 indexed train frames; real int64 ID/mask payloads
are transferred through DataLoader IPC. Pin memory/GPU waiting and concurrent-rank
throughput are unmeasured. No global page cache was flushed. LZ4/Zstd are unavailable
in the reused environment, so they are explicitly omitted, not simulated.

| Geometry | Format | Workers | Warm frames/s | CPU core equivalents |
|---|---|---:|---:|---:|
| 256_18 | npz | 0 | 138.5 | 1.01 |
| 256_18 | npz | 2 | 102.3 | 2.50 |
| 256_18 | npz | 4 | 196.5 | 4.45 |
| 256_18 | npz | 8 | 360.2 | 8.19 |
| 256_18 | mmap | 0 | 1228.6 | 0.96 |
| 256_18 | mmap | 2 | 534.5 | 3.09 |
| 256_18 | mmap | 4 | 495.7 | 3.06 |
| 256_18 | mmap | 8 | 521.1 | 3.34 |
| 352_8 | npz | 0 | 141.3 | 0.99 |
| 352_8 | npz | 2 | 111.1 | 2.80 |
| 352_8 | npz | 4 | 233.5 | 5.04 |
| 352_8 | npz | 8 | 344.7 | 8.21 |
| 352_8 | mmap | 0 | 1040.1 | 0.98 |
| 352_8 | mmap | 2 | 398.0 | 3.27 |
| 352_8 | mmap | 4 | 393.6 | 3.84 |
| 352_8 | mmap | 8 | 442.2 | 4.39 |

The cohort fits in page cache. This is not a full-dataset cold-cache benchmark.
The0-worker mmap path is currently fastest; workers/prefetch are not assumed to
improve it. Files/index sizes are compared on matching frame IDs in the JSON reports.
GPU end-to-end data stalls remain a required gate.
