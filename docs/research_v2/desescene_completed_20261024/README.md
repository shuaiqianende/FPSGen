# Completed DeSeScene evidence, 2026-10-24

B0/M1 each completed20K additional matched adaptation; same raw-GT source, seed20261032, batch2, predicted Stage1 NFE2 support. C1 scratch/NFE4 metrics are a separate cohort.

| Stage2 | Native Fine IoU | CD m | IoU0.1m | JSD3D |
|---|---:|---:|---:|---:|
| b0 | 0.31745012 | 0.21709937 | 0.14841591 | 0.39742681 |
| m1 | 0.31740584 | 0.21721119 | 0.14839911 | 0.39747850 |

M1 has no demonstrated benefit and fails the registered+1pp native-IoU gate. Its paired95% interval includes zero. Preserve this negative result; do not expand M1 training.

| Trained Encoder | CUDA ms, three-round mean | Encoder wall including H2D ms |
|---|---:|---:|
| e1_gpu3_20k | 177.497 | 177.722 |
| e2_gpu2_20k | 161.737 | 162.004 |
| e3_gpu3_20k | 167.007 | 167.264 |
| r0_gpu2_20k | 179.621 | 180.054 |

All Encoder variants use physicalGPU3, explicitFP16, the same5+20 Development inputs and three alternating rounds. These are complete Encoder timings, not full-generation latency. E1 has the highest observed20K ParentIoU but little timing benefit; E2 has the largest PyTorch timing benefit. TensorRT, fixed-K and downstream checks remain necessary before deployment selection.

The original Encoder timing queue failed while loading inline geometry; the minimal loader fix and regression test are included in the same research branch. Original failed artifacts remain untouched. No weights, point clouds or compiled caches are exported. The sensitivity CSV and NFE tables are diagnostics, not final architecture acceptance.
