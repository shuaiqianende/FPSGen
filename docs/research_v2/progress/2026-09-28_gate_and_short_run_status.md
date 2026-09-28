# 2026-09-28 — Gate Results and Short-Run Status

## Reproducibility record

| Item | Value |
| --- | --- |
| Branch | `research/fpsgen-v2` |
| Starting commit for this update | `b43a536162c7eaba1ea814351a2a14ab399e9d28` |
| Environment | `M2024-HWZ-CasFusionNet` |
| PyTorch / CUDA | 1.13.0 / 11.7 |
| MinkowskiEngine / PyKeOps | 0.5.4 / 2.3 |
| GPUs | physical GPU2 and GPU3, RTX 3090 24 GB |
| Dataset split | SemanticKITTI sequence 08 for gates |

Machine-local raw artifacts are intentionally ignored by Git:

- `outputs/research_v2/feature_correspondence/a2_20/summary.json`
- `outputs/research_v2/sinkhorn_diagnostic/b100_best/{results.jsonl,summary.json}`
- `outputs/research_v2/sinkhorn_cache/b10/` and `sinkhorn_cache/b100/`
- `outputs/research_v2/sinkhorn_visuals/` (four PLY triplets)
- `experiments/research_v2_*/` (TensorBoard logs and checkpoints)

## Gate A — source-row feature correspondence

Status: **PROCEED**.

The synthetic diagnostic showed exact TensorField source-row recovery after
`sparse()` then `slice(field)`: maximum absolute recovery error was `0.0`.
The real sequence-08 A2 diagnostic used 20 equally spaced frames.

| Measurement | Teacher | Student |
| --- | ---: | ---: |
| Final decoder feature shape | 180000 × 48 | 180000 × 48 |
| Finite features | 20 / 20 frames | 20 / 20 frames |
| Collision ratio, mean | 0.1067% | 0.1031% |

The source rows that are singleton in both the teacher `P0` and student `Pt`
voxelizations have a mean ratio of **99.6455%** and a minimum of **99.3644%**.
Thus the first Feature Guidance experiment uses `all_rows`; the collision-aware
mask remains an ablation rather than the default.

Interpretation: `y4.F` is not safe for direct row-wise comparison, but
`y4.slice(x).F` is a stable source-row-aligned representation for the intended
weak regularizer.

## Gate B — Teacher-Guided Sparse Sinkhorn Refinement

Status: **PROCEED for OT-target short training**.

The method name deliberately remains *Teacher-Guided Sparse Sinkhorn
Refinement* / *Approximate Sparse OT Endpoint Refinement*.  It is not claimed
to be exact balanced OT: marginal errors are retained as diagnostics, not used
as a hard acceptance gate.

### Selected B100 configuration

| Parameter | Value |
| --- | ---: |
| Local candidates `K` | 16 |
| Sinkhorn iterations | 100 |
| Entropy temperature `epsilon` | 0.01 |
| Endpoint interpolation `alpha` | 1.0 |
| Endpoint definition | `P_ref = P_teacher + alpha * (P_OT - P_teacher)` |

The B10 parameter sweeps selected this configuration.  The B100 evaluation
then reused cached `(P0, P_teacher, GT)` tensors, so no Teacher forward pass
was repeated during parameter evaluation.

### B100 aggregate geometry

| Metric | Raw Teacher mean | Refined mean | Improved frames |
| --- | ---: | ---: | ---: |
| Chamfer ↓ | 0.13957 | 0.06976 | 100 / 100 |
| Endpoint → GT NN ↓ | 0.14487 | 0.07273 | 100 / 100 |
| GT → endpoint NN ↓ | 0.13427 | 0.06678 | 100 / 100 |
| F-score ↑ | 0.86005 | 0.98696 | 100 / 100 |
| Threshold coverage ↑ | 0.88539 | 0.99003 | 100 / 100 |
| NN target coverage ↑ | 0.63925 | 0.77668 | 100 / 100 |
| Mean NN spacing | 0.17832 | 0.13266 | — |

Refined spacing moved closer to the GT mean spacing on **98 / 100** frames.
The refined set also has a higher near-duplicate rate (for example NN distance
below 0.05 m: `0.00986 → 0.09696`); this is recorded as a caveat and must be
checked again after Student training rather than hidden by Chamfer alone.

The cached B100 refinement averaged **270.12 ms/frame** (median `252.48 ms`)
and used at most **515.26 MB** allocated GPU memory.  This is training-target
construction only: Student inference will not execute Teacher, GT, or
Sinkhorn.

Four representative B10 visualizations were saved locally, each containing
`teacher_endpoint.ply`, `gt.ply`, and `sinkhorn_refined.ply`:

```text
outputs/research_v2/sinkhorn_visuals/b10_k16_i100_eps001_a100/
  000000/
  001357/
  002713/
  004070/
```

They are intentionally not committed because PLY is an experiment artifact.

## Feature Guidance 500-step selection

All short runs were from scratch with the same Teacher, data split, optimizer,
learning rate, batch size, and seed policy.  Values below are the final
100-step moving mean, except feature cosine where applicable.

| Run | Guidance | FM loss ↓ | Feature loss | Feature cosine ↑ | Weighted feature / FM | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| F0 | disabled | 0.37596 | 0 | 1.00000* | 0 | baseline |
| F1 | cosine, `lambda=0.01` | 0.37703 | 0.59684 | 0.40316 | 0.01905 | not selected |
| F2 | cosine, `lambda=0.05` | **0.37203** | 0.52991 | **0.47009** | 0.08271 | selected |

\* The baseline logger uses its neutral disabled value; it is not a teacher-
student representation similarity result.

All three runs reached step 499 with finite losses and no OOM.  F2 is selected
because it reduces the FM-loss short-run mean versus F0 while providing a
moderate feature-loss contribution.  This is a **short-run stability and
selection result**, not yet evidence of improved generated point clouds.

## Running jobs

At the time of this update, the following independent 5k-step comparison was
launched from scratch:

| GPU | Run | Config | Purpose |
| --- | --- | --- | --- |
| GPU2 | F0-5k | `configs/research_v2/feature_f0_5k.yaml` | baseline reference |
| GPU3 | F2-5k | `configs/research_v2/feature_f2_5k.yaml` | selected Feature Guidance candidate |

Logs are machine-local under `outputs/research_v2/feature_train/logs/` and
TensorBoard artifacts under `experiments/`.  Completion requires the fixed
sequence-08 short probe with GT BEV and conditions `000`, `100`, and `111`.

## Next decisions

1. Finish the F0/F2 5k runs and evaluate the fixed PointFlow probe.  Promote
   Feature Guidance only if geometry improves consistently without a condition
   regression.
2. Implement the Sinkhorn-refined target in Student training, then run only
   O0 (raw Teacher target) versus O1 (the B100-selected target) for 500 steps.
3. Do not start Feature + OT combined training until Feature-only and
   OT-only short probes are individually evaluated.
