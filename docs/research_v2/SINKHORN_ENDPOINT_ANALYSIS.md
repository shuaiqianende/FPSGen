# Sparse Sinkhorn endpoint Gate 2

Status: implementation complete; no real Teacher endpoint has been evaluated
in the current sandbox because its runtime lacks CUDA, MinkowskiEngine, and
PyKeOps. Therefore this gate is **PENDING**, not `PROCEED` or `STOP`.

`fpsgen.ops.endpoint_refinement` builds no dense `N x M` cost tensor. It uses
KNN edges from Teacher endpoint to GT and reciprocal GT-to-endpoint KNN edges.
The reciprocal edges ensure every target has support, which is required for a
balanced OT marginal. Duplicate edges are coalesced by minimum squared cost;
log-domain Sinkhorn then operates only on at most `2K(N+M)` candidate edges.

The FPSGen interpolation is intentionally endpoint-relative:

\[
P^\dagger_{ref}=P^\dagger_T+\alpha(\bar P_{OT}-P^\dagger_T).
\]

Thus `alpha=0` is exactly the raw Teacher endpoint. Cheap controls are
nearest-GT and unbalanced KNN barycentric projection.

Run 10 fixed frames first (replace paths with the protocol's exact subset):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_teacher_endpoint.py \
  --teacher-ckpt checkpoints/teacher_gen_stg1_bev_epoch=04.ckpt \
  --frames /data-12/M2024-HWZ/KITTI_Odometry/08/gt_/000042.npy \
           /data-12/M2024-HWZ/KITTI_Odometry/08/gt_/000100.npy \
  --k 16 --alpha 0.25 0.5 0.75 1.0 --epsilon 0.05 --iterations 100 \
  --knn-backend keops
```

The CSV is `outputs/research_v2/sinkhorn_diagnostic/results.csv` and includes
Chamfer, directed NN distances, F-score, spacing, transport length, runtime,
GPU peak memory, marginal errors, entropy, edges, seed, and commit.

## Gate decision

Set **PROCEED** only if sparse Sinkhorn improves geometric metrics stably over
raw Teacher *and* the nearest/KNN controls, with acceptable runtime and small
row/column marginal error. If it ties cheap projection or degrades coverage,
set **STOP** and do not train a Student on refined endpoints.
