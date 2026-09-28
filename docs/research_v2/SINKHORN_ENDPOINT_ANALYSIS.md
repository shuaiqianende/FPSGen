# Sparse Sinkhorn endpoint Gate 2

Status: Gate B10 completed on GPU3; Gate B is on hold pending a valid sparse
balanced-OT convergence fix.

## Gate B10 result (2026-09-28)

GPU3 ran 10 equally spaced sequence-08 frames. The local sparse-OT geometry
signal is strong: at alpha 1.0, mean Chamfer changed from 0.13811 (raw
Teacher) to 0.09717, mean F-score from 0.86571 to 0.96761, and both directed
NN distances improved on every frame. NN-target coverage increased on 9/10
frames. However, this is **not a valid balanced-OT result yet**: after 500
iterations, mean row-marginal error remained about 4.62e-7 versus the target
row mass 5.56e-6, with 88.3% of rows exceeding 1% relative error. Column
errors were near zero, indicating disconnected/poorly communicating local
support rather than adequate balanced convergence.

The original strict-balanced-OT gate is retained only as a diagnostic record.
The current research protocol selects refinement parameters by endpoint
quality, spacing diagnostics, and runtime; it does not claim exact balanced
OT. Results are retained under `outputs/research_v2/sinkhorn_diagnostic/b10/`
and `b10_i500/`.

## Selected B10 refinement parameters (2026-09-28)

Under the updated research objective of endpoint quality versus cost (rather
than exact balanced-marginal convergence), the first B10 parameter ablation
selects:

```text
K = 16
Sinkhorn iterations = 100
epsilon = 0.01
alpha = 1.0
```

On the fixed cached B10 set this configuration reached mean Chamfer `0.06862`.
It was materially better than epsilon 0.02/0.05/0.10/0.20 at the same 100
iterations, while 200/500 iterations at epsilon 0.05 had weaker quality/cost
trade-offs for this initial screen. Alpha 1.0 also outperformed 0.5 and 0.75
under the selected epsilon. Marginal errors remain diagnostic-only; this
method is named **Teacher-Guided Sparse Sinkhorn Refinement**, not exact
balanced OT.

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

For the controlled 10-frame Gate B run, use `scripts/run_sinkhorn_gate.py`
with `configs/research_v2/gate_seq08_10.txt`. It loads the Teacher once,
writes `b10/results.csv` and `b10/summary.json`, and adds NN-target coverage
to expose many-to-one collapse. It never starts B100 automatically.

## Gate decision

Set **PROCEED** only if sparse Sinkhorn improves geometric metrics stably over
raw Teacher *and* the nearest/KNN controls, with acceptable runtime and small
row/column marginal error. If it ties cheap projection or degrades coverage,
set **STOP** and do not train a Student on refined endpoints.
