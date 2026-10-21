# Dense Fidelity V2: R1/R2 Failure and Sensitivity Audit

## Scope and evidence

This audit reads the immutable V1 20K EMA checkpoints and their registered
900-frame curve artifacts. The checkpoint and curve SHA256 values are saved in
`dense_fidelity_v2_static_audit.json`. It does not infer unmeasured activation
behavior from final metrics. The ordinary sandbox shell lacks NVIDIA device
nodes, but host-level GPU2/3 execution is now available and validated. Dynamic
branch measurements and controlled activation perturbations remain pending;
GPU2/3 are allocated to the registered E1/E2/E3 training sequence.

## What the saved 900-frame results establish

At threshold 0.5, R1 predicts 124,802 parents per frame on average versus
99,670 for R0, yet its macro IoU falls from 0.40021 to 0.30716. Precision falls
from 0.68235 to 0.49196 (−19.04 points), while recall falls by only 4.59 points
(0.49866 to 0.45281). This is consistent with substantially more false
positives and weaker score discrimination, rather than a simple support-size
shortage.

The frozen fixed-K test confirms a ranking problem: at K=84,963, R0 has
precision/recall/IoU `0.70837 / 0.45366 / 0.37135`; R1 has
`0.58024 / 0.36900 / 0.28637`. Since both select exactly the same K, R1's lower
precision and recall cannot be explained by a larger prediction budget. Its
Fine-GT Recall Ceiling also falls from 0.52894 to 0.44728.

R2 predicts 156,517 parents on average, with precision 0.38622, recall 0.44379,
IoU 0.25643, and Fine-GT Recall Ceiling 0.51946. It is worse than R1 at every
registered fixed-K budget; at K=84,963 its IoU is 0.22953.

The saved artifacts contain threshold and fixed-K curves, not raw per-voxel
score tensors. Exact score histograms, threshold-margin counts, positive and
negative score quantiles, and spatial false-positive maps remain `NOT RUN`.
The registered fixed-K curves are the available evidence for ranking quality.

## Static R2 parameter check

The five learned `context_scale` values in the R2 EMA are:

| Block | `context_scale` |
|---|---:|
| Voxel Stage0.0 | 0.005598 |
| Voxel Stage0.1 | −0.001468 |
| Local Stage0.0 | −0.001353 |
| Local Stage0.1 | −0.004273 |
| Local Stage0.2 | 0.003544 |

They remain small in absolute value (maximum 0.00560 versus 0.001 initialization),
so the checkpoint does not support a claim that a large learned scalar gain
alone caused R2's regression. The three gate-bias logits per block correspond
to sigmoid values between 0.475 and 0.525. Biases alone do not show a strong
axis preference; input-dependent gate activations and local/context branch
norm ratios require a real forward pass.

The observed ordering—R2 below the already compressed R1—makes an interaction
between the width-64 bottleneck and added context plausible, but it does not
establish causality. A gamma-zero or context-branch ablation using the same
checkpoint and input/noise is still required to isolate that mechanism.

## Remaining sensitivity tests

`scripts/research_v2/hier_occ_flow/audit_dense_fidelity_dynamics.py` implements
the audit on a registered Development subset. For each of the two Voxel and
three Local Stage0 outputs, it adds fixed-seed zero-mean perturbations at 1%
and 5% of that activation's RMS, then records score errors, threshold-margin
flips, threshold metrics, fixed-K IoU at 78,694/84,963/94,087, and Fine-GT
Recall Ceiling. It also captures R2's local/axial/context activation norms and
gate averages and compares its EMA prediction with all learned context scales
set to zero under the same source noise. The run is queued until the current
GPU2/3 training sequence releases the cards. These remain sensitivity proxies;
they cannot predict a separately trained block.

The V1 stored PR/fixed-K artifacts are complete and comparable across all 900
frames (manifest SHA256 `6447076563b6c0efffbd3a07c5f695ebedf8aef1fa270a5cda9d99df73b30d53`).
R1/R2 runtime activation diagnostics are not included in that metric artifact.
