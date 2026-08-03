# Method-to-code guide

This page maps the notation in the
[FPSGen paper](https://arxiv.org/abs/2607.26645) to the released code.

## Pipeline

| Stage | Paper object | Main implementation |
| --- | --- | --- |
| BEV Flow | $B=[D,H,M]$ | `gen_img.FlowIMG` |
| Teacher Transport | $\mathcal{P}_0 \rightarrow \mathcal{P}_1^\dagger$ | `gen_teacher.DiffusionPoints` |
| Student Point Flow | $\mathcal{P}_0 \rightarrow \mathcal{P}_1^\dagger$ | `gen_student.DiffusionPoints` |

## BEV Flow

`BEVDataProcessor.points_to_bev_target` projects a complete scene
$\mathcal{P}^{gt}$ into $B_1=[D,H,M]$:

- $D$: log-normalized point density;
- $H$: normalized maximum height;
- $M$: binary occupancy.

`FlowIMG._shared_step` samples
$B_\tau=(1-\tau)B_0+\tau B_1$ and learns $v_\phi$ with
$u_B=B_1-B_0$.

## Teacher Transport

`bev_resample` implements the source sampler
$\mathcal{R}(\bar{B};N,\Sigma)$. It samples density-weighted BEV cells,
maps them to zero-height cell centers, and adds Gaussian XYZ noise.
`MinkUNet_NoTime` learns the source-indexed endpoint
$\mathcal{P}_1^\dagger$ with Chamfer reconstruction and local repulsion.

## Student Point Flow

The Student freezes the Teacher and learns the straight transport velocity:

$$
\mathcal{P}_t=(1-t)\mathcal{P}_0+t\mathcal{P}_1^\dagger,
\qquad
u_P=\mathcal{P}_1^\dagger-\mathcal{P}_0.
$$

`MinkUNetDiffIN` predicts this velocity conditioned on the generated BEV prior
and the active condition tuple.

## Conditions and coordinates

The condition tuple is $C_m=(m_lc_l,m_vc_v,m_rc_r)$ in
`[LiDAR, vehicle, road]` order. Every bit is either `1` (enabled) or `0`
(disabled).

- BEV tensors use `[batch, channel, x_index, y_index]`.
- Flattened cell indices use `x_index * grid_size + y_index`.
- A 256 x 256 BEV covers $[-50,50]\times[-50,50]$ metres.
- BEV cell coordinates use a `+0.5` center offset.
- PointFlow uses 180,000 source and output points by default.
