# Method-to-code guide

This guide maps the notation in
[FPSGen](https://arxiv.org/abs/2607.26645) to the released implementation.
It is intended to complement code-level docstrings rather than replace the
method description in the paper.

## 1. Flexible Condition BEV Flow Prior

The projection \(\Phi\) converts a complete target scene
\(\mathcal P^{gt}\) into \(B_1=[D,H,M]\):

- \(D\): log-normalized point density per cell;
- \(H\): normalized maximum point height per cell;
- \(M\): normalized binary occupancy.

`BEVDataProcessor.points_to_bev_target` implements \(\Phi\).
`FlowIMG._shared_step` samples
\(B_\tau=(1-\tau)B_0+\tau B_1\) and supervises
\(v_\phi\) with \(u_B=B_1-B_0\).

The active condition tuple is
\(C_m=(m_lc_l,m_vc_v,m_rc_r)\), where the three binary switches select
LiDAR, vehicle-layout, and road-layout cues. Training samples all eight tuples
uniformly and represents inactive cues with zero tensors.

## 2. Teacher Transport Mapping

`bev_resample` implements the training-time source sampler
\(\mathcal R(\bar B;N,\Sigma)\). It draws BEV cells from density-derived
weights, converts indices to cell-center coordinates, assigns zero height, and
adds independent Gaussian XYZ noise. The source is sampled independently from
the target point ordering.

`gen_teacher.DiffusionPoints` trains the historical network class
`MinkUNet_NoTime` as the Teacher Transport Mapping \(\Gamma_\eta\). The
resulting teacher pair is \((\mathcal P_0,\mathcal P_1^\dagger)\), optimized by
Chamfer reconstruction and local repulsion. The Lightning class name and the
network's internal residual sign are retained for compatibility with existing
checkpoints; their conversion to the paper's transport direction is documented
at the call site.

## 3. Approximate-OT Point Flow

`gen_student.DiffusionPoints` freezes the teacher and constructs the
source-indexed endpoint \(\mathcal P_1^\dagger\). It samples
\(\mathcal P_t=(1-t)\mathcal P_0+t\mathcal P_1^\dagger\) and trains
`MinkUNetDiffIN` to regress
\(u_P=\mathcal P_1^\dagger-\mathcal P_0\).

The term *Approximate-OT* refers to this learned, amortized source-indexed
coupling. It does not run an exact optimal-transport solver over the
180,000-point scenes.

## 4. Unified Inference

`DiffCompletion` in `fpsgen/inference.py` performs the full pipeline:

1. integrate the CFG BEV velocity from \(B_0\) to \(\hat B\);
2. sample \(\mathcal P_0=\mathcal R(\hat B;N,\Sigma)\);
3. integrate the CFG point velocity from \(\mathcal P_0\) to the generated
   scene.

Both ODEs use forward Euler integration from time zero to one.

## Tensor and coordinate conventions

- BEV tensors use `[batch, channel, x_index, y_index]`, not the conventional
  image ordering `[batch, channel, row=y, column=x]`.
- Flattened cell indices are
  `x_index * grid_size + y_index`.
- Integer BEV indices are mapped to metric cell centers with a `+0.5` offset.
- The released setup uses a \(256\times256\) grid over
  \([-50,50]\times[-50,50]\) metres.
- The PointFlow source and generated scene contain 180,000 points by default.
