# FPSGen Stage-1 HDiT-S / DiP-S / NCSNpp-S preparation

## Scope and common contract

This preparation adds three dense pixel-space Stage-1 velocity backbones. It does not change the existing BEVFlow objective, BEV target, condition sampler, LiDAR frontend, optimizer, or inference API.

All three use direct linear Flow Matching on the native `3 x 256 x 256` BEV target `[density, maximum-height, occupancy]`:

\[
B_t=(1-t)B_0+tB_1, \qquad v_\theta(B_t,t,C)\approx B_1-B_0.
\]

The existing weighted velocity MSE remains `[1, 2, 1]`. The eight equally sampled `[LiDAR, vehicle, road]` condition states remain unchanged. `DynamicKNNPillarEncoder` remains the shared LiDAR frontend and produces `raw_pc: [B,32,256,256]`; the layout remains `[B,2,256,256]`.

New adapters receive their concatenation, preserve exact zero for an inactive all-zero condition, and use `gate_init=0.1` rather than zero. They expose the same wrapper API as the existing backbones: `get_raw_pc_bev(points)` and `forward(xt, t, raw_pc, layout_mask)`.

`gt_possion` is the data contract in every formal and smoke configuration. No VAE, latent representation, EDM objective, REPA, perceptual loss, or SDE score scaling is introduced.

## HDiT-S-BEV

Reference: `crowsonkb/k-diffusion`, commit `4601bf085320592473f681a62808ed873d17fad5`, `k_diffusion/models/image_transformer_v2.py`, MIT. The attribution manifest is [backbone_sources.yaml](backbone_sources.yaml).

The raw BEV enters through a `4x4` patch projection, yielding a `64x64` token grid. The retained hierarchy is:

| level | grid | width | blocks | attention |
| --- | ---: | ---: | ---: | --- |
| encoder 0 / decoder 0 | 64x64 | 128 | 2 / 2 | shifted-window, window 8 |
| encoder 1 / decoder 1 | 32x32 | 256 | 2 / 2 | shifted-window, window 8 |
| middle | 16x16 | 512 | 4 | global |

The core retains 2x2 `TokenMerge`, 2x2 `TokenSplit`, and learned lerp skip merges. FPSGen adaptation is restricted to continuous Fourier FM time, a bias-free `34 -> C0/C1/C2` adapter, stage-wise additive condition injection, and a bias-free global condition contribution to the mapping network.

## DiP-S/16-BEV

Reference: `NJU-PCALab/DiP`, commit `949294f290f2380a7767649f343679e24dab4410`, `src/models/transformer/dip.py` and `src/diffusion/flow_matching/training.py`. This is an architectural reimplementation; `DiP-S/16-BEV` is an FPSGen variant name, not an upstream published checkpoint name.

The global path retains patch-16 flattening (`16x16=256` tokens), RMSNorm, QK normalization, 2-D RoPE, scaled-dot-product attention, SwiGLU, Ada modulation, and eight `FlattenDiTBlock`s. It uses `hidden=384`, six groups/heads, hence head dimension 64.

The Local Detailer is retained rather than removed: every `16x16x3` patch follows the local U-Net path `16 -> 8 -> 4 -> 2 -> 1 -> 2 -> 4 -> 8 -> 16` with channels `[64,128,256,512]`; global patch tokens are injected at the bottleneck. The condition adapter produces bias-free patch-aligned tokens and a bias-free global summary.

## NCSNpp-S-BEV

Reference: `yang-song/score_sde_pytorch`, commit `cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44`, `models/ncsnpp.py` and `configs/ve/ffhq_256_ncsnpp_continuous.py`, Apache-2.0. The implementation retains the relevant NCSN++ hierarchy while replacing only the SDE score output with FPSGen direct velocity prediction.

The retained topology is seven spatial resolutions (`256,128,64,32,16,8,4` at production input), `ch_mult=[1,1,2,2,2,2,2]`, two BigGAN residual blocks per encoder level, DDPM attention at 16, FIR kernel `[1,3,3,1]`, `input_skip` progressive input and `output_skip` progressive output. The normal time interface receives `1000*t`; `scale_by_sigma` is absent.

The initial `nf=64` and then `nf=80` audits were below the 20M core-plus-adapter target. The selected smallest compliant setting is `nf=96`: core 22.81M, adapter 1.64M, total 24.45M.

## Configurations and compatibility

- `configs/research_v2/train_bev_hdit_s_gt_possion.yaml`
- `configs/research_v2/train_bev_dip_s_gt_possion.yaml`
- `configs/research_v2/train_bev_ncsnpp_s_gt_possion.yaml`
- corresponding `smoke_bev_*_gt_possion_b1.yaml` configurations

All new classes are selected by lazy branches in `bev_backbones/factory.py`. Historical checkpoints without `model.backbone` still select `legacy`; the old DiC-S and PixelU-S source files were not changed. The required GPU smoke evidence is recorded separately in [BEV_BACKBONE_HDIT_DIP_NCSNPP_SMOKE.md](BEV_BACKBONE_HDIT_DIP_NCSNPP_SMOKE.md).

## Deliberately deferred

No formal training, quality ranking, B20/B100 evaluation, AMP, Inductor benchmark, or checkpoint selection is implied by this preparation. Those are a later phase after the smoke records are reviewed.
