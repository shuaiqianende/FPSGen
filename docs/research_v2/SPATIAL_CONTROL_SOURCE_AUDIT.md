# Spatial-control source audit

This study holds the FPSGen Stage-1 target and the `DynamicKNNPillarEncoder`
fixed.  Each new method consumes the same explicitly masked 34-channel tensor
`[LiDAR(32), vehicle(1), road(1)]`; it does not introduce three modality
encoders.

## SynFlow-BEV-S

This is a source-based BEV adaptation of
[`BabakAsadi94/FMS2`](https://github.com/BabakAsadi94/FMS2), reviewed on
2026-10-01.  The relevant source layout is `FMS/synflow/train.py` and
`torchcfm/models/unet/unet_sdm_CrackSDM_v3_encoder.py`.  The repository
describes a SPADE-conditioned U-Net for SynFlow; FPSGen retains only the
spatial-normalization idea and the multi-resolution U-Net shape.

FPSGen changes are deliberate: raw 3-channel BEV velocity output, linear Flow
Matching, continuous `t*1000`, width-scaled channels `[64,64,128,128,256,256]`,
and no source dataset, CFG objective, or mask-generator component.  It is not
an official SynFlow implementation.

## CrackSegFlow-style-BEV-S

`CrackSegFlow` is treated as a paper-based adaptation only.  The checked
public material identifies a benchmark/paper but did not provide a verified
official generator implementation usable for a faithful reproduction.  This
repository therefore never labels this model “official CrackSegFlow”.

To isolate placement, it shares the exact `SynFlowCore` parameters and time
path with SynFlow-BEV-S.  The sole first-stage change is that Lite-SPADE is
enabled in decoder stages only; encoder and middle blocks are ordinary
timestep-conditioned residual blocks.  Optional vehicle/road morphology is
off for the primary comparison.

## PixelControl-BEV-S

This is a PixelControl-style adaptation based on
[`linxin0/PixelControl`](https://github.com/linxin0/PixelControl), reviewed on
2026-10-01.  The source uses frozen large PixelDiT models, independent
condition branches, structure-aware control, and auxiliary supervision.  None
of those training-recipe components are reproduced here.

FPSGen instead reuses its local PixelDiT-S core unchanged and adds one shared
`34ch -> nonlinear 16x16 patch-token` controller.  It inserts a zero-initialized
linear residual after every Patch-DiT block with nonzero gates.  There is no
frozen pretrained backbone, cycle loss, SAM2, DINO, or direct pixel condition
branch in P1.  P0 uses the same core and encoder but generic patch/global
addition, making the P0/P1 comparison paired.
