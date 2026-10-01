# NCSNpp-S generic condition vs symmetric Lite-SPADE

## Question and fixed contract

This study compares the existing NCSNpp-S BEV velocity backbone (N0) with an
NCSNpp-S spatially adaptive condition variant (N1).  It retains the Stage-1
linear Flow Matching target, 3-channel `[D,H,M]` BEV at 256x256, uniform
continuous time, `[1,2,1]` loss weights, DynamicKNNPillarEncoder, the same GT,
AdamW/LR schedule, FP32 eager execution, batch/effective batch eight, and all
eight explicit condition states.

`condition_keep=[L,V,R]` is applied before the map is built.  Both variants
therefore receive the same masked 34-channel source:

`[masked LiDAR(32), masked vehicle(1), masked road(1)]`.

## N0: unchanged current NCSNpp-S baseline

N0 uses the pre-existing NCSN condition encoder: a shared, bias-free condition
projection and pyramid from the 32+2 source.  Its existing generic route is a
resolution-matched residual spatial addition at the encoder locations.  Its
existing global route is retained unchanged: GAP of the condition feature,
projection, and the existing 0.1-initialized global gate into the time path.
The NCSN++ width, depth, attention, FIR, progressive output, ResBlock order,
and sampling/optimizer settings are not changed.

## N1: NCSNpp-S-SPADE

N1 has the exact same NCSN++ core, condition source, condition encoder,
condition pyramid, and global condition path as N0.  It disables N0's spatial
additive residual route.  At the two primary GroupNorm sites in every BigGAN
ResBlock in the encoder, middle, and decoder it instead uses the matching
resolution pyramid feature:

`SPADE(h,C) = (1 + Delta-gamma(C)) * GroupNorm(h) + beta(C)`.

The Lite-SPADE conditioner is bias-free `1x1 -> SiLU -> depthwise 3x3`, then
gamma/beta heads.  The two heads are zero initialized, so the modulation is
neutral at step zero.  Attention blocks, the NCSN++ time-embedding order, and
the global condition route remain untouched.  No boundary gate is enabled.

Parameter inspection for the released seed-42 configs gives:

| Variant | NCSN++ core | condition including SPADE | PointPillar | total |
|---|---:|---:|---:|---:|
| N0 | 22,808,949 | 1,642,184 | 1,600 | 24,452,733 |
| N1 | 22,808,949 | 1,928,545 | 1,600 | 24,739,094 |

N1 adds 286,368 parameters, or 1.17% of N0 total, below the 5% budget.

## Validation and execution

CPU tests cover exact neutral BigGAN and core equivalence, finite gradients
that open the zero-initialized condition path, and exact `000` invariance to
random/large source tensors.  Each queue additionally runs a 16-step actual
B8 preflight before its 1,500 optimizer-step run.

GPU2 runs N0 and GPU3 runs N1 for seed 42, then a barrier ensures their B100
condition-usage and B20 CFG1/CFG2 evaluations complete before paired seed 123
starts.  The same queue records actual trailing-500 loss at step 1500
(optimizer steps 1000--1499), trailing-200 loss, condition/SPADE diagnostics,
throughput, and memory.  This study explicitly does not start 500-epoch
training.
