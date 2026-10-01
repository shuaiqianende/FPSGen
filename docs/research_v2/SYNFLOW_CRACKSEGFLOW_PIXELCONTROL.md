# FPSGen Stage-1 spatial-control design

## Fixed contract

All short runs use FP32 eager, seed 42, effective batch 8, AdamW `1e-4`, the
existing warmup/cosine schedule, uniform `t`, and the unchanged weighted
velocity loss `[1,2,1]`.  `condition_keep` explicitly records `[L,V,R]` for
all eight uniformly sampled states.  New wrappers reapply that mask before
their first learned projection, so `000` cannot depend on arbitrary inactive
source tensors.

| ID | core | condition injection | purpose |
|---|---|---|---|
| U0 | shared SynFlow U-Net | multi-scale residual add | generic control baseline |
| S0 | same U-Net | Lite-SPADE encoder + middle + decoder | symmetric spatial control |
| C0 | same U-Net | Lite-SPADE decoder only | isolate placement |
| P0 | fixed local PixelDiT-S | generic concat34 patch/global add | paired transformer baseline |
| P1 | same PixelDiT-S | zero-projection residual after every Patch-DiT block | PixelControl-style control |

`boundary_gate=false` for all primary runs.  Boundary variants are intentionally
not scheduled until U0/S0/C0 evidence exists.

## Safety and diagnostics

Tests cover zero-state invariance, isolated single conditions, left/right
alignment, geometry-only layout boundaries, finite tiny forwards, and the
zero-projection/nonzero-gate PixelControl initialization.  The training module
uses separate seeded streams for condition state, time, and flow noise so the
paired runs consume matching stochastic draws.

`scripts/test_bev_forced_condition.py` is the explicit paired synthetic
diagnostic: same `x0`, `t=0`, left/right vehicle rectangles, and matching
left/right occupancy targets.  It accepts at most 300 updates and reports
whether the matched layout beats the swapped layout.

`scripts/eval_bev_spatial_control.py` evaluates full, occupied, completion,
and boundary FM errors and maps wrong conditions using frame `i -> i+17`
instead of a batch-one self-roll.  `scripts/eval_bev_condition_usage.py`
also now supplies the explicit state to new wrappers.
