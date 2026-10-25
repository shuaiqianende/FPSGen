# Registered next: independent Stage1 / Stage2 Shortcut Models

2026-10-25. The user selected **Shortcut Models**, explicitly conditioning on current time `t` and step size `d`, and requested the two stages as separate next experiments. This is not a MeanFlow JVP experiment or a Reflow endpoint-pair experiment. Reference: [One Step Diffusion via Shortcut Models](https://arxiv.org/abs/2410.12557).

| Arm | Preferred GPU | Trainable | Frozen | Formal budget |
|---|---:|---|---|---|
| S1-SC20K | 2 | Stage1 Flow + step embedding | Encoder, Stage2 |20K exposures /10K updates, true Batch2 |
| S2-SC20K | 3 | original Dynamic PTv3 Flow + step embedding | entire Stage1 |20K exposures /10K updates, true Batch2 |

Both use seed20261032, original raw surface-hit labels and the pinned mature source checkpoints. The current EF1/EF2 runs and their900-frame evaluation retain priority and must not be stopped, overwritten or restarted for this queue. Unvalidated adapted Encoder weights are not silently substituted.

## Method and controls

The model learns `u(x_t,t,d,c)`. Retain original instantaneous FM supervision at `d=0`; use a detached EMA target for the finite-step consistency relation between one `2d` step and two consecutive `d` steps. Register the precise loss weights, time sampler, dyadic step support and new embedding initialization before training. Do not interpret adding an inference CLI step size to an untrained ordinary FM as implementing Shortcut Models.

User clarification on2026-10-26: the Shortcut **step-count comparison** uses validation **sequence08 / skip100, exactly41 frames (0,100,...,4000)**, matching the original frozen frame manifest. Stage1 and Stage2 each compare NFE1/2/4/8 with the same trained checkpoint; no separate training per NFE is required. The Stage1 sweep keeps original Stage2 at NFE4; the Stage2 sweep keeps original Stage1 at NFE2 and the same actual predicted support. Preserve threshold0.5 and FP32 Euler integration.

Match raw input, GT, ROI, frame seeds and source noise for every step count and the original-FM control. When Stage1 changes support, shared absolute Parent IDs must receive identical Stage2 source noise (as in the existing seq08 combination protocol). Report native Fine IoU, CD, corrected physical IoU0.5/0.2/0.1, JSD-BEV/3D, counts, real generation wall latency and peak VRAM. Freeze checkpoint/protocol before this comparison; sequence08 is not used for training, bootstrap targets or repeated hyperparameter selection.

Each shortcut needs a matching original-FM20K continuation control. EF1 and EF2 are not Stage1 Flow-only controls because they adapt different modules. Audit existing B0 before reusing it as the Stage2 control. Reuse exact references when possible; otherwise the scientifically necessary matched control is a separate bounded20K run, explicitly registered before launch. Do not compare extra20K Shortcut training against an untouched checkpoint and attribute the entire difference to the method.

Repeated teacher forwards on one batch do not add new successful frame exposures. Report the extra teacher forward count, actual training time and memory: equal exposures need not mean equal compute. Memory handling must retain true Batch2 and all real scene Tokens. If both supervised and consistency terms are computed on the same Batch2, sequential weighted backward passes before one optimizer update are an engineering option; they must be shown equivalent to the registered combined objective.

## Launch state and dependencies

The machine-readable registration is [desescene_shortcut_queue.yaml](../../configs/research_v2/hier_occ_flow/desescene_shortcut_queue.yaml). State is **REGISTERED_PENDING_IMPLEMENTATION_AND_ENGINEERING**, not an already runnable automatic job. The running EF coordinator does not consume this file. Implementation and real GPU checks must be completed in an independent worktree before attaching executable jobs to continuation. Do not modify active EF training sources.

Required checks include `d=0` output comparison, valid `t,d` domain, detached bootstrap targets, gradients, frozen state, original Child order, largest real Batch2, optimizer/EMA/scaler/scheduler/RNG resume and a short learning check. Existing EF numerical replay exceptions do not automatically authorize exceptions for new Shortcut runs.

The existing900-frame Development evaluation remains the learning/primary-protocol quality check; it is **not the requested step-count comparison set**. This clarification does not modify running EF/control configurations. Shortcut step-count tables are built on the41-frame sequence08 set above, including matched original-FM references. Report same-NFE effects separately from reduced-NFE acceleration. Combine both stages only after independent validation. No extra training seed or100K training is queued.
