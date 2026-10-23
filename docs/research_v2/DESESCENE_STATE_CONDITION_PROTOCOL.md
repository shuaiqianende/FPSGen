# DeSeScene State–Condition Interaction: protocol and implementation

This independent worktree implements the 2026-10-23 research specification. It preserves the active B1/C1 worktree and all previous artifacts. Branch: `research/desescene-state-condition-v1`; base: `b676da8`. The uncommitted B1/C1 implementation was copied as a read-only source snapshot; its hash and file identities are saved locally in `experiments/desescene_state_condition_v1/source_snapshot.json`.

## Evidence, rather than assumed completion

See [the generated evidence report](DESESCENE_STATE_CONDITION_EVIDENCE.md) and [CSV](DESESCENE_STATE_CONDITION_EVIDENCE.csv).

- B1 geometry and position-only arms both completed 10,000 updates / 20,000 exposures and all 900 development frames at NFE 1/2/4/8.
- Geometry input does not show a consistent benefit. NFE4 macro Fine IoU: 0.308658 versus position-only 0.309067. Both arms decline with increasing NFE, including under predicted support.
- The completed single-GPU Stage1 cohort is R0 0.420339, E3 0.420464, E1 0.425641, E2 0.422217 Parent IoU. E1's paired IoU improvement is +0.5302 percentage points, with a grouped 95% bootstrap interval [+0.3402,+0.6913]. This is one training seed; it does not include training-seed uncertainty.
- E1/E2 must not be rejected using an earlier 12-frame milestone or a different training cohort. Their final Fine-GT ceilings are 0.598639 and 0.598900, versus R0 0.590896.
- No exactly matched original Dynamic FM 20K arm was found. B1 changes the Child decoder; C1 compares Static FM with BCE.
- C1 is still running at the report snapshot. Its current weights are not substituted for a final checkpoint.

## Audited computation

Code references are relative to this worktree.

| Component | Current path | Tensor/data contract |
|---|---|---|
| Stage1 Encoder | `stage1_network.py:OccupancyConditionEncoder.forward` | Input XYZ, six geometric features, tokenizer, Voxel Backbone/Neck, Local Backbone, TPV classification neck |
| Pyramid | `stage1_network.py:134` | Three 128-channel B,C,Z,Y,X volumes; configured shapes [B,128,16,192,192], [B,128,8,96,96], [B,128,4,48,48] |
| Parent LiDAR condition | `condition.py:sample_pyramid` | Three trilinear samples at Parent centers, align_corners=False, border padding, concatenated [M,384] |
| Static embeddings | `model.py:33` | Position32 + score16 + LiDAR64 + observed32 = 144D |
| Dynamic embeddings | `model.py:39` | Original 8→64→64 state MLP, 32D scene-time embedding = 96D |
| Original early fusion | `model.py:44` | Original 240→64→64 MLP; one token per Parent |
| PTv3 | `ptv3_backbone.py:ParentPTv3` | Encoder depths2+2, widths64/128, decoder depth2 width64, original six time AdaLN-Zero blocks |
| Velocity head | `model.py` | Original [64+8+32]=104D input and eight Child outputs, same Child order |
| Integration | `inference.py:sample` | FP32 Gaussian source and FP32 Euler update; no GT in inference |
| Current C1 | `context_strategy=static` | Static PTv3 receives no dynamic state/time; FM decoder changes every NFE. BCE is direct one-pass |

The shapes above follow the configured strides and field conversion in source; the new queued GPU checks will record actual shapes and counts. Original training PTv3 explicitly uses FP32 projections/SpConv with FP16 FlashAttention internally. This study does not attribute the mature TensorRT model's latency to new models.

## M1 mechanism

`state_condition_interaction.py` adds:

- Static branch: original 144D static embeddings → 64D.
- Dynamic branch: original local-state64 and time32 → 64D.
- Interaction: `r = W_r(E_d(dynamic) * sigmoid(W_c(E_c(static))))`.
- Injection before `enc.enc0.block0` and `dec.dec0.block1`: `h += alpha_l * W_l(r)`.
- Both scalar alpha values start at zero; the projections are not simultaneously zeroed, so gates can receive a gradient.
- All original attention, CPE, MLP, time modulation, pooling, serialization and token ordering remain in place.

Parameter counts from actual CPU constructors: B0 2,677,128; M1 2,717,578; addition 40,450. Parent count does not increase.

The adapter uses per-model forward pre-hooks and a Point field, not global/vendor patches or mutable module state holding an activation. Hooks execute outside the original checkpointed blocks. Callable hook objects deepcopy correctly with EMA; a closure referencing the online model would be unsafe. Residual projection inputs explicitly follow the projection-weight dtype because the original PTv3 disables the outer FP16 autocast region.

## Matched cohort identity

| Setting | B0 | M1 |
|---|---|---|
| Model | Original Dynamic FM | Same + two residual injections |
| Frozen Stage1 | Same EMA checkpoint, SHA b60ab632…22e3d2 | Same |
| Candidates | Predicted, strict score>0.5, no expansion/truncation | Same |
| Stage1 NFE | 2 | 2 |
| Stage2 evaluation | Primary NFE4; diagnostics1/2/4/8 | Same |
| New exposures | 20,000 | 20,000 |
| Single-card batch / updates | 2 / 10,000 | 2 / 10,000 |
| Physical GPU | 2 | 3 |
| Seed | 20261035 | 20261035 |
| Loss | Original Gaussian-to-binary FM MSE | Same |
| Optimizer/schedule | Fresh AdamW, 2e-4, wd0.01, warmup500/cosine, clip1, EMA0.999 | Same |

C1 used Stage1 NFE4 and scratch initialization. The new NFE2 fine-tuning cohort is not directly pooled with C1 scores.

Both arms load **online weights from the same original-label Stage2 source**:
`hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes/checkpoints/step025000.pt`.
Its saved counter is **99,520 successful source exposures**, not exactly 100K. New experiments are 20K additional matched adaptation, not 20K total scratch training. Both restart optimizer, scheduler, scaler, EMA and successful counters. All 169 source tensors load strictly; only 18 new M1 tensor keys may be absent. Original checkpoint files are never modified.

Source identity validation checks geometry, Frozen Stage1 SHA, manifests, raw surface-hit target version, architecture and saved configuration SHA. Morphology-label checkpoints are rejected.

## Automated Phase0 and engineering queue

`run_desescene_phase0.py` waits for the existing C1 arms to finish at their registered counters. It never restarts or modifies them.

1. Each GPU waits for its existing C1 process to release the card, then runs a full 900-frame primary evaluation: Static FM NFE4 and BCE one pass, EMA, seed20261031, exact original NFE4 support.
2. Evaluation adds exact GPU KD-tree CD, corrected physical IoU@0.5/0.2/0.1m and shared FPSGen JSD against the corresponding raw-map scene. Native metrics include all missed-Parent Fine FN; support-only metrics are separate.
3. Per-case JSONL, input SHA, Parent-ID SHA, raw-GT SHA and the protocol survive independently of the final summary.
4. D1/D2 use fixed 64 development frames, three source seeds, NFE1/2/4/8, endpoint/probability ensembles, training-path velocity MSE and intermediate block responses. Global SceneNorm can propagate a local perturbation; cross-Parent response is not attributed to attention alone.
5. D3 uses the mature Frozen Stage1 at NFE2, K=84,963/120,000, a GT-free Top1.5K/six-neighbor/observed pool, and static score-neighborhood/observed ranking. Calibration is fixed by sequence/100-frame groups (hash modulo5); weights freeze before held-out evaluation. GT-only pool bounds are clearly offline, never selectors.
6. M1 GPU verification uses the two largest threshold-support scenes in that complete 900-frame NFE2 census. It checks zero-output/endpoint parity, sorting, cached path, B2 gradients, AMP and local optimizer/checkpoint replay.
7. B0/M1 each run a separate 32-update engineering smoke. No engineering weights are inherited into formal training.
8. D4 benchmarks the actual trained R0/E1/E2/E3 checkpoints with explicit FP16 on one physical GPU, 5+20 frames, three alternating rounds.
9. Results are summarized for a research GO/HOLD decision. **This queue does not launch new 20K training.**

All scheduled GPU tasks use idle-card admission, never shared-compute admission. B1/C1 timing collected during previous authorized sharing is not a performance baseline.

Reproduce the queued work:

```bash
cd /data-12/M2024-HWZ/FPSGen/worktrees/desescene-state-condition-v1
/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python \
  scripts/research_v2/hier_occ_flow/run_desescene_phase0.py \
  --source-worktree /data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun \
  --output experiments/desescene_state_condition_v1/phase0
```

## Conditions for formal B0/M1 training

- C1 and D1/D2 evidence reviewed together. Static FM versus BCE alone does not establish whether dynamic PTv3 is useful.
- Both 32-update smokes and real GPU checks pass on this exact source SHA.
- Register a machine-readable research decision `PROCEED_M1_MATCHED20K`, with rationale and hashes of actual reviewed evidence, using `register_desescene_m1_gate.py`.
- Training rejects a gate with a different source SHA, source checkpoint or missing C1 review.
- Use the full 10K-update horizon from the start: first stop at100 successful updates, then resume to500 (1K exposures), inspect learning health, then resume to10000. Optimizer, EMA, scaler, scheduler and RNG persist within that formal run.
- The 1K check only rejects numerical/engineering failures, not small-subset IoU fluctuations.
- Full900 evaluation and same-backend runtime precede any acceptance.

Pre-registered M1 GO: ≥1.0 pp full-GT native Fine IoU gain over B0; grouped paired95% interval lower bound>0; no meaningful CD/fine physical-IoU regression; Stage2 runtime increase≤10%; no B2/AMP failure. Report worst10% and worst individual deltas. <0.3 pp gain with added cost is NO-GO. 0.3–1.0 pp is uncertain and calls for a matched second seed, never automatic100K.

M3 ranker formal training requires at least2 pp held-out Fine-GT ceiling gain at fixed K. M2/M4 remain diagnostic/design-only. No new100K training is queued.

## Verification at this milestone

- CPU: 51 relevant tests passed; actual PTv3 constructors and strict source-weight compatibility were inspected without GPU forward.
- All169 baseline state tensors are equal after shared-source loading. M1's initial scalar gates are zero.
- GPU forward, numerical quality, stress/resume, learned gains and speed remain **NOT TESTED** until the idle-card queue runs.
- Complete checkpoints, raw predictions and caches remain server-local under ignored experiment paths.
