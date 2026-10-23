# DeSeScene V-next D0: Support and FM Diagnostics

Date: 2026-10-23
Scope: zero-training diagnostics on the frozen R0 20K EMA checkpoint and an existing Stage2 FM checkpoint.

## Protocol

- Stage1 checkpoint: `experiments/dense_fidelity_v2_single_gpu/r0_gpu2_20k/checkpoints/exposure00020000.pt` (EMA, 20K frame exposures).
- Stage1 evaluation: all 900 frozen development records, Stage1 NFE=2, original FP16 autocast, fixed seed policy from the registered R0 config.
- Data manifest and per-frame input hashes are recorded in `experiments/desescene_v_next/d0_support_curves/r0_support_curve_900.json` and `r0_support_union_900.json`.
- GT is used only after candidate sets are formed, to calculate the Fine-GT Recall Ceiling. It never influences candidate selection.
- Fixed-K comparisons use deterministic score ranking with lower flattened Parent ID as the score-tie break.

## Stage1 score-ranked support ceiling

| Fixed Parent K | Mean Fine-GT Recall Ceiling | Macro Parent IoU | Macro Precision | Macro Recall |
|---:|---:|---:|---:|---:|
| 78,694 | 52.80% | 0.3805 | 0.7523 | 0.4482 |
| 84,963 | 54.95% | 0.3907 | 0.7338 | 0.4703 |
| 94,087 | 57.71% | 0.4016 | 0.7070 | 0.4995 |
| 100,000 | 59.30% | 0.4064 | 0.6900 | 0.5165 |
| 120,000 | 63.79% | 0.4136 | 0.6357 | 0.5658 |
| 150,000 | 68.72% | 0.4079 | 0.5654 | 0.6216 |
| 200,000 | 73.58% | 0.3748 | 0.4695 | 0.6782 |

The score ranking has recoverable recall headroom as K grows, but the parent IoU peaks near K=120K and then declines as false positives accumulate. A larger raw candidate set is not itself an acceptable deployment method.

## GT-free candidate-pool union diagnostic

Base K=84,963, alpha=1.5. The pool was formed as `TopK_1.5K(Stage1 score) ∪ radius-1 valid Parent neighborhood(TopK_K) ∪ observed input Parents`.

| Pool component | Mean candidate count | Mean Fine-GT Recall Ceiling |
|---|---:|---:|
| TopK K | 84,963 | 54.95% |
| TopK 1.5K | 127,445 | 65.18% |
| Neighborhood of TopK K | 273,751 | 80.38% |
| Observed input Parents | 12,560 | 13.25% |
| Full union | 278,008 | 80.64% |

The union adds 193,045 candidates over the base K and reaches 80.64% mean ceiling. This demonstrates that nearby support contains substantial GT coverage, but it costs 3.27× the base token budget. It does **not** prove a fixed-budget selector can identify those useful additions. A1 should proceed only as a small ranking-head study with a strict K budget and predicted-support validation; do not send the full union into PTv3.

## Existing Stage2 FM NFE diagnostic

The available 25K-update Stage2 FM checkpoint was evaluated on 16 development frames with **Oracle support**. It is not deployable and must not be interpreted as a predicted-support result.

| NFE | Macro Fine IoU | Macro F1 |
|---:|---:|---:|
| 1 | 0.6106 | 0.7580 |
| 2 | 0.5590 | 0.7169 |
| 4 | 0.5324 | 0.6946 |
| 8 | 0.5192 | 0.6832 |

Quality falls monotonically as NFE rises in this diagnostic. This flags an FM trajectory/conditioning issue; it does not identify whether the cause is the vector field, solver trajectory, or support/score shift. The next matched Stage2 pilots therefore use predicted parents and explicitly compare NFE 1/2/4/8 against a direct BCE control after their 1K-exposure learning check.

## Decision

1. Continue B1 first: same parent token budget and PTv3 trunk, with Child-level observed geometry statistics and a parameter-matched position-only control.
2. Run the short packed batch-2 engineering smoke before any long Stage2 run.
3. Do not launch A1 as a full 20K training run yet. The pool has high ceiling but is far over budget; only a fixed-budget reranker can justify further training.
4. Treat C1 as the next efficiency experiment after B1 learning/quality diagnostics. Its critical control is a same-support direct BCE head.

## Artifacts

- `experiments/desescene_v_next/d0_support_curves/r0_support_curve_900.json`
- `experiments/desescene_v_next/d0_support_curves/r0_support_union_900.json`
- Oracle NFE source: `worktrees/hs-occflow-384-stage2/experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes/validation/update_025000.json`
