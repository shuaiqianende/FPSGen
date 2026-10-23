# DeSeScene experiment evidence audit

2026-10-23T04:38:47.607953+00:00

All values below are read from saved run artifacts. Existing runs and artifacts are unchanged.

| Run | Training status | Updates | Exposures | Evaluation |
|---|---|---:|---:|---|
| child_bridge_geometry_20k | COMPLETE_REQUESTED_BUDGET | 10000 | 20000 | 900-frame result present |
| child_bridge_position_20k | COMPLETE_REQUESTED_BUDGET | 10000 | 20000 | 900-frame result present |
| static_context_fm_20k | RUNNING_OR_INTERRUPTED_CHECK_LIVE_PID | 5128 | 10256 | NOT COMPLETE |
| static_context_bce_20k | RUNNING_OR_INTERRUPTED_CHECK_LIVE_PID | 5128 | 10256 | NOT COMPLETE |

| B1 variant | NFE | Fine IoU | F1 |
|---|---:|---:|---:|
| child_bridge_geometry_20k | 1 | 0.336641 | 0.498687 |
| child_bridge_geometry_20k | 2 | 0.321710 | 0.482677 |
| child_bridge_geometry_20k | 4 | 0.308658 | 0.467977 |
| child_bridge_geometry_20k | 8 | 0.301860 | 0.460207 |
| child_bridge_position_20k | 1 | 0.336982 | 0.499031 |
| child_bridge_position_20k | 2 | 0.323009 | 0.484056 |
| child_bridge_position_20k | 4 | 0.309067 | 0.468369 |
| child_bridge_position_20k | 8 | 0.301763 | 0.460027 |

B1 has no demonstrated geometry-input benefit on these 900 frames. Both B1 models show decreasing native IoU with increasing NFE. This does not distinguish mean estimation from integration/model errors.

| Encoder (single-GPU 20K cohort) | Parent IoU | F1 | Fine-GT ceiling |
|---|---:|---:|---:|
| r0_gpu2_20k | 0.420339 | 0.587249 | 0.590896 |
| e3_gpu3_20k | 0.420464 | 0.587482 | 0.590616 |
| e1_gpu3_20k | 0.425641 | 0.592649 | 0.598639 |
| e2_gpu2_20k | 0.422217 | 0.589628 | 0.598900 |

| Encoder minus R0 | IoU delta (pp) | Group bootstrap 95% CI (pp) | Worst frame delta (pp) |
|---|---:|---|---:|
| e1_gpu3_20k | +0.5302 | [+0.3402, +0.6913] | -1.6810 |
| e2_gpu2_20k | +0.1878 | [-0.2249, +0.5426] | -2.4168 |
| e3_gpu3_20k | +0.0125 | [-0.1847, +0.1965] | -1.6130 |

These are single-training-seed results. E1 recovered by the final milestone and exceeds R0 macro IoU in this cohort; it is not rejected based on the earlier 12-frame or different-cohort result. Trained same-backend latency, fixed-K ranking and full Stage2 compatibility are still unverified.

## Protocol limits

- C1 static FM and BCE: frozen mature Stage1 EMA, predicted support, Stage1 NFE4, strict score >0.5, batch2, scratch 20K. C1 final results are pending.
- No strictly matched original dynamic FM 20K control was found. B1 child-conditioned models cannot fill that role.
- Old mature dynamic FM uses Oracle support with 25% empty neighbors and constant score1. Its source weights may initialize a separately identified matched fine-tuning cohort; its old metrics are not a predicted-support control.
- Existing native reports do not contain CD, 0.5/0.2/0.1m physical IoU, or JSD. Those are NOT TESTED here.
- Saved evaluation stage times are diagnostic and were partly collected during authorized GPU sharing; they are not dedicated performance benchmarks.
- No new 100K training or M2/M3/M4 formal training is authorized by this audit.
