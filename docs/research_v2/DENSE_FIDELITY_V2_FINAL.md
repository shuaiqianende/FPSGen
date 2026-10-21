# HS-OccFlow Dense Fidelity V2 — Live Research Status

**Status: E2 complete and promising; E3 training in progress.** This is an
interim report, not a final architecture selection. E2's trained quality is
close to R0 at the registered 20K exposure budget, while fixed-K evaluation
and trained-checkpoint deployment timing remain open gates.

## Candidate results

| Variant | Training | 900-frame threshold-0.5 IoU | F1 | Fine-GT recall ceiling | Encoder timing evidence | Assessment |
|---|---:|---:|---:|---:|---|---|
| R0 Original | registered 20K reference | 0.40021 | — | 0.57367 | random-init screen 177.38 ms; prior reference | Quality baseline |
| E1 Bottleneck-96 | 20K complete | 0.24872 | 0.39418 | 0.46979 | random-init screen 170.35 ms | Rejected on quality |
| E2 Asymmetric3D rank-64 | 20K complete | **0.39083** | **0.55787** | **0.55613** | random-init screen 157.11 ms | Promising; continue validation |
| E3 Lightweight Skip | 6,060/20,000 exposures at latest progress check | not run | not run | not run | random-init screen 165.53 ms | Training in progress |

E2 is 0.00938 absolute IoU below R0, approximately 2.34% relative. Its
Fine-GT recall ceiling is 0.01754 absolute below R0. The final E2 validation
measured 900 frames at threshold 0.5 with EMA and NFE=2. The registered
screening targets (IoU >= 0.395 and recall ceiling >= 0.563) are narrowly
missed, so E2 is not yet a passing replacement.

E2's stored validation also reports threshold sweeps, which are not fixed-K
results: IoU is 0.40613 at threshold 0.25 and 0.34228 at threshold 0.75.
Do not interpret those rows as the registered fixed candidate budgets. The
fixed-K evaluation at 78,694 / 84,963 / 94,087 is still pending.

The 157.11 ms E2 and 177.38 ms R0 Encoder numbers are short random-initialized
engineering screens, not trained-checkpoint inference benchmarks. They
suggest a compute reduction but do not establish E2's actual deployment
speed. PyTorch explicit-FP16 and TensorRT-FP16 timing on the trained E2 model
has not run.

## Training state

- E1 completed 5,000 updates / 20,000 successful frame exposures, followed by
  its 900-frame EMA evaluation.
- E2 completed 5,000 updates / 20,000 exposures. Its final checkpoint and
  `validation/update_005000.json` are preserved in the experiment directory.
- E3 is actively training. The latest observed log reached 1,515 successful
  updates / 6,060 exposures (30.3% of budget); recent update records show
  finite losses and gradients. Its 5K-exposure interim evaluation reported
  IoU 0.03496 and recall ceiling 0.16913, which is weak but is only the
  registered training-health checkpoint. Prior pilots showed large 5K
  instability, so this is not used as a candidate-selection result. The final
  decision remains the 20K evaluation.

GPU2/GPU3 are the registered training devices. This shell has not been able
to query NVIDIA management state, so this report makes no independent claim
about current utilization. Do not launch another job on those GPUs while E3
owns the training slot.

## Next gates

1. Finish E3's registered 20K training and EMA validation on the frozen 900
   frames.
2. Run fixed-K curves for E2 and E3 using the registered budgets.
3. Benchmark trained candidates on the same GPU and precision path; include
   PyTorch explicit FP16 and TensorRT FP16 where export is supported.
4. Select a candidate only after both quality and speed evidence are
   available. No 100K training or Stage2 retraining has started.

## Provenance

Results are stored under
`experiments/dense_fidelity_v2/training/`:

- E1: `e1_bottleneck96_20k_gpu23/summary.json` and
  `validation/update_005000.json`.
- E2: `e2_asymmetric3d_20k_gpu23/summary.json` and
  `validation/update_005000.json`.
- E3: `e3_lightweight_skip_20k_gpu23/train_rank0.jsonl`,
  `train_rank1.jsonl`, and `checkpoints/`.

Large checkpoints and per-update logs are not included in Git. The E2 final
quality values above are copied from its frozen JSON result; this report does
not rewrite the run artifact.
