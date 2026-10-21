# OMD3D V1 Training Results and Progress

## Protocol

This pilot compares R0 Original, R1 Dense Bottleneck, and R2 OMD-3D under the same Stage1 flow-matching setup. Each arm starts from scratch, shares the registered Flow U-Net initialization, uses the same training manifest/order, seed, global batch of four (two RTX 3090 devices, local batch two), FP16 training, AdamW schedule, and EMA. The pilot budget is 20,000 successful frame exposures (5,000 successful updates) per arm. The original formal 100K-frame runner and its guard remain unchanged. No Stage2 training is part of this study.

The branch is `research/hs-occflow-dense-omd3d-v1`, based on `6cdca9bb5cff03631f0f622fe8e8ad02d49397e7`. Dataset manifests contain 17,870 training records and 900 development records. All three arms completed their full 20K budgets and 900-record EMA evaluations.

The three registered YAML configs were compared before continuing R1: R0 and R1 differ only by `model.encoder_variant`; R2 adds the `omd3d` encoder selector and its registered OMD block settings. Shared training, data, seed, batch, and evaluation fields match across the configs.

## Engineering smoke

Each variant completed 20 successful DDP updates / 80 global frame exposures with finite loss, nonzero Encoder and Flow gradients, synchronized rank hashes, and saved resumable checkpoints. R2's initial and update-20 checkpoints show parameter changes in all five blocks' axial X/Y/Z branches, observation gates, and context scales, confirming that these added components participate in training.

| Variant | Updates | Frame exposures | Initial FM loss | Final smoke FM loss | Status |
|---|---:|---:|---:|---:|---|
| R0 Original | 20 | 80 | 2.008 | 1.980 | PASS |
| R1 Bottleneck | 20 | 80 | 2.010 | 1.983 | PASS |
| R2 OMD-3D | 20 | 80 | 2.010 | 1.983 | PASS |

## 1K exposure check

Each arm completed 250 successful updates / 1,000 successful frame exposures. One GradScaler overflow occurred at the same attempted sample position in each arm; it was retried, did not count as a successful update, and advanced the recorded data cursor by one global batch. All three runs therefore preserve the same attempted sample order. Rank model hashes remained synchronized. The end-of-checkpoint validation used the registered 12-frame light subset, not the full 900-frame development set.

| Variant | Train seconds | Final logged FM loss | Light Val Precision | Recall | F1 | IoU | Active parents | Fine-GT recall ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 Original | 560.17 | 0.700 | 0.03683 | 0.48741 | 0.06811 | 0.03535 | 1,809,742 | 0.48786 |
| R1 Bottleneck | 397.79 | 0.648 | 0.03681 | 0.48645 | 0.06806 | 0.03532 | 1,807,810 | 0.48682 |
| R2 OMD-3D | 477.03 | 0.662 | 0.03648 | 0.48230 | 0.06746 | 0.03500 | 1,807,998 | 0.48241 |

The 1K results are only a training-health screen. They are too early and use too few validation frames to establish an architecture ranking or a quality conclusion. The higher speed of R1 at this stage is consistent with its lower Stage0 operation count, but official Encoder benchmarking will use trained weights after the full matched-budget runs.

## Formal 20K runs

| Variant | Successful exposures | Successful updates | Status |
|---|---:|---:|---|
| R0 Original | 20,000 / 20,000 | 5,000 / 5,000 | TRAINING + 900-record EMA validation COMPLETE; final logged FM loss 0.0804; one AMP overflow attempt skipped |
| R1 Bottleneck | 20,000 / 20,000 | 5,000 / 5,000 | TRAINING + 900-record EMA validation COMPLETE on physical GPUs 2/3; final FM loss 0.08080; one AMP overflow attempt skipped; checkpoint SHA256 `4cfaae1189f1fc29109ec05a1536031ba16b3c92a3173e386b6ab01774a5b755` |
| R2 OMD-3D | 7,208 / 20,000 latest observed | 1,802 / 5,000 latest observed | RUNNING on physical GPUs 2/3; 5K checkpoint and registered 900-record EMA validation complete; latest observed FM loss 0.12445; one AMP overflow attempt skipped |

R0 completed exactly 5,000 successful updates / 20,000 successful frame exposures on physical GPUs 2 and 3. The authoritative rank-0 training log ends with finite FM loss 0.0804; one AMP overflow attempt was skipped and excluded from the successful exposure count. The final checkpoints `exposure00020000.pt` and `step005000.pt` were written with synchronized ranks. The 900-record EMA evaluation completed at threshold 0.5 / NFE=2: macro Precision 0.68235, Recall 0.49867, F1 0.56790, IoU 0.40021, mean active parents 99,671, and Fine-GT recall ceiling 0.57367; micro Precision/Recall/F1/IoU are 0.68933/0.48067/0.56640/0.39508. The exposure-20K checkpoint SHA256 is `77ef88676eaaad0122947a4763c0e79dcab33064fd7ff2e1de73ee3d442cfe5f`. The exact 10,000-exposure checkpoint is saved as `exposure00010000.pt`, and its registered 900-record development evaluation completed. At threshold 0.5 / NFE=2 it reports macro Precision 0.55020, Recall 0.35225, F1 0.42156, IoU 0.26855, mean active parents 86,206, and Fine-GT recall ceiling 0.41529; micro Precision/Recall/F1/IoU are 0.55768/0.33634/0.41961/0.26551. Compared with 5K, macro Recall rose from 0.19182 to 0.35225, IoU from 0.04318 to 0.26855, and Fine-GT recall ceiling from 0.20489 to 0.41529, while active parents fell from 500,277 to 86,206. The exact same 12 light-check records appear in both full evaluations: between 5K and 10K their macro Recall changes from 0.19127 to 0.35373, IoU from 0.04175 to 0.26977, Fine-GT recall ceiling from 0.20487 to 0.41882, and active parents from 500,307 to 84,822. Thus the severe 5K support collapse partially recovered by 10K, while the 20K evaluation demonstrates further recovery. The registered threshold remains 0.5; no mid-run parameter changes were made. The registered exposure milestones are 2K, 5K, 10K, and 20K; the 5K, 10K, and 20K milestones invoke the full 900-record development evaluation. An additional offline evaluator, `scripts/research_v2/hier_occ_flow/eval_dense_encoder_curves.py`, computes exact strict-threshold PR curves and deterministic fixed-K metrics without changing the live training process. Fixed candidate budgets are frozen at 78,694 / 84,963 / 94,087 (R0 10K full-dev count quartiles) before cross-variant comparison; all three final checkpoints will use this same diagnostic. A separate EMA benchmark, `scripts/research_v2/hier_occ_flow/benchmark_trained_encoder_pilot.py`, is prepared for three 5-warmup/20-timed rounds per checkpoint at explicit FP16, with input hashes and both CUDA-event and H2D-inclusive wall timing; variant execution order will be alternated during the comparison. Detailed logs, checkpoint hashes, skipped-attempt records, and per-frame validation outputs remain under the ignored local `experiments/omd3d_v1/` tree and will not be committed.

R1 completed exactly 5,000 successful updates / 20,000 successful frame exposures. Its rank-0 log ends at finite FM loss `0.08080`; one AMP overflow attempt was skipped. The synchronized final EMA checkpoint SHA256 is `4cfaae1189f1fc29109ec05a1536031ba16b3c92a3173e386b6ab01774a5b755`. The full 900-row validation has unique `(sequence, frame)` IDs and used the registered threshold 0.5 / NFE=2. R1 macro Precision/Recall/F1/IoU are `0.49193 / 0.45282 / 0.46280 / 0.30715`, mean active parents `124,814`, and Fine-GT recall ceiling `0.53100`; micro Precision/Recall/F1/IoU are `0.49651 / 0.43355 / 0.46290 / 0.30115`. Against R0's 20K full-development results, R1 is lower by 9.31 percentage points in macro IoU and 4.27 points in recall ceiling, despite having about 25,143 more active parents on average. This establishes a matched-budget regression for the plain bottleneck variant; the held-out OMD-3D R2 remains necessary to test whether observation-modulated context recovers quality.

R2 reached its registered 2,000-exposure checkpoint at update 500. The synchronized EMA checkpoint `exposure00002000.pt` has SHA256 `c6386caafd638e1475d01bb80703629665ced5c71adb98289d8eb6874e9b2573`. Its 12-record light validation uses exactly the same `(sequence, frame)` set as R0/R1. At threshold 0.5 / NFE=2, R2 macro Precision/Recall/F1/IoU are `0.03671 / 0.41741 / 0.06706 / 0.03478`, mean active parents `1,554,708`, and Fine-GT recall ceiling `0.41844`. On this early light check, R2 is close to R1 in Recall (`−0.43 pp`), IoU (`−0.031 pp`), and Fine-GT recall ceiling (`−0.46 pp`); versus R0, its Recall and Fine-GT ceiling are lower by about `1.34 pp`, and IoU by `0.020 pp`. This is a training-health milestone only, not the final matched-budget conclusion. Training resumed after validation without changing the protocol.

At the registered 5K exposure milestone (update 1,250), R2 completed the full 900-record EMA validation at threshold 0.5 / NFE=2. Macro Precision/Recall/F1/IoU are `0.05057 / 0.14822 / 0.07401 / 0.03850`, mean active parents `414,714`, and Fine-GT recall ceiling `0.15743`; micro Precision/Recall/F1/IoU are `0.05058 / 0.14676 / 0.03909 / 0.03909`. The R2 5K checkpoint `exposure00005000.pt` has SHA256 `90ab9596a824f5972553a0e3a7269f978b1d79992c0d12d09129151132f6e716`. All three 5K validations contain the same 900 unique development `(sequence, frame)` records. Relative to R1 at the same budget, R2 gains `0.95 pp` macro Recall, `0.22 pp` macro IoU, and `1.04 pp` Fine-GT recall ceiling, with about 6,431 more active parents. Relative to R0, R2 remains lower by `4.36 pp` macro Recall, `0.47 pp` macro IoU, and `4.75 pp` Fine-GT recall ceiling, with about 85,563 fewer active parents. This is an interim result: the matched 10K and 20K evaluations remain necessary. R2 training resumed under the unchanged schedule; the latest live-session output observed update 1,802 / 7,208 exposures with finite FM loss `0.12445`.


The frozen 5K threshold diagnostics show R2 modestly ahead of R1 at all three registered cutoffs, while still behind R0:

| Variant | Threshold | Macro Recall | Macro IoU | Mean Active Parents | Fine-GT Recall Ceiling |
|---|---:|---:|---:|---:|---:|
| R0 Original | 0.25 | 0.53077 | 0.04305 | 1,675,518 | 0.54764 |
| R1 Bottleneck | 0.25 | 0.46169 | 0.04112 | 1,510,700 | 0.47625 |
| R2 OMD-3D | 0.25 | 0.47211 | 0.04215 | 1,510,141 | 0.48761 |
| R0 Original | 0.50 | 0.19182 | 0.04318 | 500,277 | 0.20489 |
| R1 Bottleneck | 0.50 | 0.13871 | 0.03626 | 408,283 | 0.14707 |
| R2 OMD-3D | 0.50 | 0.14822 | 0.03850 | 414,714 | 0.15743 |
| R0 Original | 0.75 | 0.04623 | 0.02697 | 99,404 | 0.05058 |
| R1 Bottleneck | 0.75 | 0.02579 | 0.01699 | 71,281 | 0.02770 |
| R2 OMD-3D | 0.75 | 0.02966 | 0.01919 | 75,982 | 0.03180 |

This is an interim threshold summary, not a replacement for the registered 101-point PR/fixed-K comparison on final checkpoints.

After R0, the restricted workspace shell could not access NVIDIA device nodes, so admission used the authorized elevated read-only `nvidia-smi` query. It reported GPUs 2/3 idle at 20/19 MiB, 0% utilization, with no compute processes. R1 was first attempted with the default base Python and failed before training because `torch_scatter` was missing; it made no updates. The runner was then restarted using `/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python` (PyTorch 2.0.1+cu117, `torch_scatter` 2.1.2, MMCV 1.7.2). The successful launch verified both manifests, initialized the model from scratch, and synchronized both ranks at update 0. The CPU-only contract/unit tests pass (13 passed); they do not substitute for GPU training or evaluation.

The frozen threshold sweep helps distinguish a score-calibration shift from loss of all geometric support. On the paired 12 frames, threshold 0.25 at 5K gives macro Recall 0.53024 and mean active parents 1,675,462, versus Recall 0.64471 and 2,392,920 at 2K for the same threshold; at threshold 0.5 the paired values change from Recall 0.43078 / 1,598,741 parents at 2K to 0.19127 / 500,307 at 5K. Across all 900 dev records at 5K, threshold 0.25 gives macro Precision 0.04494, Recall 0.53077, IoU 0.04305, and mean active parents 1,675,518; threshold 0.5 gives Precision 0.05383, Recall 0.19182, IoU 0.04318, and 500,277 parents; threshold 0.75 gives Precision 0.06472, Recall 0.04623, IoU 0.02697, and 99,404 parents. Thus the large fixed-0.5 Recall change coincides with a substantial prediction-score/support shift, while IoU is nearly unchanged between thresholds 0.25 and 0.5. This is diagnostic only: the registered threshold remains 0.5, and no threshold was retuned.

At the 2K milestone, R0's registered 12-frame light validation reported Precision `0.03682`, Recall `0.43078`, F1 `0.06743`, IoU `0.03498`, mean active parents `1,598,741`, and Fine-GT recall ceiling `0.43184`. This is below the earlier 1K light-screen recall/IoU (`0.48741` / `0.03535`). The concern was retained for the scheduled 5K, 10K, and 20K full-development checks; the fixed threshold and protocol were not changed.

R1 reached its registered 2K-exposure checkpoint at update 500 and completed the 12-frame EMA light validation. At threshold 0.5 / NFE=2 it reports macro Precision `0.03702`, Recall `0.42173`, F1 `0.06764`, IoU `0.03509`, mean active parents `1,557,534`, and Fine-GT recall ceiling `0.42304`. These light-screen values are close to R0's 2K values and are not a full-development comparison; R1 continues to the registered 5K/10K/20K checks.

At the registered 5K exposure milestone (update 1,250), R1 completed the full 900-record EMA evaluation. At threshold 0.5 / NFE=2, macro Precision/Recall/F1/IoU are `0.04792 / 0.13871 / 0.06989 / 0.03626`, mean active parents are `408,283`, and Fine-GT recall ceiling is `0.14707`. Against R0's exact 5K evaluation on the same development manifest, R1 is currently lower by 5.31 percentage points of Recall, 0.69 points of IoU, and 5.78 points of Fine-GT recall ceiling; its mean active-parent count is lower by about 92,000. This is an interim matched-budget result, not a final architecture rejection; R1 training continues unchanged to 20K.

At the registered 10K exposure milestone (update 2,500), R1 completed the full 900-record EMA evaluation and saved `exposure00010000.pt` (SHA256 `2dd816dda3abef8f9f8a16b46cff05b0631d830fbd5b6e633202a206387f20af`). The R0 and R1 evaluation files contain the same 900 `(sequence, frame)` IDs. At threshold 0.5 / NFE=2, R1 macro Precision/Recall/F1/IoU are `0.60193 / 0.19496 / 0.28974 / 0.17040`, mean active parents `43,473`, and Fine-GT recall ceiling `0.24244`. Relative to R0's 10K evaluation, R1 has `+5.17 pp` Precision but `−15.73 pp` Recall, `−13.18 pp` F1, `−9.82 pp` IoU, and `−17.29 pp` Fine-GT recall ceiling; it predicts about 42,734 fewer active parents on average. This is a substantial interim regression at the frozen threshold. The registered protocol still requires the full 20K matched budget, so R1 continues unchanged; this result is not hidden or used to tune the threshold.

The same saved validation rows also include frozen diagnostic thresholds. At threshold 0.25, R1 macro Recall is `0.35269` and IoU `0.22537`, versus R0 `0.51294` / `0.28887`; the lower R1 recall therefore is not explained only by the fixed 0.5 cutoff. This is an interim threshold diagnostic from the registered validation, not a threshold-selection exercise; final cross-arm comparisons will use the pre-registered offline PR and fixed-K analysis.

## Final matched-budget results

All three variants completed the registered 20K training and 900-frame EMA validation. The R2 20K checkpoint is `exposure00020000.pt` (SHA256 `915ec529dd5fc49cbe202304d63d300c23d5d0c556211ba269fa262aa52ee051`). At threshold 0.5 / NFE=2, R2 reports macro Precision/Recall/F1/IoU `0.38623 / 0.44377 / 0.40309 / 0.25643`, mean active parents `156,503`, and Fine-GT recall ceiling `0.51945`; micro Precision/Recall/F1/IoU are `0.38711 / 0.42384 / 0.40464 / 0.25364`. All 900 frame IDs are unique, and the evaluation geometry hash matches the registered geometry. The paired-curve aggregator verified all three curve outputs against the same 900-frame manifest. No 100K run, OMD mechanism ablation, Stage2 adaptation, or full-system quality claim was part of this pilot.

## Matched 20K PR and fixed-candidate evaluation

The registered evaluator processed the same 900 development frames for each EMA checkpoint with 101 score thresholds and fixed candidate budgets of 78,694 / 84,963 / 94,087. All three outputs report the same development-manifest SHA256 (`6447076563b6c0efffbd3a07c5f695ebedf8aef1fa270a5cda9d99df73b30d53`) and 900 paired frame IDs. The paired JSON is `experiments/omd3d_v1/curves/paired_comparison_20k.json`.

| Variant | Threshold .5: Macro P / R / F1 / IoU | Active parents | Fixed-K IoU at 78,694 / 84,963 / 94,087 |
|---|---:|---:|---:|
| R0 Original | 0.68235 / 0.49866 / 0.56789 / 0.40021 | 99,670 | 0.36086 / 0.37135 / 0.38298 |
| R1 Bottleneck | 0.49196 / 0.45281 / 0.46281 / 0.30716 | 124,802 | 0.27914 / 0.28637 / 0.29391 |
| R2 OMD-3D | 0.38622 / 0.44379 / 0.40309 / 0.25643 | 156,517 | 0.22230 / 0.22953 / 0.23781 |

At matched candidate budgets, R2 remains below R1 and R0, so this pilot does not demonstrate an occupancy-quality improvement. Full threshold curves and per-frame paired deltas remain in the machine-readable JSON; the registered threshold was not retuned.

## Trained EMA Encoder timing

Each 20K EMA encoder was measured on physical RTX 3090 GPU2 using explicit FP16, three rounds of five real-frame warmups and 20 timed frames per round (60 timed samples per variant). The same environment and selected 25 development records were used for all variants. This isolates the trained Stage1 Encoder and is not full Stage1+Stage2 generation latency.

| Variant | CUDA Event mean ± std (ms) | Median / P95 (ms) | H2D + encoder wall mean (ms) | Peak allocated (GiB) |
|---|---:|---:|---:|---:|
| R0 Original | 178.26 ± 0.42 | 178.15 / 178.92 | 178.43 | 1.940 |
| R1 Bottleneck | 138.37 ± 4.45 | 137.74 / 138.64 | 138.56 | 1.940 |
| R2 OMD-3D | 173.58 ± 0.37 | 173.54 / 174.25 | 173.80 | 1.940 |

R1 is about 22.4% faster than R0 in this Encoder-only test, while its matched-budget IoU is lower by roughly 8.5 percentage points at K=84,963. R2 is about 2.6% faster than R0, a small change relative to cross-variant uncertainty, and its fixed-K quality is worse. These encoder timings do not establish a full-system speedup. The JSON files in `experiments/omd3d_v1/encoder_benchmark/` contain all per-frame samples, warmup records, precision audits, GPU metadata, and percentiles.

The host's cuDNN runtime-version query reports a mismatch (PyTorch build 8.5.0, discovered runtime 8.4.0). The encoder benchmark completed, and the error is recorded in each result JSON instead of discarding its timing data. This shared environment limitation should be resolved before treating small timing differences as deployment guarantees.
