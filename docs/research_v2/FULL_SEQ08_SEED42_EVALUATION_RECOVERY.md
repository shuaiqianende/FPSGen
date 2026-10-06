# Full seq08 seed42 evaluation: numerical failure and recovery

## Protocol

Physical GPU3 only; all 4,071 available sequence08 input/GT pairs,
`gt_possion`, LiDAR-only `100`, literal zero vehicle/road maps,
Euler10, exact Gaussian seed42 on **every** frame. Dense inference remains
FP16; CFG2 used Inductor, while the documented CFG1 recovery uses eager for
all three models. Geometry, Euler state and metrics remain FP32.
All checkpoints completed 500 epochs / global step299000.

Artifacts: `outputs/research_v2/formal_bev_compare/full_seq08_seed42_gpu3_20261006/`.

## Initial outcome (2026-10-06)

All three CFG2 full-sequence evaluations completed and validated. Subsequently,
FPSGen CFG1 raised `FloatingPointError: Non-finite BEV generated during Euler
sampling` before its first 100-frame progress message. The queue stopped;
it did not skip a frame or silently substitute another precision, seed,
checkpoint, guidance scale, or backend. No original failed-frame identity was
logged, so the exact failing frame is unknown.

| Method | CFG | Frames | IoU | Occ F1 | Completion F1 | Height MAE (m) | Mass-TV |
|---|---:|---:|---:|---:|---:|---:|---:|
| FPSGen | 2 | 4071 | 0.785946 | 0.878180 | 0.838093 | 1.048968 | 0.329436 |
| SiD2-style | 2 | 4071 | 0.776974 | 0.872550 | 0.829988 | 1.104831 | 0.335235 |
| DiP-S/16 | 2 | 4071 | 0.787033 | 0.879102 | 0.839896 | 1.044345 | 0.328739 |

These are completed CFG2 results, not a claim that all CFG1 evaluations have
completed. Small mean differences alone are not statistical or across-seed
evidence. DiP training used DDP2×16 for the first50 epochs then GPU3×32 for
the remaining450; global batch32 remained unchanged, but data ordering and
private RNG streams changed on resume.

## Diagnosis and safeguards

`scripts/evaluation/diagnose_bev_sampling_nonfinite.py` independently replayed
the first100 frames with seed42, CFG1 and FP16 Inductor. All100 were finite.
This rules out an invariably failing first100-frame path in that diagnostic
process; it **does not** establish the root cause or prove the original
measurement correct.

The original evaluator is being retried using
`scripts/evaluation/eval_bevflow_checked.py`. This wrapper leaves successful
sampling math unchanged (retaining step tensors for post-failure inspection).
On a numerical failure it records frame identity,
condition statistics and forensic per-Euler-step FP16 Inductor / FP16 eager /
FP32 eager replays before re-raising the error. Diagnostic replays never
replace the failed sample or contribute headline metrics.

The queue now supports `--resume`: validate and reuse completed stages,
preserve incomplete outputs and logs in `failed_attempts/`, and rerun only
unfinished stages under the same protocol. The previous queue status and
campaign manifest are preserved. An invalid completed result or a protocol
mismatch stops resume rather than overwriting it.

Original-program retries failed on frame000002 and frame000003. Captured
frame000003 had finite `xt`, `t=0`, and raw LiDAR condition, but the first
network output already contained196608 nonfinite elements. Thus this failure
starts in the first forward, not after cumulative Euler divergence. Same-input
FP16 Inductor, FP16 eager and FP32 eager diagnostic replays were all finite;
diagnostic synchronization/allocation can mask the fault.

A separate minimal GPU probe confirmed the installed Inductor `baddbmm`
decomposition fails to ignore NaN additive input when beta=0. A process-local
correction made that minimal probe exact to eager (max error0), **but did not
resolve the real evaluator's frame000003 failure**. It is therefore a rejected
root-cause/fix hypothesis for this incident and is NOT enabled in the current
queue. No PyTorch installation or model source was modified.

## Recovery execution

FP16 eager CFG1 recovery launched in tmux `fpsgen_full_seq08_seed42_eager_gpu3`.
Consult `queue_status.json` and the current stage log for actual progress.
`cfg1_backend_exception.json` documents the backend change. All three CFG1
methods use the same eager backend; completed CFG2 results remain Inductor.
An eager failure still stops the queue; there is no automatic retry/precision
fallback or frame skipping. Exact low-level cause remains unresolved, and
a successful eager recovery is a mitigation, not proof of that cause.

The original-program FP16 eager retry passed its first100 frames (25.9s)
without a nonfinite sample, including both previously captured failing frames.
This is an initial smoke result, not a full-sequence pass. The queue remains
running and will generate the complete comparison only after all CFG1 stages
also validate all4071 frames.

## Final outcome

The eager recovery completed successfully on2026-10-06 at07:01:54 UTC
(15:01:54 Asia/Shanghai). All three CFG1 stages validated4071 frames, exact
seed42 and finite metrics. The three preserved CFG2 stages also validated
all4071 frames. The campaign is COMPLETE; it is no longer running.
No skipped frame, precision fallback, weight/GT change, or beta=0 correction
was used in the final results. The exact low-level compiled failure cause
remains unresolved; successful eager evaluation establishes the mitigation,
not that cause.

See [final results](FULL_SEQ08_SEED42_FINAL_RESULTS.md) for all six quality
rows and training/statistical limitations.

CPU verification:28 tests passed, including failed-artifact preservation,
refusal to archive completed results, the isolated beta=0 correction and
portable consistency checks of the published CSV/JSON bundle.
