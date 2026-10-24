# Local state pilot: current progress and next experiments

Captured: 2026-10-24T21:42:16.516829+08:00

**Training is still in progress. Final L0/L1 quality is NOT TESTED.** This dated snapshot supersedes the earlier GPU1 assignment; the actual pair is GPU2 position and GPU3 position+attention.

| Arm | GPU | Successful updates | Successful exposures | Recent500 seconds/update | Remaining training hours, estimate |
|---|---:|---:|---:|---:|---:|
| position | 2 | 3287 /10000 | 6574 /20000 | 2.211 | 4.12 |
| attention | 3 | 3284 /10000 | 6568 /20000 | 2.224 | 4.15 |

Single training seed20261032, real packedBatch2, same original raw-GT source and Frozen Stage1 NFE2 support. AMP skipped attempts are excluded from exposure counts. ETA follows the last500 successful updates and excludes evaluation; changing scene costs can move it.

## Loss in200-successful-update windows

| Updates | B0 original reference | Position | Position+Attention |
|---|---:|---:|---:|
| 1–200 | 0.606988 | 0.608793 | 0.608891 |
| 201–400 | 0.590184 | 0.593062 | 0.592469 |
| 401–600 | 0.587233 | 0.590508 | 0.590054 |
| 601–800 | 0.581761 | 0.584566 | 0.582374 |
| 801–1000 | 0.610569 | 0.612784 | 0.610879 |
| 1001–1200 | 0.589987 | 0.588498 | 0.587612 |
| 1201–1400 | 0.575984 | 0.578030 | 0.575680 |
| 1401–1600 | 0.578843 | 0.579053 | 0.579497 |
| 1601–1800 | 0.566889 | 0.564990 | 0.568162 |
| 1801–2000 | 0.575266 | 0.573112 | 0.572369 |
| 2001–2200 | 0.565413 | 0.563942 | 0.564949 |
| 2201–2400 | 0.545695 | 0.545347 | 0.543401 |
| 2401–2600 | 0.573409 | 0.574163 | 0.571560 |
| 2601–2800 | 0.565653 | 0.564013 | 0.563433 |
| 2801–3000 | 0.562535 | 0.561998 | 0.561562 |
| 3001–3200 | 0.575715 | 0.575363 | 0.574135 |

Full and partial windows are distinguished in `loss_200updates.csv`. Batch-order checks are saved in summary.json: different AMP skip histories can misalign successful-step windows. Training loss does not establish a geometric quality gain.

## Automatic continuation and candidates

The transferred supervisor adopts the original running GPU3 process and launches the formerly waiting position arm on idleGPU2. Source hash, configurations, seeds, optimizer horizon and all existing artifacts are preserved. The transfer audit and exact server-local supervisor script are retained under runtime/.

After each20K run, the active queue automatically evaluates900 Development frames (NFE4, native Fine IoU/F1, exact KD-tree CD, corrected physical IoU and JSD), followed by64-frame NFE/conditioning diagnostics. Compare against completed B0; report matched Parent support, worst-frame deltas and actual runtime before selecting a candidate.

Next candidates are Encoder-only adaptation with frozen Stage1 Flow, matched Encoder+Flow adaptation, and hypothesis-driven scratch retraining. Each is limited to20K successful exposures, Batch2 and the same training seed. **Their engineering runner/checks and automatic training continuation are not yet implemented. They are registered candidates, not active or auto-queued training.** Existing verified scratch20K controls can be reused when identities match. No automatic100K or additional training seeds.

Completed C1/B0/M1, Stage1 quality and three-round Encoder timings remain in `../desescene_completed_20261024/`. The current snapshot adds progress,200-step losses, GPU2 scheduling and follow-up readiness. Checkpoints and prediction clouds remain server-local.
