# C1 recovery and fixed training-seed policy

Snapshot: 2026-10-23 15:12 CST.

## Training seed

All new matched B0/M1 arms, including engineering configurations, now use training seed **20261032**, the same seed as the current C1 Static FM/BCE pair. No additional training seed or automatic second-seed retraining is scheduled. Historical cohorts keep their actual seed identities; their metadata is not rewritten to imply a matched comparison.

The small multi-sample forward diagnostics estimate inference variability only. They do not train additional models. Formal paired evaluation keeps its registered fixed seed.

## Interruption found during live inspection

The previous C1 training and Phase0 supervisor processes were absent at the status check. The machine had not rebooted. Both training stdout logs ended on ordinary finite updates, without a terminal exception or final training summary. The reason for process exit is **UNKNOWN**.

| Arm | Last logged update | Complete recoverable update | Recoverable exposures | Discarded uncheckpointed updates |
|---|---:|---:|---:|---:|
| Static FM | 9,204 | 9,000 | 18,000 | 204 |
| BCE | 9,222 | 9,000 | 18,000 | 222 |

Both saved configurations, source hashes and data/sampler identities match the original worktree. Both checkpoints contain optimizer, EMA, scaler, scheduler and rank RNG states. Both are resumed with the original seed20261032, batch2 and full10,000-update horizon.

Checkpoint SHA256 at recovery:

- FM: `261aea2ee0dfd179eef3c5754a0620ee8cdd06e96d0a129789fe9d82e6b70536`.
- BCE: `837c167bfb29febc554c53fc34e921135afdbd372d450ced3f31391c21a602ec`.

The saved states and pre-recovery training logs were copied to a separate server-local recovery directory before continuation. Original training JSONL is retained and resumed by append. It therefore contains duplicate update numbers after9,000. Loss-window summaries must use the most recent row for each successful update, or the explicit attempt boundary in the recovery metadata, rather than counting every JSONL row as a distinct retained update.

The final retained optimization budget remains20,000 successful frame exposures. Recomputing discarded updates adds408 physical frame executions for FM and444 for BCE. This recovery overhead is disclosed and is not treated as additional retained optimization budget.

## Observed restart checks

Both processes now run in independent background sessions (PPID1) with separate recovery stdout files. GPU2 resumed FM; GPU3 resumed BCE. The supervisor is also detached and recognizes the explicitly resumed PIDs.

At the first replay audit:

- FM:58 overlapping updates had exactly matching batch identities; maximum absolute scalar-loss difference0.0001230.
- BCE:64 overlapping updates had exactly matching batch identities; maximum scalar-loss difference0.0000003576.

This confirms the sampled frame sequence at those updates. It does not establish bitwise equality of every activation or parameter, and such a claim is not made.

The automatic continuation remains: finish C1, release the card, evaluate primary900 with raw-map geometry metrics, then D1–D4 and M1 engineering checks. New formal20K training still requires the recorded research GO decision. No100K run is scheduled.
