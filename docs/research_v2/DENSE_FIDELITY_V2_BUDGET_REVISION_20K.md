# Dense Fidelity V2: 20K Exposure Budget Decision

## Decision

Use **20,000 successful frame exposures per architecture** as the comparison
budget. Do not continue the 100K-exposure R0/E3 long-run plan for this study.
The existing R0, E1, E2, and E3 pilot checkpoints already completed 20,000
successful exposures each (5,000 optimizer updates at global batch 4), so no
restart is needed for the registered 20K comparison.

The matched pilot runs use seed `20261021`, the same training and development
manifests, global batch 4, the same 5,000-update warmup-cosine schedule, and
the same shared Flow initialization hash. The R0 20K checkpoint is retained in
the OMD V1 worktree; E1/E2/E3 checkpoints are in the Dense Fidelity V2
worktree. The original-path source commit differs for R0, so reports should
retain that provenance rather than describing all arms as built from one
identical commit.

## Superseded long-run attempt

The separate R0 run configured for 100,000 exposures was gracefully paused
after 3,380 successful updates / 13,520 successful exposures. Its SIGTERM
handler saved `resume_latest.pt` and wrote a `PAUSED` summary. This partial
run used the 100K cosine horizon and is **not** a 20K comparison result; keep
it for provenance only and do not resume it as one of the matched 20K arms.

Its output remains under:

```text
experiments/dense_fidelity_v2_longrun/r0_gpu2_100k/
```

No historical checkpoint, pilot output, or report was overwritten or removed.

## Single-GPU batch-2 option

A new single-GPU batch-2 run would require 10,000 updates to reach 20,000
exposures. It is unnecessary for the current architecture comparison because
the existing arms already share global batch 4 and the same exposure budget.
If run later as an operational variant, report it separately: equal exposure
does not make its optimizer-update count or batch dynamics identical to the
global-batch-4 study.

## Verified run counts

| Arm | Successful exposures | Updates | Global batch | Result |
|---|---:|---:|---:|---|
| R0 Original, 20K pilot | 20,000 | 5,000 | 4 | Complete requested budget |
| E1 Bottleneck-96 | 20,000 | 5,000 | 4 | Complete requested budget |
| E2 Asymmetric3D rank-64 | 20,000 | 5,000 | 4 | Complete requested budget |
| E3 Lightweight Skip | 20,000 | 5,000 | 4 | Complete requested budget |
| R0 100K long-run attempt | 13,520 | 3,380 | 4 | Paused; excluded |

All four 20K arms use the same train manifest SHA256
`8dc632d341ffc2c6f0c139322cd24767616d570cf84c91fb4e516a26d7bab064` and
development manifest SHA256
`6447076563b6c0efffbd3a07c5f695ebedf8aef1fa270a5cda9d99df73b30d53`.
