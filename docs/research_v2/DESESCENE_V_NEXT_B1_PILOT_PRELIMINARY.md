# DeSeScene V-next B1: preliminary 1K-exposure pilot

Date: 2026-10-23
Status: preliminary diagnostic; matched 20K-exposure runs are in progress.

## Protocol

Two Stage2 variants were initialized from the same registered model initialization and trained with the same packed batch-2 data order, seed, predicted Stage1 support, Stage1 NFE=4, threshold 0.5, FM objective, and optimizer schedule. The only intended difference is the Child input: `geometry_stats` uses input-only per-Child hit/count/local-coordinate statistics; `position_only` uses the same Child position encoding and decoder capacity while ignoring those statistics. Ground-truth occupancy is used only as the training target and for post-hoc evaluation.

Each pilot completed 500 successful updates = 1,000 successful frame exposures. Both resumed from their saved pilot checkpoints toward 10,000 updates = 20,000 exposures. The evaluation uses the same first eight fixed development records and sampling seed 20261031, with predicted support. It is an early learning diagnostic, not the registered 900-frame quality result.

Artifacts:

- Geometry variant training: `experiments/desescene_v_next/child_bridge_geometry_20k/`
- Position-only control: `experiments/desescene_v_next/child_bridge_position_20k/`
- Per-variant diagnostic JSON: `pilot1000_eval8.json`
- Frozen Stage1 checkpoint SHA256: `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`

## Eight-frame diagnostic

| Child input | NFE | Native Fine IoU | Native Fine F1 | 0.15m center-proxy IoU | Mean Stage2 time (ms/frame) |
|---|---:|---:|---:|---:|---:|
| Geometry statistics | 1 | 0.207540 | 0.342999 | 0.259192 | 109.1 |
| Geometry statistics | 2 | 0.207624 | 0.343114 | 0.259266 | 179.0 |
| Geometry statistics | 4 | 0.207664 | 0.343168 | 0.259306 | 355.3 |
| Geometry statistics | 8 | 0.207683 | 0.343195 | 0.259325 | 699.4 |
| Position-only control | 1 | 0.207543 | 0.343003 | 0.259198 | 110.4 |
| Position-only control | 2 | 0.207626 | 0.343116 | 0.259268 | 175.3 |
| Position-only control | 4 | 0.207666 | 0.343171 | 0.259307 | 348.2 |
| Position-only control | 8 | 0.207687 | 0.343199 | 0.259328 | 692.1 |

At 1K exposures the two variants are nearly indistinguishable on these eight records. The absolute quality is still low at this early checkpoint. These results neither establish a benefit nor rule one out; the matched 20K runs and 900-frame evaluation are required. The NFE curve is also nearly flat here, so this pilot alone does not establish a preferred sampler or prove FM is inferior to BCE.

Timing in this table is diagnostic evaluation timing and is not a formal inference benchmark. Candidate counts, score definition, and per-frame records are preserved in the JSON artifacts. Candidate support is predicted; no GT parent is inserted into the inference set.

## Current execution state at report time

- GPU2 geometry-statistics variant: resumed toward 10,000 successful updates / 20,000 exposures.
- GPU3 position-only control: resumed toward the same target after its matched eight-frame diagnostic.
- Both use global batch 2 (one single-GPU experiment per physical GPU). No Stage1 or Stage2 checkpoint has been modified.
- The next decision point is the matched 20K/900-frame comparison. Only then should the Child-Aware Bridge be considered for a larger study or the C1 Static-Context-vs-BCE experiment be scheduled.
