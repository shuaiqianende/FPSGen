# HS-OccFlow data audit — measured CPU milestone

Branch: `research/hs-occflow`, base `612cc976cc7f632eb1e1e1e3a765ff546fcf1acc`.
Existing worktree modifications, checkpoints and ongoing jobs were preserved.

The frozen numeric frame pairing/time-block protocol contains 17,870 train,
900 development and 360 guard frames, with no seq08. This milestone processed
**16 real frames only**: sequences00/05/09/10, frames0/200(train),500/550(dev).
It does not characterize population P99 or the true largest training scene.

GT comes from `map_clean.npy` plus calibrated poses and raw scans, using the
existing `_full_candidate()` before voxel downsampling, Poisson, or fixed-count
sampling. A cKDTree crops the map first and preserves row order. Frame00/000000
was checked against a complete-map reference scan: **1,322,038 rows exactly equal**.
Raw candidates across this cohort range from564,780 to1,624,505 points.
Map, pose, calibration, raw-scan and source-script SHA values are archived in
[raw source manifest](hs_occflow_reports/raw_source_manifest.json).
Canonical source manifest SHA: `c9b5841c54818f59ca0095f806c047d9e935c806a206f18d050ccea2ff784f38`.

This is **surface-hit occupancy**. Missing map points do not prove free space.
The original map construction's exact0.1m provenance is not established here.
Randomly retaining90% of raw candidates preserves95.39% of native fine support
on256+18 (mean over16 frames). This sensitivity must be considered in evaluation.
Raw-vs-historical-Poisson native support IoU is0.3391 for256+18 and0.3720 for352+8;
historical180k labels are therefore not substituted for this task's raw targets.

Labels are sorted uint32 parent IDs plus uint8(K8) or uint32(K18) masks.
No dense fine grid, full parent XYZ, float occupancy, or raw candidate cache is
persisted. Versioned index files and shard hashes are committed by a single
atomic manifest replacement. Explicit resume verifies source/geometry/version
and checksums; orphan partial shards are ignored/replaced. The tests cover
resume, corruption and parent/child bit ordering.

| Setting | Parent tokens P50 / sample max | Packed MB/frame | Est. train GB | Quantization P90 (cm) |
|---|---:|---:|---:|---:|
| 256_18 | 84,404 / 127,703 | 0.651 | 11.63 | 8.52 |
| 320_8 | 118,072 / 178,738 | 0.566 | 10.12 | 9.71 |
| 352_8 | 137,319 / 206,914 | 0.656 | 11.72 | 9.05 |
| 384_8 | 156,792 / 236,151 | 0.747 | 13.36 | 8.52 |

Capacity estimates multiply **sample mean** label bytes by17,870; they exclude
input LiDAR, optional feature caches and checkpoints. Only this small cohort
exists locally. Full preprocessing is gated on a locked geometry decision.

See [all64 frame/geometry audit rows](hs_occflow_reports/resolution_audit.json).
