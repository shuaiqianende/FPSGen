# Actual step1000 PLY inference on GPU1

The user explicitly authorized physicalGPU1 for this export. It was idle before
admission; inference used CUDA_VISIBLE_DEVICES=1 (localcuda:0), and GPU1 returned
to idle after completion. The training-only GPU2/3 admission policy was not
modified. Stage2 training remains in its original tmux and output directory.

Snapshot:1000 successful updates,3520 cumulative successful frame exposures.
SHA256: `00627719093af63d0f7f63bee2dbcc8a33a832e009c4b0542d59724cddb74643`.
The current Stage2 snapshot and the same frozen final Stage1 EMA are used.
No training weights, optimizer, data order or training-source SHA was changed.

Frame: seq08/000000. Output root:
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_ply/step001000_seq08_000000`.
Checkpoint snapshot is in the sibling `snapshots/stage2_step001000.pt`.

The new CLI lives outside the hashed training-source directory:
`scripts/research_v2/hier_occ_inference/export_stage2_snapshot.py`.
It validates the ORIGINAL training configuration/checkpoint identity, then
constructs a separate diagnostic predicted-candidate policy. This avoids
mutating a training config just to make its identity resemble deployment.
It refuses a busy GPU1, never stops jobs, and does not alter global settings.

## Actual output

Stage1 runs its LiDAR Encoder once and4 Flow evaluations, selecting175582
predicted parent voxels at score>0.5. That threshold is an existing uncalibrated
diagnostic setting; it was not chosen or tuned using seq08. Stage2 was trained
with Oracle support, so predicted-parent export is a distribution-transfer
diagnostic, not evidence of a fully validated predicted-parent model.

| Stage2 NFE | Predicted online fine voxels | Predicted EMA fine voxels |
|---|---:|---:|
|1|465499|601663|
|2|486134|604064|
|4|494557|605346|
|8|497911|605891|

Each NFE and weight uses the same initial Gaussian seed and fixed conditions.
The raw endpoint threshold is0, corresponding to the symmetric binary target.
No extra outlier filter, mesh processing or fixed180K point allocation is used.
PLY vertices are the native768×768×64 surface-hit occupied-cell centers.

Input-only predicted exports finish BEFORE optional GT labels are opened.
Separately labeled Oracle exports use the previously audited384+8 raw-map
target for this SAME frame, with geometry/input SHA checks. All245376 GT parent
voxels plus61344 nearby empty parents give306720 Oracle tokens; scores are
neutral for both positives and negatives. Oracle onlineNFE8 has895181 fine
voxels, and GT has908870. Neither Oracle export nor its comparison is a
GT-free deployment result. The original Poisson180K cloud is not used.

Convenient root-level PLY hardlinks use
`stage2_predicted_online_nfe{1,2,4,8}.ply`,
`stage2_predicted_ema_nfe{1,2,4,8}.ply` and corresponding `oracle` names.
They share storage with the nested export files. Also saved:
`stage1_coarse_occupancy.ply`, `partial_lidar.ply`, `gt_fine_occupancy.ply`.

All16 Stage2 PLYs passed actual header/point-count/body-length/finiteXYZ checks.
Core Stage2 PLY payload totals about146MB, with hardlinks adding no duplicate
payload. Two inference-policy CPU tests passed. The complete manifest and
provenance remain next to the PLYs; a small actual GPU audit is committed under
`stage2_384_audit/step001000_gpu1_ply_export.json`. PLYs and checkpoints are
excluded from Git. No quality superiority claim follows from a successful
export or from point counts alone.
