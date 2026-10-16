# 384 Stage1 raw-map occupancy data

Main geometry is locked by explicit user selection:384x384x32,2x2x2 refinement,
fine768x768x64, open50m sphere and Z(-4,4.4). Geometry SHA:
`c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b`.
Parent size0.2604167x0.2604167x0.2625m; fine0.1302083x0.1302083x0.13125m.

Train17,870, development900, excluded temporal guard360. Frozen manifest bytes
are preserved; no seq08 is accessed. GT comes from `_full_candidate()` map pose,
local crop, height and viewpoint support before Poisson/fixed-point processing.
This is surface-hit occupancy, not verified solid occupancy or free space.

Current snapshot: **8017 voxelized frames,
7898 transactionally committed frames** out of18,770.
Full data audit ready: **False**. Progress is a snapshot,
not an assertion that the full train split is usable. Full Stage1 requires the
assembled train/development manifests and successful full audit.

Output: `/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage1occ/experiments/hier_occ_flow/occ384_full_parallel`. CPU tmux: `fpsgen_occ384_rawmap_parallel`.
Eight low-priority workers process at most1,024 frames per sequence chunk.
Each validates raw-source checksums and independently verifies direct parent
rasterization equals fine-child OR. Labels are uint32 IDs plus uint8 masks.
Shards are atomically committed; the final dataset uses hardlinks to worker
shards, avoiding a second copy. Check unique inode size when measuring storage.
The prior sequential109-frame partial dataset remains intact.

An exact small integer lookup replaces structured-array viewpoint membership,
without changing target support. Four real candidate arrays, including1,322,038
rows of00/000000, match the original exactly in coordinates and order. Raw
candidate time fell from4.45s to1.48s on that frame, excluding map-index setup.
Initial slow-worker committed shards were preserved on restart; implementation
history and source hashes are recorded. See[lookup proof](hs_occ384_stage1_reports/viewpoint_lookup_audit.json).
