# Source provenance

VoxDet reference files are pinned to
[`vita-epfl/VoxDet@e10d8280de96d4f03f54b616e57e7c6754044fa4`](https://github.com/vita-epfl/VoxDet/tree/e10d8280de96d4f03f54b616e57e7c6754044fa4).
Its license is retained at `VoxDet/LICENSE`.

PWM reference files and the PTv3 adapter/trunk are pinned to the private
[`shuaiqianende/PWM@c6e809486db3d0b6f8503318dfea246c06c7a50e`](https://github.com/shuaiqianende/PWM/tree/c6e809486db3d0b6f8503318dfea246c06c7a50e).
Retain the upstream authorship comments in its vendored Pointcept PTv3 source.
These private research sources are included for this private FPSGen experiment;
no new public redistribution permission is asserted.

`SOURCE_MANIFEST.json` records every selected source's SHA-256 and each adapted
module's SHA-256. `scripts/research_v2/voxel_point/vendor_sources.py` reproduces
the extraction from the pinned repository clones. Reference files here remain
unchanged; active extracted modules are in `fpsgen/voxel_point/vendor/`.
The scoped `.gitattributes` retains upstream trailing whitespace in these
verbatim reference trees; active adapted source uses ordinary whitespace checks.

The active OT adapter reuses the existing FPSGen sparse graph builder and
log-domain Sinkhorn kernel. It adds plan-coverage diagnostics and acceptance
policies without modifying those original FPSGen files.
