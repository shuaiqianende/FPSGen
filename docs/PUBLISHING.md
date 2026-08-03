# Publishing checklist

This document separates the source release from local validation artifacts.
Only files tracked by Git should be uploaded to a public repository.

## Public source

Keep these paths in the release:

- `fpsgen/` and `scripts/`;
- `configs/train_*.yaml` and `configs/smoke_*.yaml`;
- `docs/`, `README.md`, `requirements.txt`, `environment.yml`, and
  `pyproject.toml`;
- `fpsgen/models/ChamferDistancePytorch/` and its upstream MIT notice;
- `third_party/licenses/` and `.gitignore`.

## Local-only artifacts

Do not upload:

- `checkpoints/`, `experiments/`, and `outputs/`;
- dataset directories, preprocessed NumPy/PCD files, logs, TensorBoard data,
  compiled `.so` files, and CUDA build caches;
- `configs/*.local.yaml` or any configuration containing a host-specific path;
- `RELEASE_AND_LOCAL_MANIFEST.md`, which is intentionally ignored and records
  local absolute paths and checkpoint hashes.

The repository's `.gitignore` covers these paths. Verify the final staging area
with:

```bash
git status --short
git check-ignore -v checkpoints/ experiments/ outputs/
rg -n '/home/|/data-[0-9]|LiDiff-main|FPSgen-main' \
  --glob '!RELEASE_AND_LOCAL_MANIFEST.md' \
  --glob '!docs/PUBLISHING.md' \
  --glob '!experiments/**' --glob '!outputs/**' --glob '!checkpoints/**' .
```

The last command should return no host-specific paths in public files.

## Required manual decisions before publication

1. Add a top-level `LICENSE` after selecting the license and copyright holder.
   The bundled Chamfer implementation has its own MIT notice; that notice does
   not choose a license for FPSGen itself.
2. Keep the final paper authors, repository URL, and BibTeX entry synchronized
   with the README citation section.
3. Pretrained checkpoints are released separately through the download links in
   README; do not commit binary files to this repository. Record hashes and
   provenance when publishing a new checkpoint revision.
4. Re-run the smoke checks in a clean environment and record the exact package,
   CUDA, compiler, and checkpoint versions.

## Minimal clean-release verification

```bash
python -m compileall -q fpsgen scripts
python scripts/check_environment.py
python scripts/verify_cuda_ops.py
python -m fpsgen.utils.eval_path_multirange --help
python scripts/generate_eight_conditions.py --help
git diff --check
```

The evaluation help command requires the Chamfer extension to have been
installed as described in `docs/INSTALL.md`. The complete real-data evaluation
also requires a prepared dataset and compatible BEV/Student checkpoints.
