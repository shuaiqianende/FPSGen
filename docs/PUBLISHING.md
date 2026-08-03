# Release checklist

Keep source code and release metadata in Git. Keep datasets, generated files,
and binary checkpoints outside the repository.

## Include

- `fpsgen/`, `scripts/`, `configs/`, `docs/`, and `third_party/licenses/`;
- `README.md`, `LICENSE`, `requirements.txt`, `environment.yml`, and
  `pyproject.toml`;
- the bundled `fpsgen/models/ChamferDistancePytorch/` source and MIT notice.

## Exclude

- `checkpoints/`, `experiments/`, `outputs/`, and all dataset files;
- TensorBoard logs, generated NumPy/PCD/PLY files, compiled `.so` files, and
  CUDA build caches;
- `configs/*.local.yaml` and any file containing a machine-specific path;
- `RELEASE_AND_LOCAL_MANIFEST.md`.

## Verify

```bash
git status --short
git check-ignore -v checkpoints/ experiments/ outputs/
rg -n '/home/|/data-[0-9]|LiDiff-main|FPSgen-main' \
  --glob '!RELEASE_AND_LOCAL_MANIFEST.md' \
  --glob '!docs/PUBLISHING.md' \
  --glob '!experiments/**' --glob '!outputs/**' --glob '!checkpoints/**' .

python -m compileall -q fpsgen scripts
python scripts/check_environment.py
python scripts/verify_cuda_ops.py
python -m fpsgen.utils.eval_path_multirange --help
python scripts/generate_eight_conditions.py --help
git diff --check
```

## Before release

- [ ] Confirm the project license and copyright holder.
- [ ] Keep the README paper link and BibTeX entry up to date.
- [ ] Publish checkpoint hashes and provenance with every new checkpoint.
- [ ] Record package, CUDA, compiler, checkpoint, and seed versions for final
      experiments.
