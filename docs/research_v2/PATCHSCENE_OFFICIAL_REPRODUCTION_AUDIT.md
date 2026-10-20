# PatchScene official checkpoint reproduction audit

## Status

The official pretrained model has been strictly loaded and run on real
SemanticKITTI sequence 08 inputs. The one-frame smoke and the first-eight-frame
paired evaluation completed. Full sequence inference is currently running in
tmux; do not treat the smoke metrics below as the final 408-frame result.

## Provenance

- Upstream: `https://github.com/AFARI-Research/PatchScene`
- Frozen upstream commit: `49de6d848663956e57cdd2f95d3b364e7a7afbe6`
- Official SemanticKITTI checkpoint: `epoch=99.ckpt`
- Checkpoint SHA256: `f50bf65f419c32bb40afa14e30440e1c4ea839dd92c678f8b4487e0d2b36140a`
- Strict checkpoint load: passed; model state matched with no missing or
  unexpected parameters. Inference uses the official EMA model, as selected by
  the README inference entry point.
- Architecture: 59,009,777 parameters in each online and EMA model.
- Official generation method and schedule: `Ring_Spread_V2`, 15 denoising
  steps. The wrapper calls the official generator method and does not replace
  its denoising, ring propagation, fusion, or checkpoint weights.

The PatchScene source and checkpoint live outside the FPSGen tracked source
tree under `worktrees/patchscene-official`; neither is included in FPSGen Git.

## Runtime environment

The available project runtime used for strict loading and GPU inference is
Python 3.9, PyTorch `2.0.1+cu117` / CUDA `11.7`, and PyTorch Lightning `1.8.1`.
This differs from the upstream README's suggested PyTorch 2.4.1 / CUDA 12.1
environment. An isolated environment install was attempted, but package-index
network access timed out / was denied. No existing project environment or
system CUDA installation was modified. The official checkpoint loads strictly
and the official model completes inference in the available runtime; the
runtime difference remains recorded as a reproducibility caveat.

## Inference-only audit

`patchscene_infer_seq08.py` reads only the same FPSGen `input_/*.npy` XYZ
frames supplied to HS-OccFlow. It constructs the official occupancy input and
ring patch bounds without instantiating the official map/annotation dataset.
It does not load `map_clean.npy`, `pc_full`, `pc_full_occu`, or any GT during
model inference. GT is loaded later by `eval_patchscene_unified.py` through
the existing HS-OccFlow `RawMapSource` and frozen viewpoint-support protocol.

Source inspection of `network/model.py::genralized_Ring_Spread_CFG` shows that
`last_ring_xs` is consumed for within-frame ring-to-ring propagation. The
official test script stores `pred_last`, `xs_last`, and `pose_last` after each
frame, but the default Ring_Spread_V2 generation path does not read them to
condition the next frame; `pc_curr_in_last_occu` and `guidance_weight` remain
`None`. Therefore target frames can be inferred independently in the official
default mode. The wrapper resets ring state per frame while retaining the
official within-frame ring sequence.

The 408 audited `input_` scans each contain 18,000 XYZ points and were verified
as byte-exact XYZ subsets of their matching SemanticKITTI `velodyne/*.bin`
scans. Both models are supplied that same 18K subset. The paired smoke also
verified identical per-frame input-file SHA256 and identical target-GT SHA256
between the two model evaluators.

## Initial results (smoke only)

The first eight ordered sequence-08/stride-10 frames completed official
15-step inference. Mean occupied native voxels were 411,782; mean centers
inside the common HS evaluation ROI were 408,319. Mean model inference time
was about 32.7 seconds/frame, with 8.92 GiB peak PyTorch allocated memory
(approximately 16.6 GiB process-visible GPU memory during the run).

The shared metric evaluator used the exact GPU KD-tree Chamfer implementation,
corrected fixed-physical-grid IoU at 0.5/0.2/0.1 m, and the audited FPSGen
JSD implementation. Both native centers and 1.08M native-voxel jitter points
were evaluated against the same raw-map GT frames. See
`HS_OCCFLOW_VS_PATCHSCENE_UNIFIED_EVAL.md` for the paired eight-frame table.

## Full evaluation

- tmux session: `patchscene_native_408_gpu3`
- launcher log: `/tmp/patchscene_native_408_gpu3.log`
- output: `experiments/patchscene_eval/official_seq08_skip10_native_15steps_gpu3_run2/`
- intended frames: 408, IDs `000000, 000010, ..., 004070`
- status is stored in that output's `progress.json` and `protocol.json`.

The full run is resumable only with the same checkpoint, commit, seed, step
count, frame manifest, and output protocol. This report must be updated from
the final result files after all 408 predictions and unified metrics finish.
