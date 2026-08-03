# Validation record

The repository was validated from scratch on an NVIDIA GeForce RTX 3090 with
Python 3.9, PyTorch 1.13.0 + CUDA 11.7, MinkowskiEngine 0.5.4, and PyTorch
Lightning 1.8.1.

- The bundled Chamfer operator compiled and passed a CUDA forward/backward test.
- Stage 1 BEV smoke training completed one batch from a real preprocessed KITTI
  sequence (loss: 2.440) and wrote a checkpoint.
- Stage 2 teacher smoke training completed one batch (loss: 2.320) and wrote a
  checkpoint.
- Stage 3 student smoke training loaded that new teacher checkpoint and
  completed one batch (loss: 0.154).
- The new BEV and student checkpoints completed one real input-frame inference
  with one BEV and one PointFlow integration step, producing 180,000 XYZ
  points.
- Checkpoint test mode reached one real batch for every stage. A later audit
  corrected test loading to retain full ground truth and separated `test/*`
  logging from `train/*`; numerical test losses from the earlier execution are
  therefore intentionally not reported as evaluation values.
- The audited student passed a 2,048-point CUDA forward/backward test:
  stage-2 weights loaded strictly, the frozen teacher stayed in evaluation mode
  with no gradients, and student parameters received gradients.
- Dataset alignment was checked on a real sequence: frame filenames selected
  the matching per-sequence poses, labels matched their point arrays, and test
  samples contained 180,000 full versus 18,000 partial points.
- Inference strictly loaded the smoke student and BEV checkpoints, reconstructed
  the expected XYZ coordinates from a single occupied BEV cell, and rejected
  an empty occupancy mask with an explicit error.
- The built-in BEV Euler sampler completed a one-step CPU check and returned a
  finite three-channel tensor with shape `[1, 3, 256, 256]`.
- The paper-alignment audit verified that a single occupied cell is sampled at
  its metric center (`-49.8046875 m` for cell zero on the 256-cell axis) and
  that BEV height uses the maximum point height.
- Inference reconstruction independently recovered center cell 128 at
  `0.1953125 m` on both BEV axes with zero sampling noise.
- After enabling the complete `[D,H,M]` PointFlow condition, a fresh 2,048-point
  CUDA forward/backward check passed (loss: `0.127758`): the frozen teacher had
  zero gradient tensors and the student had 355 gradient-bearing tensors.
- After removing the unused Teacher displacement diagnostic, a 2,048-point
  CUDA forward/backward check passed (loss: `1.456591`). The only recorded
  quantities were `train/loss`, `train/loss_cd`, and `train/loss_knn`.
- The eight-condition generator completed full model inference on GPU 1 using
  BEV epoch 179 and Student epoch 2: all eight generated clouds contained
  180,000 finite XYZ points, and the manifest contained all eight condition
  records. The public script now writes PLY by default; its legacy PCD output
  remains available with `--output-format pcd`.
- An earlier sequence `00` run was only a trajectory smoke check on training
  data and must not be reported as a test-set result.
- The final copied-checkpoint evaluation used physical GPU 1, the LiDiff
  BEVFlow epoch-499 checkpoint, Student epoch 09, condition `100`, every
  100th frame (41 frames), and BEV/PointFlow steps `10/1`. It used raw
  `velodyne` input and the pose-cropped `map_clean.npy` reference, with DCD and
  EMD disabled. The resulting Chamfer was `0.330378 m` (std `0.112203 m`),
  completion IoU was `0.430485/0.308478/0.174546` at voxel sizes
  `0.5/0.2/0.1 m`, 3D JSD was `0.499620`, and BEV JSD was `0.327917`.
  The run produced 41 PLY files and no PCD files.

These checks document local reproducibility and are not a substitute for an
official benchmark table. Before publication, re-run the final protocol after
installing the public environment and record the exact checkpoint hashes and
random seeds.
