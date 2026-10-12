# GPU and independent environments

Preflight on2026-10-13: four RTX3090 GPUs,24GB nominal each; driver580.159.03,
CUDA driver API13.0, installed CUDA toolkit11.7.64. GPU2/3 were free except
small Xorg allocations. Existing GPU0 workloads were inspected read-only and
never stopped or used. All experiment kernels ran with CUDA_VISIBLE_DEVICES
2,3 or2,3; logical indices0/1 map to those physical cards. Sandboxed nvidia-smi
could not see the driver; approved host execution supplied the actual evidence.

System CUDA/drivers/Python/PATH/shell startup and the existing FPSGen environment
were not modified. Base interpreter is Python3.9.19. Two venvs under the repository
have independent copies of the existing repository-local wheel packages,
followed by dependency installation **only in these new venvs**:

- `env/voxdet_density_py39`
- `env/pwm_ptv3_pointflow`

This is an explicit compatibility track, not a Python3.7/PyTorch1.10.1/CUDA11.3
historical-environment reproduction. The available11.7 toolkit and verified
Torch2.0.1+cu117/RTX3090 combination were selected; no claim is made that the
historical environment was impossible to install. Official module bodies and
parameter scale remain preserved. Full MMDetection3D camera/data registries
are unnecessary because selected official LiDAR modules use installed MMCV
factories directly. Architecture preservation is audited separately. Final import extraction also preserves
the original MMCV tokenizer/MMEngine neck BaseModule sources. Same-seed complete
state SHA256 is identical before/after that alignment; final GPU2 one-update
regression passes (base_module_parity.json).

Final relevant versions: Torch2.0.1+cu117, torchvision0.15.2+cu117,
NumPy1.23.5, SciPy1.10.1, timm0.9.16, spconv-cu1172.3.6,
CUDA torch-scatter2.1.2+pt20cu117, FlashAttention2.5.9.post1,
KeOps2.3, einops0.8.0; VoxDet additionally MMCV1.7.2, MMEngine0.10.7,
OpenCV4.8.1.78. Final pip check passes in both venvs. Full freeze files are
in voxel_point_reports. The existing FPSGen Torch2.0.1+cu117/NumPy1.23.5
versions remain unchanged.

Reproduction setup script refuses to overwrite either existing venv.
The actual initial venv setup used independent file copies to avoid re-downloading
5.3GB of available packages per environment; reproduction installs pinned
wheels in empty venvs. All dependency/build/cache paths are repository-local.
No hard-linked package installation or shared writable site-packages is used.

## Verified gates

Stage1: GPU2 FP32 forward/backward1 update; GPU2 FP16 AMP16 successful updates;
GPU3 FP32 and FP16 one update each; GPU2+3 DDP32 updates,B1/GPU. Main/aux density,
offset supervision and gradients finite. GPU2 peaks: FP32≈4.80GiB,
FP16≈4.74GiB. DDP≈4.82GiB/rank, steady step≈0.76s, all-in average≈0.94s.

Teacher: GPU2 actual180000 source+180000 context16 updates pass; GPU3 mixed
GT/predicted sources,5 real frames16 updates pass. Peaks≈9.00/9.85GiB,
steady≈1.42/1.44s per training step. Source/context cardinality never reduced.
Block checkpoint parity:512-point numerical equivalence test (separate from
180k gate), output and parameter gradient differences0.

OT: actual180k GT-source K16 on20 scenes and predicted-source K32 on2 scenes.
Reciprocal graph, coalescing, FP32 sparse plan and marginals evaluated.
GT K16 wrapper runtime≈0.82s, allocated peak≈0.57GiB. Predicted K32≈1.18–1.43s,
peak≈1.07GiB. Marginal/geometry acceptance and failures are in the results.

Student: GPU2 and GPU3 each16 actual180k updates pass. GPU2+3 DDP32 successful
updates pass after cache fix; each rank sees16 GT and16 predicted sources.
Peak≈5.46GiB/rank; steady≈2.2s including frozen Teacher/OT target generation,
velocity loss and backward. All gradients finite.

End-to-end: first5 real frames and20-frame final run pass, single Stage1 forward,
Euler8 PointFlow,180k finite float32 final points, no GT/Teacher/Sinkhorn
in deployment. Peak≈2.98GiB. GT evaluated only in a separate process.

All reported GiB peaks use torch.cuda.max_memory_allocated; CUDA context and
allocator reservation are not included. Cold first-step autotuning/compilation
is reported separately from warmed timings. These are smoke measurements,
not formal mixed-sequence throughput or comparison with BEVFlow latency.

## Compatibility failures and corrections

The full timeline/evidence is preserved in compatibility_failures.json:
non-Flash Teacher OOM; Torch2.0 nonreentrant/Flash saved-leaf failure;
hierarchy-parent checkpoint closure memory accumulation; sparse/dense input
parity correction; SpConv eval AMP algorithm failure; and KeOps DDP device-cache
failure. The final successful runs use block reentrant checkpointing without
hierarchy-parent captures, preserve exact unpool CPE inputs, use sparse eval
FP32/Flash QKV FP16, and isolate KeOps caches by visibility/rank.

launch.sh checks physical GPU IDs before CUDA initialization, sets local
Torch-extension/KeOps/XDG/tmp caches and uses only logical cuda:0/1. It supports
DDP for training when visibility is2,3. It launches one requested command,
creates no long training queue and performs no destructive process management.
