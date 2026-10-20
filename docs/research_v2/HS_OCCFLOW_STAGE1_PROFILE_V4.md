# HS-OccFlow Stage1 V4 Profile

Status: profiling implementation is ready; full module profile is pending a safe GPU2/3 window.

## Frozen system

- V4 branch starts from V3 commit `f51a74096ce9bd4575164fce04e8f54c6fa540c3`.
- Stage1 EMA SHA: `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`.
- Stage2 Online SHA: `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`.
- PyTorch `2.0.1+cu117`; RTX 3090, batch 1, Stage1 2 NFE and Stage2 4 NFE.
- V3 features remain enabled for comparisons: GPU candidate selection, observed-child mask and occupancy decode, Stage2 static condition cache, PTv3 multilevel topology cache, PTv3 AMP FP16 and batch-1 fastpath.

## Initial matched GPU3 baseline

One registered 5-warmup/20-frame run was completed with V3 features and the V4 runner before adding the extra hash audit. It is a valid same-GPU timing snapshot, but must be repeated with the cache variant under the final V4 runner before claiming a speedup.

| Metric | Mean | P95 |
|---|---:|---:|
| Stage1 Encoder CUDA Event | 204.40 ms | 205.88 ms |
| Stage1 Flow ×2 CUDA Events | 107.80 ms | 108.13 ms |
| Stage2 PTv3 ×4 CUDA Events | 144.30 ms | 178.80 ms |
| Network CUDA Events | 456.50 ms | 490.86 ms |
| Full generation Wall Time | 480.92 ms | 523.74 ms |
| Peak allocated memory | 3.01 GiB | 3.01 GiB |

Raw records: `experiments/hs_inference_opt_v4/baseline/v3_gpu3_5x20/`.

## Profiling breakdown

`scripts/research_v2/hier_occ_inference/profile_stage1_v4.py` measures real Seq08 frames with module-level CUDA events and host submit times. It records tokenizer input feature construction, `torch.unique`, `scatter_max`, Point MLP leaves, dense encoder leaves, Flow leaves, condition projections, interpolation and concatenation. Leaf/module timings are inclusive where modules contain child work; they must not be blindly summed. The full profile has not yet run because GPU2 and GPU3 were occupied by external processes on repeated admission checks. The script applies the existing strict GPU admission policy and does not terminate processes.

## Next result fields

The next safe run will include three representative frames (low, middle and high candidate counts), with these reported separately:

- Encoder total and per-leaf/operator CUDA time and host submission time.
- Tokenizer filtering/features, Point MLP, unique and scatter-max.
- Flow two-step total, condition projection calls, Flow blocks, interpolation and concatenation.
- Call counts and Candidate count for each scene.

No result is inferred for unmeasured operators. See `HS_OCCFLOW_STAGE1_FLOW_CACHE_V4.md` for the cache implementation and parity status.
