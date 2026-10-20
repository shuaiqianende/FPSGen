# HS-OccFlow V6 precision audit

## Scope

The V6 worktree uses the original Stage1 EMA and Stage2 Online checkpoints (SHA256 values recorded in the benchmark protocols), batch 1, Stage1 2 NFE, Stage2 4 NFE, threshold 0.5, the existing GPU candidate/observed/decode pipeline, and the V3 multilevel PTv3 topology cache. Geometric XYZ, voxel IDs, source noise, and Euler state are not cast to FP16. The V6 explicit conversion is applied only to independent inference model instances after strict FP32 checkpoint loading.

## Runtime findings

The original V0 path is not a uniform FP16 network. The Stage1 encoder and Stage1 flow have FP32 stored parameters and run under FP16 autocast. Stage2 retains FP32 stored parameters and runs under the configured PTv3 AMP path; its attention kernel already uses half precision internally. The Stage1 and Stage2 Euler state updates remain FP32.

The full explicit variant converts all floating parameters in each selected network to FP16 after checkpoint load. The observed parameter storage is:

| Component | FP32 bytes | Explicit FP16 bytes | Reduction |
|---|---:|---:|---:|
| Stage1 encoder | 76,226,272 | 38,113,136 | 50% |
| Stage1 flow | 2,096,292 | 1,048,146 | 50% |
| Stage2 | 10,708,512 | 5,354,256 | 50% |
| Total | 89,031,076 | 44,515,538 | 50% |

This is parameter storage only, not total runtime memory. Runtime hooks on representative LiDAR tokenizer Linear/BatchNorm, VoxDet Conv3d/BatchNorm/ReLU, PTv3 Linear/QKV/FFN/SceneNorm and sparse convolution modules confirmed half activations in the explicit run. Stage1 frequency buffers are kept FP32; trig/time values are formed in FP32 before the learned projection. LiDAR XYZ is voxelized at its original precision, and the six constructed features are cast only at the network entry. Sparse coordinates and indices remain integer / geometry precision. Both flow integrations remain FP32.

The diagnostic hooks did not observe functional Softmax operators, so this audit does not claim their internal accumulator dtype. Norms are stored and executed in FP16 in `fp16_full`; the selective mode preserves normalization parameters/statistics in FP32 and casts output back to the incoming activation dtype. A selective encoder smoke completed, but was slower than full FP16 and was not promoted. Selective Stage2 was not run as a full benchmark.

## Implementation details

`explicit_fp16_copy()` rejects training modules, converts the inference copy, optionally installs FP32 normalization precision islands, and restores floating buffers named `frequencies` to FP32. The ordinary training/checkpoint path is unchanged. No global default dtype is modified. The conversion helper is covered by CPU tests; actual GPU runtime dtypes and peak allocations were measured separately with diagnostic hooks, outside formal timing.
