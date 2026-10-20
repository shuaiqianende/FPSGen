# HS-OccFlow Inference Speed V5 — Results

## Outcome

V5 profiled the Stage1 Conv3d path and tested dense-Encoder `torch.compile`, Conv–BatchNorm fusion, and Stage1 Flow CUDA Graph on physical GPU2 (RTX 3090). It did **not** produce a demonstrated, quality-preserving end-to-end speedup over V4. The V4 configuration remains the recommended default. The experiment did identify a modest optional classification-neck compile result, but its paired full-generation interval includes zero and it changes candidate/occupancy outputs slightly.

Frozen protocol: Stage1 EMA and Stage2 Online checkpoint SHA256s remained `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2` and `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`. Stage1=2 NFE, Stage2=4 NFE, threshold 0.5, same batch-1 inputs, compiled Stage1 Flow, GPU candidate/observed/decode, Stage2 AMP FP16 and multilevel topology/static condition caches were preserved.

## Performance

Same-GPU standard five warmups plus twenty timed frames:

| Variant | Encoder ms | Stage1 Flow ×2 ms | Stage2 PTv3 ×4 ms | Network Event ms | Generation Wall ms | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|
| V4 same-GPU baseline | 203.739 | 60.623 | 143.205 | 407.567 | 431.280 | 3.334 |
| Compile classification neck | 194.495 | 60.964 | 144.855 | 400.314 | 428.336 | 3.334 |

The candidate reduced Encoder Event by 9.244 ms, but the paired full-generation reduction was 2.944 ms with 7.772 ms paired SD and 3.406 ms 95% CI half-width; the interval crosses zero. Hence <400 ms was not achieved. The 350 ms target is not supported by these results. The 1.08M-point jitter was deliberately skipped in these V5 runs; V5 does not provide a new generation-plus-jitter result.

## Quality and parity

Classification-neck compilation changed Stage1 scores/candidate IDs on all 20 timed frames (mean candidate count +3.1; mean fine-occupancy symmetric difference 2,754.4 voxels). On the fixed 41-frame Seq08/skip100 set, metric changes relative to the V4 compiled-Flow reference were small: CD +0.029%; IoU@0.5 −0.000856 percentage points; IoU@0.2 −0.000757 pp; IoU@0.1 −0.000345 pp; JSD-BEV +0.000037; JSD-3D +0.000007. These pass the proposed aggregate screening thresholds, but do not compensate for the uncertain full-pipeline speed gain or establish exact parity.

## Component conclusions

- **Conv3d profile:** 48 Conv3d calls total 138.5–138.9 ms per Encoder on three real frames. Large 3×3×3, 128-channel, full-resolution convolutions dominate. Nsight Systems observed an Ampere `sm86_xmma...f16f16...f32` cuDNN kernel and layout transform kernels. Hardware utilization counters could not be read (`ERR_NVGPUCTRPERM`), so percentages for Tensor Core/SM occupancy/DRAM are unavailable.
- **Encoder compile:** classification neck saves about 9.2 ms in Encoder, but not a statistically established full generation gain. Other compiled targets were exact or nearly exact in smoke but showed no compelling speedup. Classification-neck cold compile was about 19.57 seconds in module smoke and about 14.55 seconds in the full benchmark process.
- **Conv–BN fusion:** rejected; slower by about 5.4 ms for all eligible layers and large feature deviations. No quality evaluation was needed after this failure.
- **Stage1 Flow CUDA Graph:** exact single-frame output; compiled graph replay 59.95 ms versus 60.62 ms V4 Flow mean, a negligible cross-test difference. Not adopted; no full-generation graph benchmark.
- **TensorRT:** not tested; TensorRT/Torch-TensorRT/ONNX Runtime/trtexec were absent.
- **cuDNN environment:** PyTorch was built against cuDNN 8.5.0 but resolved runtime 8.4.0. The experiment did not change the environment. This should be corrected in a separate controlled environment before kernel-level conclusions are generalized.
- **Nsight Compute:** attempted, but performance counters were restricted by `ERR_NVGPUCTRPERM`; no system permission changes were made.

## Evidence and reproduction

Detailed data lives under ignored local artifacts in `experiments/hs_inference_opt_v5/`; only code and this report are intended for Git. Key artifacts:

- Profile: `conv3d_profile/gpu2_3frames.json`, `conv3d_profile/ncu_top_conv.csv`
- V4 timing: `baseline/v4_compiled_flow_gpu2_round1_5x20/summary.json`
- Compile timing: `encoder_compile/classification_neck_gpu2_round1_5x20/summary.json`
- 41-frame metrics: `quality_41/classification_neck_metrics.json`
- BN fusion and graph smoke: respective `conv_bn_fusion/` and `cuda_graph/` JSONs

GPU tests ran on GPU2 only; no other job was terminated or preempted. 20/20 timed samples completed for both formal benchmark runs. Full 41-frame metric evaluation completed. No 408-frame evaluation was warranted because no V5 candidate demonstrated a robust speedup. No Stage2 training or model/checkpoint changes occurred.
