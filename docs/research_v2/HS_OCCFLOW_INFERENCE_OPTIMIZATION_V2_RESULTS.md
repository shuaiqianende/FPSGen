# HS-OccFlow 推理优化 V2：阶段结果

## 实验身份

- 独立分支：`research/hs-occflow-inference-opt-v2`，基线 `98338a752c3810459c0d2f8678808ef8f37b9577`。
- GPU：物理 GPU2，RTX 3090；PyTorch 2.0.1+cu117 / CUDA 11.7。
- 权重、NFE、Stage1/Stage2 算法、候选阈值和几何均未改变：Stage1 EMA 2 NFE，Stage2 Online 4 NFE，阈值 0.5。
- Benchmark：相同 5 帧预热 + 20 帧正式计时；每帧重新执行模型；XYZ 输入预载入；没有计入磁盘读取、指标或文件写入。
- Stage1 / Stage2 checkpoint SHA256 分别为 `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2` 与 `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`。

## 精度审计

实际 dtype 见 [精度审计](HS_OCCFLOW_ACTUAL_PRECISION_AUDIT.md)。Stage1 Encoder 原路径已经包含 FP16 autocast，输出 Pyramid 为 `[FP32, FP16, FP16]`；Stage2 PTv3 核心明确使用 FP32，而输入投影与 Decoder 在外层 AMP 中执行。旧报告错误称 Encoder 为 FP32；历史性能数字未改写。

## 等价 GPU pipeline

新增 GPU 路径：

- Stage1 endpoint 上直接 GPU 阈值、球/AABB ROI、可选邻域扩展、预算规则和排序编码 Parent IDs。
- GPU float64 LiDAR voxelize，再用候选 ID searchsorted 构造 observed-child。
- GPU 直接从 `[M,8]` occupancy 解码 Fine XYZ，不经过 bitmask pack/unpack。
- Stage2 每个 Euler frame 预先计算静态坐标排序、位置/分数/LiDAR/observed 投影；每步仍重算当前 state、time、融合、完整 PTv3 和 Velocity Decoder。

Frame 100 真机审计：169,841 Parent；CPU/GPU Candidate IDs 和 Scores 完全相同，Observed Mask 完全相同；Stage2 4-step endpoint 最大/平均差均为 0，Occupancy 差异为 0，Fine 坐标与顺序完全相同。审计记录：`experiments/hs_inference_opt_v2/gpu_pipeline/frame00100_parity.json`。

静态条件缓存独立测试中，4-step wall median 从 407.57 ms 到 397.35 ms，约 1.026×；收益较小。PTv3 serialization / SparseConv 索引本轮没有跨步缓存，因为还没有安全可复用的 vendor 接口。

## 5+20 帧性能对比

所有时间单位为 ms/frame。旧基线数字来自已完成的同 GPU 5+20 报告；新结果为本分支完整 20 帧运行。

| 指标 | 原始路径 | GPU pipeline + 缓存 + CUDA jitter | 变化 |
|---|---:|---:|---:|
| 网络 Forward CUDA Event | 645.15 | 618.68 ± 61.81 | 1.043×，约快 4.1% |
| 完整生成 Wall Time | 838.86 | 631.39 ± 63.10 | 1.329×，约快 24.7% |
| Stage1 Encoder CUDA Event | 213.09 | 203.90 ± 0.69 | 原生 autocast 路径 |
| Stage1 Flow 2 NFE | 109.16 | 108.32 ± 0.38 | — |
| Candidate selection | — | 0.68 ± 0.02 CUDA Event；50.66 ± 0.27 CPU submit Wall | 旧/新阶段边界异步行为不同，不将 submit wall 当端到端 GPU 耗时 |
| Stage2 condition construction | 旧报告 130.55 CPU Wall | 5.08 ± 0.76 CUDA Event；4.82 ± 0.67 CPU submit Wall | 旧值含 CPU voxelize/传输同步；新值拆开报告，不把异步提交时间称作 GPU 总时延 |
| Stage2 静态条件投影缓存 | — | 2.69 ± 0.29 CUDA Event | 每帧一次，独立 single-frame endpoint parity exact；其开销被纳入完整生成 Wall |
| Stage2 4-step network | 322.89 | 306.47 ± 61.63 | 观察到约少 5.1%，单次配对并非独立归因实验 |
| Occupancy decode | 43.94 | 1.48 ± 0.23 | 新路径直接 GPU decode；旧值 CPU wall 与新值不同步边界，仅作诊断 |
| Peak allocated / reserved | 3.01 / 3.84 GiB | 3.01 / 3.84 GiB | 不变 |
| 原生 Fine jitter（1.08M 点） | 1764.69 | 398.03 ± 190.52 | 4.43×，约快 77.4% |
| 完整生成 + 1.08M jitter | 2603.55 | 1029.41 ± 226.26 | 2.53× |

新完整生成 median 为 630.73 ms、P90 703.60 ms、P95 726.50 ms；jitter median 为 311.56 ms、P90 665.01 ms、P95 680.34 ms。完整生成+jitter 的 median 为 982.54 ms、P95 1380.78 ms。20 帧峰值 allocated/reserved 分别为 3.007/3.838 GiB。

最终逐帧结果：`experiments/hs_inference_opt_v2/combined/hs_2x4_gpu_pipeline_cuda_jitter_warm5_eval20_final/{per_frame.csv,per_frame.jsonl,summary.json,protocol.json}`。20/20 帧原生 Fine Occupancy 坐标与历史 Baseline 保存结果完全一致；输出点数 20/20 精确为 1,080,000。正式生成速度改善不能用网络 CUDA Event 代替，表中两项分开。Candidate 与条件部分额外记录 GPU CUDA Event 和 CPU submit wall；由于动态 CUDA 输出和不同同步边界，二者不可互相替代，也不据其简单相减宣称阶段 speedup。

## Native Fine CUDA jitter

Sampler 在模型原生 `fine_size` 上对预测非空体素分配 exact-target 点数，使用各向异性 XYZ voxel size、GPU 双精度 ROI/索引检查及球面拒绝采样；GPU 随机 proposal 未解决的近切线样本只在极少尾项上用已有 CPU hit-and-run fallback，不裁剪坐标、不丢点。Frame 100 有 803,307 个占用 voxel，1,080,000 点约 0.72 s；warm 之后隔离 GPU/CPU 微基准记录约 0.72 s vs 2.51 s。5+20 run 中 20 帧平均回退 39.6 点，范围 9–86 点；每帧都精确得到目标点数，所有输出通过来源细体素与 ROI 检验。

注意 GPU jitter 与 CPU NumPy Seed 使用不同 RNG 实现，点坐标不会逐位相同。主优化 20 帧中平均 Occupancy 为 601,539、最大 803,307，低于 1.08M 预算，所以所有占用体素均被覆盖；中心点和 jitter 的原生 Fine support 不变。Frame 100 额外使用同一 RawMapSource GT 和统一物理网格 / JSD 实现，做了 3 个 CUDA seed 的完整点级诊断：中心点 CD 0.177932 m、IoU@0.5/0.2/0.1 为 50.305/36.701/14.975%；Jitter 平均 CD 0.174969 m（seed std 0.000017 m），IoU 分别 52.531/39.045/15.796%（最大 seed std 0.013/0.022/0.001 个百分点），JSD-BEV 0.273450±0.000246、JSD-3D 0.405739±0.000112。它仅是单帧结果，不能外推成全序列质量增益；408 帧 108 万点质量评测没有运行。

## 仍未测试 / 不纳入结论

| 优化 | 状态 |
|---|---|
| Stage1 `force_fp32` vs 原生 autocast | 单帧特征与耗时已审计；尚无 41 帧质量通过证据，不推荐切换 |
| PTv3 FP16 / selective mixed | Not tested；当前保持 FP32 |
| FP16 Pyramid grid_sample | Not tested |
| channels_last_3d、TF32/cuDNN benchmark | Not tested |
| torch.compile / CUDA Graph | Not tested |
| PTv3 serialization / sparse topology cross-NFE cache | 未实现 |
| 完整 41/408 帧指标重测 | 本轮未运行；等价 pipeline 已有 20/20 Exact Occupancy parity |

## 推荐当前配置

可采用 GPU candidate selection、GPU observed-child、GPU occupancy decode、Stage2 固定条件投影缓存和 CUDA jitter；保持当前权重、NFE、模型精度路径不变。要评估点级 CD/IoU/JSD 时仍须走原有统一指标 pipeline。本报告的数据只是工程性能结果，不是新质量提升结果。

## 复现命令

```bash
CUDA_VISIBLE_DEVICES=2 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python \
  -m pytest -q tests/test_hs_inference_fast.py

CUDA_VISIBLE_DEVICES=2 /data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python \
  scripts/research_v2/hier_occ_inference/benchmark_hs_patchscene_inference.py \
  --model hs_occflow --physical-gpu 2 \
  --data-root /data-12/M2024-HWZ/KITTI_Odometry \
  --config configs/research_v2/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes.yaml \
  --stage2-checkpoint /data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-384-stage2-score-gtparents/experiments/hier_occ_flow/stage2_384_real_score_gt_only_morph_100kframes/checkpoints/step025000.pt \
  --output experiments/hs_inference_opt_v2/combined/REPRODUCE_RUN \
  --gpu-candidate-selection --gpu-observed-mask --gpu-occupancy-decode \
  --stage2-static-cache --jitter-backend cuda \
  --parity-reference-dir /path/to/original/baseline/predictions --no-save-predictions
```

Benchmark 在 GPU admission 失败、已有计算进程、输出目录非空、checkpoint identity 不匹配或任一逐帧 Occupancy parity 失败时会退出。
