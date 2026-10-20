# HS-OccFlow 实际推理精度审计

## 审计身份

- 基线代码：`98338a752c3810459c0d2f8678808ef8f37b9577`，优化分支 `research/hs-occflow-inference-opt-v2`。
- GPU：物理 GPU2，NVIDIA GeForce RTX 3090。
- 软件：PyTorch `2.0.1+cu117`，CUDA `11.7`。
- 输入：SemanticKITTI seq08 frame 100；Stage1 EMA SHA256 `b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`；Stage2 Online SHA256 `4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`。
- 运行审计：`experiments/hs_inference_opt_v2/precision_audit/frame00100_modes_warmed.json`。Forward hooks 记录叶子模块输入、输出和权重 dtype；每种 Encoder 精度先预热一次，再取 3 个 CUDA Event 样本的中位数。

## 结论

旧速度报告曾写 Stage1 Encoder 在 FP32 下执行；这是不正确的。`FrozenStage1OccupancyAdapter.encode_features()` 在 [stage1_occ_adapter.py](../../fpsgen/hier_occ_flow/stage1_occ_adapter.py) 的 74–81 行中明确对 `encode_condition()` 包含 CUDA FP16 autocast。运行时金字塔输出是 `[FP32, FP16, FP16]`。因此，现有 `original` 已经是 Encoder autocast 路径，不能把再次打开 autocast 当成新优化。

Stage1 Encoder 参数仍为 FP32；autocast 决定算子激活/运算 dtype，不会把模型权重永久转换成 FP16。运行 hook 观察到 Conv/Linear 等 autocast 支持算子使用半精度激活；几何索引、scatter/reduction 及部分归一化/融合输出仍可见 FP32。

Stage1 Flow 在外层 FP16 autocast 中执行。其模块输出同时出现 FP16 与 FP32，主要因归一化、运算组合及状态更新仍采用 FP32。

Stage2 也不能概括为“全 FP16”。Benchmark 在 Stage2 外层启用 autocast，所以 position/score/LiDAR/observed/fusion 投影和最终 velocity decoder 的 Linear 使用 FP16 输出；其权重为 FP32，LayerNorm 可输出 FP32。相反，[stage2_model.py](../../fpsgen/hier_occ_flow/stage2_model.py) 第 135–146 行将 PTv3 特征转成 FP32，并在第 144–145 行显式关闭 autocast执行 `sparsify / embedding / encoder / decoder`；这段 PTv3 主干运行 FP32。实际 Stage2 velocity 输出为 FP16，积分时使用 `velocity.float()`。

## 精度模式实测

Frame 100 同一模型权重、输入、随机种子，比较当前 `original`、强制 FP32 Encoder、显式 FP16 autocast。时延每种模式均在该模式的一次预热后测 3 次：

| Encoder 模式 | Encoder CUDA Event 中位数 | Pyramid dtype | 相对 original 的最大绝对特征差 | 平均绝对特征差 |
|---|---:|---|---:|---:|
| `original` | 204.42 ms | FP32 / FP16 / FP16 | 0 | 0 |
| `force_fp32` | 314.85 ms | FP32 / FP32 / FP32 | 0.017834 | 0.000197（各层 mean-abs 的平均） |
| `autocast_fp16` | 204.30 ms | FP32 / FP16 / FP16 | 0 | 0 |

用相同噪声种子和 Stage1 2-NFE Flow 比较 endpoint：`original` 与 `autocast_fp16` score 逐元素一致，Candidate ID 一致（169,841 个）；`force_fp32` score mean/max abs difference 为 `0.0002186 / 0.0063477`，候选数 169,850，和原始路径有 9 个 ID 不同。因此强制 FP32 属于会改变生成支持的精度变体，尚未通过完整几何质量门槛。

`original` 与显式 `autocast_fp16` 在当前调用位置得到逐元素完全相同的三个特征图；它们是同一路径，不能作为独立精度收益。Frame 100 的原始和 autocast Encoder 计算分别约 204 ms，强制 FP32 约 315 ms。该比较是单帧微基准，不能外推为完整 20 帧收益；FP32 对特征的变化也尚未完成 41 帧几何质量评估。

## 之前 5+20 帧性能协议的更正

仅修正文档中的 dtype 描述，原先记录的计时数值没有改动：旧 20 帧结果仍是完整生成 838.86 ms/frame、网络 CUDA Event 645.15 ms/frame、CPU native jitter 1764.69 ms/frame。原 `protocol.json` 是当时错误的 metadata，本报告作为可审计更正；原文件未被覆盖，以保留已保存实验的原始记录。

## 限制

- 该审计没有证明 `force_fp32` 会改善或保持最终 Occupancy/CD；在完成质量回归前，默认仍使用 `original`。
- PTv3 AMP、半精度 `grid_sample`、`channels_last_3d`、TF32、compile 和 CUDA Graph 仍未启用，也未计入后续优化时延。
- Hook 只在独立诊断运行中使用，不用于正式 5+20 计时。
