# HS-OccFlow 与 PatchScene 推理速度对比（5 帧预热 + 20 帧计时）

本次实测已完成。两模型在物理 GPU2（RTX 3090）、batch=1 上顺序运行；各自先用 5 个真实帧预热，再对相同的 20 个 seq08 帧计时。两份逐帧记录中的输入文件 SHA256 和 XYZ SHA256 均逐项一致。PatchScene 使用官方 Ring_Spread_V2、10 个扩散步、官方 EMA checkpoint；HS-OccFlow 使用 Stage1 2 NFE（EMA）和 Stage2 4 NFE（Online，step 25000）。

## 计时结果

单位均为毫秒。表中格式为 **mean ± std / median / P90 / P95**，统计基于 20 个正式计时帧。

| 指标 | HS-OccFlow 2+4 | PatchScene 10 steps |
|---|---:|---:|
| 纯网络 Forward CUDA Event | 645.15 ± 61.02 / 654.48 / 719.00 / 727.03 | 15264.35 ± 6.29 / 15265.29 / 15271.33 / 15274.63 |
| 完整生成 Wall Time | 838.86 ± 112.83 / 855.18 / 964.34 / 970.72 | 24009.43 ± 1032.05 / 23869.77 / 25214.89 / 25549.50 |
| 原生体素中心点输出 | 23.75 ± 6.46 / 24.88 / 30.18 / 33.27 | 8.11 ± 3.05 / 7.90 / 11.24 / 14.18 |
| 原生体素内 1.08M Jitter | 1764.69 ± 428.03 / 1762.95 / 2268.35 / 2405.94 | 1289.43 ± 411.52 / 1160.11 / 1893.09 / 1916.39 |
| 生成 + 1.08M Jitter | 2603.55 ± 495.16 / 2640.04 / 3199.77 / 3299.51 | 25298.86 ± 1336.25 / 25253.88 / 27096.77 / 27256.77 |
| Peak allocated | 3.01 GiB | 8.92 GiB |
| Peak reserved | 3.84 GiB | 14.59 GiB |

PatchScene 相对 HS-OccFlow 的平均时延比：网络 CUDA Event **23.66×**，完整生成 **28.62×**，生成后再产生 1.08M 点 **9.72×**。这是本次配置下的实测比值；NFE/扩散步数与模型结构不同，不代表等计算预算比较。

## 网络调用与分阶段开销

- HS-OccFlow 每帧运行一次共享 VoxDet Encoder、Stage1 Flow 2 次、Stage2 模型 4 次：6 次生成场网络评估，另有 1 次条件 Encoder 调用（总计 7 次学习模块前向）。Stage2 复用同一次 Encoder 的多尺度特征。
- PatchScene 每帧有 7 个 Ring，每个 Ring 执行 10 步，实际 U-Net 前向为 **70 次/帧**（本轮共 1,400 次）。
- HS-OccFlow 平均 Encoder CUDA Event 213.09 ms；Stage1 Flow 两步合计 109.16 ms；Stage2 条件构造 130.55 ms；Stage2 网络四次合计 322.89 ms。
- PatchScene 平均输入体素化 64.74 ms、Ring 准备 1071.91 ms、U-Net CUDA Event 累计 15264.35 ms；Ring 采样、融合及 CPU/GPU 搬运等其余环节约 6904.51 ms。

完整生成 Wall Time 从已预载入 CPU 的 XYZ 开始，到模型原生 Occupancy 生成结束；不含磁盘读取、点云指标、可视化和文件写入。网络 Forward 使用累计 CUDA Event，故与包括 CPU 调度及数据搬运的 Wall Time 不同。Jitter 单独按 CPU Wall Time 计时，默认 exact_target，20/20 帧均恰好输出 1,080,000 点。

## 实验身份

- 帧：预热 `[0, 800, 1600, 2400, 3200]`；计时 `[100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500, 3700, 3900]`。
- GPU：物理 GPU2，NVIDIA GeForce RTX 3090；PyTorch `2.0.1+cu117`，CUDA `11.7`。
- HS Stage1 SHA256：`b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`；Stage2 SHA256：`4279a1f334cb924b3d9b12a5b6ccafb482f25016b0e1266bbd6ef838e6fdfafe`。
- PatchScene 上游 Commit：`49de6d848663956e57cdd2f95d3b364e7a7afbe6`；EMA 权重 SHA256：`f50bf65f419c32bb40afa14e30440e1c4ea839dd92c678f8b4487e0d2b36140a`。
- 精度审计更正：Stage1 Encoder 调用内部启用 FP16 autocast；其三层条件金字塔实际输出为混合 dtype（FP32、FP16、FP16）。Stage1 Flow 在 FP16 autocast 中运行；Stage2 输入投影/融合与速度解码器处于外层 autocast，而 PTv3 主干显式禁用 autocast、输入转 FP32。旧报告中的“Encoder 实际为 FP32”描述有误；本文原始时延数值未作修改。
- 两种方法的平均原生占用体素数：HS 601,539；PatchScene 297,220。
- PatchScene Ring_Spread_V2 在帧间独立；没有跨帧状态条件。

## 产物与限制

完整逐帧记录和原始模型协议保存在 [HS 输出目录](../../experiments/inference_benchmark/hs_occflow_2x4_gpu2_warm5_eval20/) 与 [PatchScene 输出目录](../../experiments/inference_benchmark/patchscene_10step_gpu2_warm5_eval20/)。配对汇总位于 [comparison summary](../../experiments/inference_benchmark/comparison_gpu2_warm5_eval20/summary.json) 和 [逐帧配对 CSV](../../experiments/inference_benchmark/comparison_gpu2_warm5_eval20/per_frame.csv)。PatchScene 20 帧原生 Occupancy 预测已保存，可在不重新执行模型的情况下做后续质量评估。

这是 5+20 帧性能测试，不是质量评测；本报告不据此比较 CD、IoU 或 JSD。Jitter 时间显著受 CPU 拒绝采样/数组生成影响，且此次 HS 与 PatchScene 原生占用格数不同，因此不能把该耗时视为神经网络速度。
