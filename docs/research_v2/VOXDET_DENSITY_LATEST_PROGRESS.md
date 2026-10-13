# VoxDet-Density + PointFlow 最新进展

**后续更新：Stage 1 已于2026-10-13 16:45完成25,000次成功更新。GPU2/3最终评测：density IoU45.04%，0.5m anchors IoU49.01%，CD0.28728m。完整最新结果见 [最终预算checkpoint评测](VOXDET_DENSITY_FINAL25000_EVALUATION.md)。下文保留14:22运行中快照。**

快照时间：**2026-10-13 14:22，Asia/Shanghai**。本报告对应正在运行的训练；进度与 ETA 会继续变化。机器可读记录：[progress_20261013_1422.json](voxel_point_reports/progress_20261013_1422.json)。

## Stage 1 正式训练

VoxDet-Density 正在物理 GPU **2、3** 上训练。每卡 batch 2，global batch 4，FP16 AMP；计划完成 **25,000 次成功 optimizer update / 100,000 次帧曝光**。

| 项目 | 快照值 |
| --- | ---: |
| 成功更新 | 16,455 / 25,000（65.82%） |
| 帧曝光 | 65,820 / 100,000 |
| 最近 500 次更新平均耗时 | 0.992 秒 / update |
| 预计剩余时间 | 约 2 小时 21 分钟 |
| 预计完成时间 | 当日 16:44 左右 |
| GPU2 / GPU3 实际显存占用 | 21,983 / 21,975 MiB |
| GPU2 / GPU3 峰值 allocated 显存 | 19.060 / 19.060 GiB |
| 最近 500 次更新梯度 | 全部有限 |
| AMP 跳过更新 | 6 次，双 rank 同步跳过 |
| 最新已保存 checkpoint | `checkpoint_016000.pt` |

训练目录：

```text
/data-12/M2024-HWZ/FPSGen/experiments/voxel_point/voxdet_density_formal_gpu23_b2_roi_20261013_094356
```

查看实时进度：`tmux attach -t voxdet_density_gpu23`。该 tmux 会话是只读监控；训练是独立的后台进程。

训练序列为 **00–07、09、10**；固定时间块 development split 与相邻帧 guard 划分后，训练 / development / guard 为 **17,870 / 900 / 360 帧**。seq08 不参与训练或超参数选择。按用户后续授权，Stage 1 对 GT 中少量 ROI 外点进行显式过滤并记录数量，以保留后的真实点数监督密度；不复制或夹紧 GT 点。生成点数始终为 180,000。

运行配置 SHA256：`a24d25fc7c32e56fdd4b3201ba823a8c3b2e181805aebdf756c55a90df496a77`。
划分 manifest SHA256：`c8916b2958c383d465f2979cde64adf9a48f14dd146efcf8f8d8c8b807360807`。

## 损失下降

下表为两张卡成功更新的训练 loss 均值，非验证 loss。最初 100 次更新平均为 **0.83334**；最近 500 次为 **0.28075**，下降约 **66.3%**。

| 更新区间 | 平均总 loss |
| --- | ---: |
| 1–1,000 | 0.49204 |
| 1,001–2,000 | 0.38234 |
| 2,001–4,000 | 0.35822 |
| 4,001–6,000 | 0.33767 |
| 6,001–8,000 | 0.32618 |
| 8,001–10,000 | 0.31390 |
| 10,001–12,000 | 0.30379 |
| 12,001–14,000 | 0.29343 |
| 14,001–16,000 | 0.28604 |
| 16,001–16,455 | 0.28088 |

损失为：

```text
L_density = L_pos + 0.1 L_bg + 0.5 L_coarse + 0.1 L_dist + 0.1 L_mass
L_total   = L_density + 0.1 L_dir + 0.3 L_aux
```

`L_pos/L_bg` 分别在有效非空 / 空体素上计算 log1p count SmoothL1；`L_coarse` 为 2×2×2 count 求和后的 log-count 一致性；`L_dist` 为归一化质量分布 TV；`L_mass` 为相对总质量误差；`L_dir` 为 occupied 连续段六方向偏移 L1；辅助 head 使用同一组密度损失。Stage 1 不使用语义 CE、扩散或 Flow Matching。

## 网络与空间合同

保留官方 VoxDet LiDAR tokenizer、3D ResNet、GeneralizedLSSFPN、SpatiallyDecoupledFPN、六方向几何引导及四层 VoxelAttentionAggregation；没有减小网络深度或宽度。上游固定 commit 为 `e10d8280de96d4f03f54b616e57e7c6754044fa4`。原结构参数量 **22,084,900**，密度适配后 **22,082,470**。

明确的适配包括：两个语义 head 改为单通道非负密度、监督改为密度及密度派生偏移、原 intensity 输入改为 normalized range、空间坐标与圆形 ROI 适配，以及 `create_grid` 中 grid_sample 轴顺序和体素中心归一化修正。输入仅取 XYZ，不使用 `input_[:,3]` 的语义标签。

训练网格为 **256×256×32 = 2,097,152 体素**，范围 `[-50,50]×[-50,50]×[-4,4.4]`，实际体素尺度 **0.390625×0.390625×0.2625 m**。圆形有效 mask 使用体素与半径 50 m 圆盘存在交集的规则；anchor 使用截断高斯采样，保持体素与圆形边界约束。

## 已完成的 seq08 中期评估

以下质量结果全部来自固定 **11,000-update checkpoint**，不是当前 16,455-update 权重。seq08 按 frame ID `0,100,…,4000` 固定抽取 **41 帧**；density occupancy 阈值固定为 1.0。测试没有用于挑选 loss 权重或阈值。

| 指标 | 结果 |
| --- | ---: |
| 原训练网格 density occupancy IoU，macro | **44.35%** |
| 原训练网格 density occupancy IoU，micro | 43.81% |
| 原训练网格 density occupancy F1，macro | 61.28% |
| density Mass-TV | 0.49240 |
| coarse density Mass-TV | 0.34970 |
| partial-input density IoU，macro | 12.36% |
| 180k anchors 的 0.5 m point occupancy IoU，macro | **47.33%** |
| 180k anchors 的 0.5 m point occupancy IoU，micro | 47.45% |
| 同一 anchors 在原训练网格的 point occupancy IoU，macro | 40.93% |
| partial-input 的 0.5 m point occupancy IoU，macro | 17.69% |

0.5 m 是额外评测分辨率，训练网格未变。point occupancy 使用真实生成点落入体素的二值占用，不使用 morphological closing；它与 density≥1 的定义不同。相同 anchors 的 point occupancy IoU 从原网格 40.93% 到 0.5 m 网格 47.33%，增加 **6.40 个百分点**。这些输出是 Gaussian anchors，尚未经过正式训练的 PointFlow。

详细协议与结果：[seq08 stride100](VOXDET_DENSITY_SEQ08_STRIDE100.md)、[0.5 m IoU](VOXDET_DENSITY_VOXEL05_IOU.md)、[step4000 五帧预览](VOXDET_DENSITY_STEP4000_PREVIEW.md)。

## Stage 2 与下一步

PWM 上游固定 commit 为 `c6e809486db3d0b6f8503318dfea246c06c7a50e`。PTv3 Mapping Teacher、sparse Sinkhorn refinement、PointFlow velocity FM 和 Euler 推理 adapter 已实现并完成初始 180k smoke；**Teacher 和 Student 尚未开始正式训练**。

初始 Teacher / Student 显存 smoke 分别约 9–9.85 / 5.46 GiB；这些是 mechanics smoke，不代表已经获得高质量映射。初始 20 个 GT-source OT 场景的 geometry gate 通过，但 balanced marginal gate 0/20，必须按非严格平衡几何 refinement 标注。初始未训练密度的 predicted-source 诊断曾拒绝 refinement 并 fallback Teacher，不能据此判断当前已训练 Stage 1 的实际表现。

正式 Stage 2 前仍需要：完成 Stage 1；生成训练集 predicted-density cache；对齐 ROI 过滤后可变 GT context 点数的数据合同，同时始终保留 180k source 与输出；混合 GT / predicted source 训练 Teacher；通过 endpoint 和 20→100 场景 OT 质量 gate；随后运行 PointFlow pilot。正式推理不加载 GT、Teacher 或 Sinkhorn，不传递 Stage 1 隐藏特征。

CPU 单元测试累计 **27 项通过**。初始完整 smoke / provenance 见 [首轮实现归档](VOXDET_DENSITY_POINTFLOW_RESULTS.md) 和 [源码审计](VOXDET_DENSITY_SOURCE_AUDIT.md)。本次上传仅包含源码、配置、测试、文档与小型 JSON 报告；数据、虚拟环境、checkpoint、点云及 compiled extensions 不上传。当前 Stage 1 训练继续运行，不自动启动其他长期训练队列。
