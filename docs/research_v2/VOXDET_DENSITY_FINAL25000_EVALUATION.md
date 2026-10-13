# VoxDet-Density：25,000-update 最终预算 checkpoint 评测

2026-10-13，使用物理 **GPU2、GPU3** 完成评测。Stage 1 在当日 **16:45** 完成 25,000 次成功更新 / 100,000 次帧曝光，训练约 **7 小时 1 分钟**。最新权重为 `checkpoint_025000.pt`；Teacher / PointFlow 尚未正式训练，本报告中的生成点云均为 **180,000 个 Gaussian voxel anchors**。

## 冻结评测协议

- 测试为 seq08 的固定 frame ID `0,100,…,4000`，共 **41 帧**；两卡分别运行 21 / 20 帧，完整性合并检查通过，无漏帧或重复帧。
- 最终预算 checkpoint25,000 与此前固定的 checkpoint11,000 均重新使用 **GPU FP32，关闭 TF32** 评测；不根据 seq08 选择 checkpoint、训练超参数或 occupancy 阈值。
- 原训练网格为 `256×256×32`，体素尺度 `0.390625×0.390625×0.2625 m`；density occupancy 阈值固定为 **pred_count≥1.0**，GT occupancy 为真实 count>0。
- 0.5 m 点云 IoU 使用 `floor(XYZ/0.5)`，任意点落入即占用，不使用 morphological closing。它与 density occupancy 定义不同。
- 两个 checkpoint 使用同一帧绑定 seed、CPU count allocation 与截断 Gaussian sampler，180k source 保持不变。GT 仅用于评测，不生成预测点源；Teacher、Sinkhorn、PointFlow 均未参与。
- Point KNN 使用 FP32 KeOps symbolic reduction，不构造 dense180k×180k 矩阵。Chamfer 为 `0.5×(mean nearest Euclidean pred→GT + GT→pred)`，单位米；F-score / coverage 距离阈值为 0.1 / 0.2 m。
- 以下为逐帧 **macro mean**；micro 指标和逐帧结果在 JSON / CSV 中单独保留。测试 GT 41 帧均为180k，无 ROI 外点过滤。

## 结果

| 指标 | 11,000 updates，同次 GPU 复测 | 25,000 updates |
| --- | ---: | ---: |
| Density occupancy IoU ↑ | 44.35% | **45.04%** |
| Density occupancy F1 ↑ | 61.28% | **61.92%** |
| Occupied-voxel count MAE ↓ | 1.25173 | 1.25189 |
| Log-count MAE ↓ | 0.05735 | **0.05360** |
| Density Mass-TV ↓ | 0.49240 | **0.47484** |
| Coarse density Mass-TV ↓ | 0.34970 | **0.33777** |
| Mass calibration relative error ↓ | 5.29% | **1.70%** |
| Mean predicted mass | 189,269 | **181,706** |
| Anchors point IoU，原训练网格 ↑ | 40.92% | **42.53%** |
| Anchors point IoU，0.5 m ↑ | 47.32% | **49.01%** |
| Anchors point F1，0.5 m ↑ | 64.13% | **65.67%** |
| Anchors Chamfer，m ↓ | 0.29540 | **0.28728** |
| Anchors F-score@10 cm ↑ | 14.33% | **15.18%** |
| Anchors F-score@20 cm ↑ | 58.13% | **60.18%** |
| GT coverage@10 cm ↑ | 13.86% | **14.60%** |
| GT coverage@20 cm ↑ | 63.04% | **63.95%** |
| Spacing mean，m | 0.18663 | 0.17918 |
| Spacing CV | 0.55497 | 0.55672 |
| Generated out-of-range ratio | 0% | 0% |

最终 density IoU 的 micro 值为 **44.36%**。Partial-input baseline 的 density IoU 为 **12.36%**，0.5 m point IoU 为 **17.69%**。

相比11k，density IoU 增加 **0.69 个百分点**，0.5 m anchors IoU 增加 **1.69 个百分点**，F-score@20 cm 增加 **2.05 个百分点**。改善较温和，总质量标定的改善更明显；occupied count MAE 几乎不变且略增。最终 log-count MAE 仍高于 partial-input baseline 的0.04935，不能用单一训练 loss 或只选改善指标证明密度回归已充分解决。

点源尚未完成表面恢复：F-score@10 cm 只有15.18%，coverage@20 cm 为63.95%，CD为28.73 cm。当前结果验证的是 Stage 1 点源质量；不能作为已训练 PointFlow 的最终质量。Spacing CV 基本不变，未在本次指标中表现出明显额外 collapse。

GPU11k复测与原CPU11k结果的 density IoU 差值约 **0.000018 个百分点**。小数值差异经 stochastic residual allocation 放大后，0.5 m anchor IoU 为47.323%（原CPU47.326%）；前后比较采用本次同设备精度的两组结果。

## 运行与数值核对

最终权重全部参数 / buffer 有限，所有密度与点指标有限。训练原始汇总 `status=FAIL_NONFINITE_GRADIENT` 是因为代码要求**所有尝试**梯度有限：总计25,011次尝试中有11次 AMP overflow，双rank同步跳过；实际25,000次成功更新的梯度全部有限。保留原始汇总，另存完成审计 [stage1_completed25000.json](voxel_point_reports/stage1_completed25000.json)。最近500成功更新训练 loss 均值为 **0.26587**。

评测峰值 allocated 显存约 **1.81 GiB / GPU**。除去每张卡首帧预热，最终checkpoint平均：

| 项目 | 时间 |
| --- | ---: |
| Stage 1 单次前向，GPU FP32 | 391.5 ms / frame |
| Density D2H + count allocation + Gaussian anchors，CPU | 173.7 ms / frame |
| 两项之和 | 565.1 ms / frame |
| Point geometry evaluation，非推理组成部分 | 269.3 ms / frame |

以上不包括输入读取 / 输出文件保存，也不包括尚未正式训练的 PointFlow。此数值不是完整两阶段推理延迟，也不是 FP16 吞吐 benchmark。

## 文件与后续

最终checkpoint SHA256：`0af3007573297c2f963b3302e35f59d6f2d28c3163a8b8a826509b82462474d6`。
最终评测展开配置 SHA256：`9ad4b8df0485677a69d3452b35225123472bc33f99d51c27672d22f30c342755`。
评测配置：[step25000](../../configs/research_v2/voxel_point/evaluate_stage1_seq08_step25000.yaml)、[step11000](../../configs/research_v2/voxel_point/evaluate_stage1_seq08_step11000.yaml)。

小型报告：[最终逐帧 JSON](voxel_point_reports/stage1_seq08_stride100_gpu23_step25000.json)、[11k GPU 对照 JSON](voxel_point_reports/stage1_seq08_stride100_gpu23_step11000.json)、[比较摘要](voxel_point_reports/stage1_final25000_comparison.json)。CPU单元测试 **28项通过**，包括不等长 shard 加权、point metric 合并及协议不一致拒绝。

本地完整 artifacts：

```text
/data-12/M2024-HWZ/FPSGen/experiments/voxel_point/voxdet_density_formal_gpu23_b2_roi_20261013_094356/seq08_stride100_final25000_gpu23_20261013_175804
```

下设 `step25000/merged/density_evaluation.json`、`per_frame_metrics.csv`，及两张卡保存的密度 / counts / 180k anchors。大型 artifacts 和 checkpoint 不提交。

Stage 2 仍需要对齐 ROI 过滤后的 GT context 合同、生成训练帧 predicted-density cache、正式训练 Mapping Teacher 并通过混合source endpoint / OT质量gate；本次评测未自动启动任何新的长期训练。
