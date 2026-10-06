# Stage-1 BEVFlow：完整08序列最终测评

完成时间：2026-10-06 15:01:54（北京时间）。状态：**六组测评全部 COMPLETE**。
本报告替代本轮原定的 B200×3 generation seeds 最终汇总，不改写历史 B20/B100/B200 记录。

## 1. 实验范围与统一协议

- 方法：FPSGen Legacy BEVFlow、SiD2-style Residual U-ViT-S-BEV、DiP-S/16-BEV。
- 三个方法均使用完成500 epoch、global step299000的最终 checkpoint；不按测试分数挑选 checkpoint。
- SemanticKITTI sequence08 的全部 **4071个可用 input/GT 配对帧**，stride1；没有筛掉困难帧或失败帧。
- 每一帧、每个方法均使用相同 Gaussian **seed42**，不是按帧编号递增 seed，也不是三 seed。
- LiDAR-only `100`；vehicle/road 为显式零图，无额外语义条件。
- GT 为 `gt_possion`，180000点；不是另一套完整原始点云 GT。
- Euler10，分别测 CFG1、CFG2；物理 GPU3。
- dense FP16；geometry、Euler state、GT、metrics 保持 FP32。
- **CFG1：所有三个方法统一 eager；CFG2：所有三个方法统一 Inductor。**

这是 **Stage-1 BEV 生成指标**，不包含 PointFlow 或最终点云补全指标，也不是本轮新跑的 condition-usage 测评。
DiC 此前暂缓，没有本轮最终结果，不能补填到比较表。

## 2. 完整序列生成质量

以下是4071帧逐帧指标的均值。Height 为 GT-occupied cell 上的 MAE，单位米。

| 方法 | CFG | IoU ↑ | Occupancy F1 ↑ | Completion F1 ↑ | Height MAE(m) ↓ | Density Mass-TV ↓ |
|---|---:|---:|---:|---:|---:|---:|
| FPSGen | 1 | 0.783077 | 0.876426 | 0.835700 | 1.086834 | 0.340377 |
| SiD2-style | 1 | 0.770553 | 0.868534 | 0.824833 | 1.147833 | 0.352273 |
| DiP-S/16 | 1 | 0.780767 | 0.875134 | 0.835291 | 1.066958 | 0.339898 |
| FPSGen | 2 | 0.785946 | 0.878180 | 0.838093 | 1.048968 | 0.329436 |
| SiD2-style | 2 | 0.776974 | 0.872550 | 0.829988 | 1.104831 | 0.335235 |
| DiP-S/16 | 2 | 0.787033 | 0.879102 | 0.839896 | 1.044345 | 0.328739 |

精确数值、各指标 frame standard deviation、backend 和协议字段：
[final_quality_full_seq08_seed42.csv](results/full_seq08_seed42_20261006/final_quality_full_seq08_seed42.csv)。
frame std 描述场景差异，不能当成多训练 seed 的稳定性误差。

## 3. 参数与实际末段训练损失

末500/200步损失来自实际每步累计记录，不是低频 TensorBoard scalar 的平均。
参数包含 PointPillar；DiP total 包含98304个冻结位置编码参数。

| 方法 | Core | Condition | PointPillar | Total | 最后500步 FM loss | 最后200步 FM loss |
|---|---:|---:|---:|---:|---:|---:|
| FPSGen | 21498179 | 2032736 | 1600 | 23532515 | 0.081481 | 0.081484 |
| SiD2-style | 44114188 | 360968 | 1600 | 44476756 | 0.089036 | 0.089300 |
| DiP-S/16 | 39038467 | 3489794 | 1600 | 42529861 | 0.100232 | 0.099600 |

训练协议限制必须保留：

- FPSGen、SiD2：500 epoch 为 DDP2×B16、global batch32。
- DiP：前50 epoch 为 DDP2×B16，后450 epoch 按用户要求迁移为 GPU3×B32。
  global batch32、已恢复的 optimizer/LR schedule 保持，但 world size、数据顺序和未序列化的私有 RNG stream 改变。
  **不能把 DiP 描述为全程相同双卡 DDP 或 bitwise 连续训练。**
- FPSGen 原始 rank manifest、完整 GradScaler skip 计数和训练 wall time 缺失，保持 unknown。
- SiD2 wall time100880.1秒（约28.02小时），两个 rank 各记录113次 skipped update。
- DiP 单卡续训段 wall time130052.5秒（约36.13小时），该段记录99次 skipped update；不是整段500 epoch的总耗时或总跳步数。
- 现有 peak memory 是训练 manifest 的各自统计窗口值，不能无条件当作同一硬件/执行方式下的显存排名。

机器可读的参数、loss、wall time、memory 和限制：
[training_and_parameters.csv](results/full_seq08_seed42_20261006/training_and_parameters.csv)。

## 4. Paired bootstrap

每个方法使用相同4071帧和 seed42，按帧进行配对 bootstrap：10000次，bootstrap seed20261001。
下面 delta 均为 **FPSGen − baseline**。IoU/Completion 正值对 FPSGen 有利；Height/Mass-TV 负值有利。

| CFG | 对比 | IoU delta [95% CI] | Completion F1 delta [95% CI] |
|---|---|---|---|
| 1 | FPSGen − SiD2 | +0.012524 [0.011910, 0.013130] | +0.010867 [0.010323, 0.011408] |
| 1 | FPSGen − DiP | +0.002310 [0.001710, 0.002911] | +0.000409 [-0.000115, 0.000936] |
| 2 | FPSGen − SiD2 | +0.008971 [0.008293, 0.009648] | +0.008105 [0.007498, 0.008699] |
| 2 | FPSGen − DiP | -0.001087 [-0.001757, -0.000411] | -0.001803 [-0.002384, -0.001211] |

**解释限制**：连续帧存在时间相关；这些是条件于一个 sequence、一个训练 seed 和一个 generation seed 的描述性 frame-level CI。
不能直接当作跨训练 seed、独立道路场景或所有生成噪声的泛化显著性。
完整五个指标的区间与方向：
[paired_bootstrap_full_seq08_seed42.json](results/full_seq08_seed42_20261006/paired_bootstrap_full_seq08_seed42.json)。

## 5. 数值异常与恢复

CFG2 三个方法完整完成后，FPSGen 的 CFG1 FP16+Inductor 前向出现 NaN。
失败重试定位到000002/000003帧；捕获的000003帧在t=0第一次前向就产生非有限值，输入为有限值。
独立同步诊断、同输入 eager/compiled 重放未必复现，确切底层根因仍未确定。

一个独立 `baddbmm(beta=0)` Inductor 最小错误已复现，但其局部修正没有修复实际测评，
因此是**被否定的本次有效修复假设**，未用于最终结果。没有升级 PyTorch/CUDA 或修改环境安装。

最终只把所有方法的 CFG1 dense backend 显式改为 FP16 eager，未改变精度、GT、seed、sampler、CFG或权重。
完成结果先校验复用，失败日志与输出归档后再重跑未完成阶段。
三组 CFG1 最终均完成4071帧并通过同帧/seed/finite校验；CFG2 原始完成结果保持不变。

详见 [恢复记录](FULL_SEQ08_SEED42_EVALUATION_RECOVERY.md) 和
[backend exception](results/full_seq08_seed42_20261006/cfg1_backend_exception.json)。
不能把 CFG1/CFG2 差异全部归因为 guidance，因为两个设置同时采用了不同推理 backend。

## 6. 证据支持的结论

1. 本次完整序列下，FPSGen 在两种 CFG 的五个质量均值都优于 SiD2-style。
2. CFG1 下 FPSGen 的 IoU 高于 DiP，但 Completion F1 差异很小且 frame CI 包含0；DiP Height 更好。
3. CFG2 下 DiP 五个均值略优于 FPSGen：IoU +0.109个百分点、Completion F1 +0.180个百分点。
   不能声称 FPSGen 全面领先，也不能在上述训练/统计限制下宣称 DiP 架构普遍优越。
4. FPSGen 参数更少、FM loss 更低，但最低训练损失不保证最佳生成质量。
5. 当前结论只涉及本次 BEV 质量；不把历史 condition-usage 数值或缺失的推理速度移入本表。

## 7. 结果文件与复现入口

Git 中只保存小型汇总和 provenance，不上传 checkpoint、4071帧逐帧CSV、生成图像或大日志。

- [完整报告JSON](results/full_seq08_seed42_20261006/final_report.json)
- [原始campaign manifest](results/full_seq08_seed42_20261006/campaign_manifest.json)
- [最终COMPLETE状态](results/full_seq08_seed42_20261006/queue_status.json)
- [校验与checkpoint指纹](results/full_seq08_seed42_20261006/evaluation_provenance.json)
- 本机原始结果：`outputs/research_v2/formal_bev_compare/full_seq08_seed42_gpu3_20261006/`。
- [完整序列队列与校验](../../scripts/evaluation/evaluate_full_seq08_seed42.py)、
  [原测评器](../../scripts/eval_bevflow_lidar_only.py)。
  CLI：`CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/evaluate_full_seq08_seed42.py --output outputs/<fresh-root> --cfg1-dense-backend eager`。
  需要本机数据、最终checkpoint和记录的环境；小型汇总本身不能重建全部原始点云。

源码基线为`e4aa153`，测评时部分 orchestration 为未提交文件；本次提交补充这些文件。
SiD2-style 是 paper-based scaled implementation，不是官方 SiD2 源码复现。
