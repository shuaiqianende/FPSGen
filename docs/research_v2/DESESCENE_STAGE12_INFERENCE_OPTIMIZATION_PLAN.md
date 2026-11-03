# Stage1 / Stage2 后续推理优化：必做清单

本记录适用于当前 T1 深度矩 U1+A Stage1，以及正在训练的带／不带 LiDAR Encoder 注入的 Stage2。现有两组训练及其 41 帧质量评测不得因本计划中断或改变协议。以下是后续必须实施和验证的推理优化任务，不代表当前模型已经获得对应加速。

## 必做一：Stage1 Flow 编译

- 将当前 T1 Stage1 的 **Flow U-Net** 作为独立 `torch.compile` / Inductor 编译目标；不要把仅编译 LiDAR Encoder 的某个 Block 当作完成此项。
- 保持 Encoder、权重、原始 LiDAR、几何、Observed Mask、种子、Stage1 NFE=2、FP32 源噪声和 Euler 积分语义。沿用经过验证的 `no_grad` 兼容路径，并检查当前显式 FP16 和时间编码的 dtype。
- 先独立 Smoke 和数值检查，再做相同 GPU、相同真实帧、相同输入与种子的 Eager vs 编译配对测速；区分 Flow 核心与包含噪声／Mask／积分的完整 Stage1 计时。记录预热、首次编译耗时、图中断、重编译和显存。
- 历史 V7 编译 Flow ×2 为 **62.46 ms**；T1 最近诊断中 `sample` ×2 为 **105.78 ms**，只编译了 Encoder 第一个 Block。两者模型和计时边界不同，不能把差值直接认定为当前模型可追回的收益。

## 必做二：Stage2 PTv3 多层拓扑缓存

- 为当前两种 Stage2 模型分别接入并验证此前 `multilevel` 拓扑缓存：同一帧的四次 NFE 复用不随状态变化的序列化顺序、attention padding、pooling 索引；每步仍重新计算动态特征和 Flow，保持 NFE=4。
- 当前 Stage2 已物理去除 Score，其中一个方案还去除 LiDAR 特征分支；既有 inference-context 路径必须明确适配这两个接口，不能假设旧的有 Score 模型可直接运行，也不能静默回退。
- 比较缓存关／开在相同 Checkpoint、真实帧、Parent ID、随机源和 GPU 下的输出、数值差异、Candidate／Fine Occupancy 和 41 帧 CD、IoU、JSD，并记录 Stage2 四步和完整生成的 wall time、显存及缓存构建开销。无法保证正确性时保留无缓存参考路径。
- 历史 V7 Stage2 ×4 为 **97.64 ms**；T1 最近诊断缓存关闭时为 **178.23 ms**。这是不同模型和配置的历史结果，只能作为待验证线索，不能直接归因或预估当前收益。

## 必做三：条件准备与完整生成

- 接回并验证此前 GPU Candidate、Observed Mask 和 Occupancy Decode 路径，减少当前 CPU 往返；保持 ROI、坐标、体素索引、点数和排序语义。
- T1 诊断的 CPU Candidate／Mask／特征采样约 **157.73 ms**。必须单独计时并验证与原条件结果一致，不得将这部分时间从端到端数字中排除后与历史总时间相比。
- 训练完成并完成原计划的 seq08/skip100 41 帧评测后，在不抢占其他任务的空闲 RTX3090 上，对两组最终 EMA 分别进行同后端 5 帧预热 + 20 帧计时和必要的重复配对。报告 Encoder、Stage1 两步、条件准备、Stage2 四步、Occupancy Decode、完整生成、峰值显存、冷启动与逐帧差异；是否包含点数固定的 Jitter 必须注明。
- 推理采用 Stage1 预测 Parent；Stage2 不注入 Score；无 Encoder 注入组不接收 LiDAR Pyramid。质量评估不得采用 GT Parent 伪装成端到端效果。实验只使用 seq08/skip100 的 41 帧，不运行 900 帧。

## 来源

- V7 三轮配对和质量：[HS_OCCFLOW_INT8_V7_FINAL.md](../../../hs-occflow-int8-ptq-v7/docs/research_v2/HS_OCCFLOW_INT8_V7_FINAL.md)
- T1 GPU3 单轮分项诊断：[DESESCENE_T1_STAGE2_GPU3_COMPONENT_TIMING.md](../../../desescene-stage2-fp16-inductor-v1/docs/research_v2/DESESCENE_T1_STAGE2_GPU3_COMPONENT_TIMING.md)
- PTv3 缓存设计：[HS_OCCFLOW_PTV3_TOPOLOGY_CACHE_RESULTS.md](../../../hs-occflow-int8-ptq-v7/docs/research_v2/HS_OCCFLOW_PTV3_TOPOLOGY_CACHE_RESULTS.md)
