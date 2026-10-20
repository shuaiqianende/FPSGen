# HS-OccFlow 2+4 推理优化选项与审计清单

本文件供后续方案分析使用。它区分了已经测到的开销、代码中能看到的优化机会，以及尚未验证的想法。所有性能优化都需要先做数值/质量 parity，再在同一 GPU、相同帧和 5+20 计时协议下复测。

基准：[`HS_OCCFLOW_VS_PATCHSCENE_INFERENCE_SPEED_20FRAMES.md`](HS_OCCFLOW_VS_PATCHSCENE_INFERENCE_SPEED_20FRAMES.md)。物理 GPU2、RTX 3090、batch=1，Stage1 2 NFE、Stage2 4 NFE。完整生成均值 838.9 ms，网络 CUDA Event 累计 645.1 ms，整帧峰值 3.01 GiB allocated / 3.84 GiB reserved。

**精度审计更正：**Stage1 Encoder 的 `encode_features()` 内部已经启用 FP16 autocast，输出金字塔为混合 dtype（实测 `[FP32, FP16, FP16]`），因此不能把 Encoder 描述为 FP32，也不能将现有 autocast 当作新优化。Stage1 Flow 在 FP16 autocast 下运行。Stage2 的输入投影/融合与速度解码器在外层 autocast 中，但 PTv3 主干显式关闭 autocast并以 FP32 特征运行。详见 `HS_OCCFLOW_ACTUAL_PRECISION_AUDIT.md`。本文件下方原先“Encoder FP32→FP16”的建议应理解为对比 `original`、`force_fp32`、`autocast_fp16` 三种真实执行路径，而非从默认 FP32 开启半精度。

## 当前耗时构成

| 子项 | 均值 | 可观察到的实现特征 |
|---|---:|---|
| Stage1 Encoder | 213.1 ms | 每帧一次；条件特征供 Stage1 Flow 与 Stage2 共享 |
| Stage1 Flow | 109.2 ms | 两次串行速度场调用，约 54.6 ms/步 |
| 候选 Parent 选择 | 64.9 ms | Stage1 分数执行 `.cpu().numpy()` 后走 NumPy 选择 |
| Stage2 条件构造 | 130.5 ms | 多尺度 `grid_sample`，且 observed-child 计算把输入转 CPU/Numpy |
| Stage2 网络 | 322.9 ms | 四次串行 PTv3+解码器调用，约 80.7 ms/步 |
| Occupancy 解码 | 43.9 ms | 状态和 ID 转 CPU/Numpy 后展开 Fine Voxel |
| 完整生成 Wall Time | 838.9 ms | 包含上述生成步骤、同步与传输；不含磁盘 I/O、指标和保存 |

分段时间不是严格可加项：Network 项使用 CUDA Event；候选选择/条件/解码项含 CPU Wall Clock；部分 CUDA 工作是异步排队，且候选 `.cpu()` 会触发同步。优化前应增加一个**诊断计时模式**，在阶段边界分别记录 CUDA Event 和同步 Wall Time；正式性能基准则保留无额外阶段同步的端到端测量。

## 优先级 1：降低 CPU/GPU 往返

1. **GPU 上选择 Stage1 Parent。** 将阈值、有效 ROI mask、空间邻域扩张和 Parent ID 编码改为 GPU Tensor 操作。当前整张 score 体积先 D2H 再用 NumPy 筛选，随后候选 ID 又 H2D。目标是避免整张 score 的传输和强制同步，仅将最终候选 ID 保留在 GPU。
2. **GPU 上构造 observed-child mask。** 当前 `observed_children()` 将输入点转 CPU/Numpy、做 Fine Voxel 化与查询，再将结果复制回 GPU。把输入点 voxelize、Parent/Child 编码和候选 ID 查表实现为 GPU kernel/矢量 Tensor 操作；验证与现有 CPU 实现逐位一致。
3. **GPU 上完成 Occupancy 解码。** 避免 `ids.cpu().numpy()` 与 `occupancy.cpu().numpy()`；在 GPU 展开 bitmask/8 子格坐标，最后只在需要导出点云时复制输出。
4. **输入传输与流水线。** 若改成流式多帧吞吐，可用 pinned memory、`non_blocking=True` 并让 CPU 预处理下一帧与 GPU 推理当前帧重叠。这能提高序列吞吐，通常不降低单帧依赖链延迟。当前 18K 点输入很小，预期收益须实测。

## 优先级 2：降低重复的 Stage2 每步开销

1. **缓存静态 Parent 几何。** 同一帧四步使用相同 Parent IDs；缓存排序 `order/inverse`、整数网格坐标、中心坐标、归一化位置、batch 向量和 PTv3 serialization metadata。当前 `Stage2OccupancyModel.forward()` 每次都对 IDs `argsort`，重新算坐标/位置，并重建 Point 与 serialization。
2. **缓存固定条件投影。** Parent Score、LiDAR pyramid sample、observed mask、position 都在四个 NFE 中不变。将其投影成静态 token condition 一次；每步只计算动态 Local State 和 time embedding，再融合。需要改写投影层使 checkpoint 数学语义一致或做严格输出 parity；不要直接把完整 PTv3 特征缓存，因为 state 每步变化。
3. **PTv3 结构复用。** 同一批 Parent token 的稀疏拓扑/序列化索引固定。考虑将“构造 point/sparsify/serialization/pooling index”与“特征计算”拆开，让四步共享索引。先 profiling 确认这些确实在每步重复且耗时明显，再改 vendor adapter。
4. **算子融合。** 将 local state、time embedding 和静态条件的多个 Linear/concat/fuse 尽可能合成较少 GEMM，减少中间 Tensor 和 kernel launch。确认新旧网络输出误差、耗时和显存。

## 优先级 3：混合精度和 GPU 执行优化

1. **Stage1 Encoder 精度对照。** 默认 `original` 已为 FP16 autocast；比较 `force_fp32` 与 `autocast_fp16`，并确认 `original` 与 autocast 参考路径一致。对比金字塔、Stage1 scores、候选 Parent、Fine Occupancy、CD/IoU及耗时。若原始与 autocast 一致，不得将其报告为新增优化。
2. **Stage2 条件采样精度。** 当前 `sample_pyramid()` 对每层 `feature.float()` 后调用 3D `grid_sample`，最终再 `.half()`。测 PyTorch/CUDA 对 half grid_sample 的支持和误差；若半精度路径可用，可减少 FP32 临时拷贝/带宽。另可把三层特征在通道维合并后单次采样，再按通道切分，降低三次 launch，但需对比内存峰值和采样 parity。
3. **`channels_last_3d`。** 对 VoxDet 3D Conv 与 Stage1 Flow U-Net 分别测试 `torch.channels_last_3d`，配合 FP16/cuDNN；检查所有插值、拼接、GroupNorm、自定义模块是否触发昂贵 layout 转换。只保留端到端确实变快的格式。
4. **cuDNN benchmark / TF32。** 固定 shape 的 Conv3d 可试 `torch.backends.cudnn.benchmark=True`；若保留 FP32 Encoder，可测 TF32。每项单独开关，检查精度及稳定性。
5. **`torch.compile` / Inductor。** 可尝试编译规则较固定的 VoxDet Encoder、Stage1 U-Net 或 Stage2 MLP 子块。PTv3 的动态 token 数、Point 对象、SpConv/自定义 sparse ops 可能造成 graph break 或不支持；编译时间和重编译成本必须摊开报告。PyTorch 2.0.1+cu117 环境下需单独验证 Triton/CUDA 兼容，不预设会更快。
6. **CUDA Graph。** 固定帧内 Stage2 Parent token 数和 shape，可尝试 capture 后重放四步；但每帧候选数变化、动态 Point/SpConv 对象及内存地址约束会增加实现复杂度。可按 token 数 bucket/pad，但 padding 必须从注意力、归一化和 PTv3 序列化中正确屏蔽，且不能影响预测。CUDA Graph 适合减少 CPU launch 开销，不能消除 PTv3 的数学计算。
7. **权重驻留与预分配。** 模型已常驻 GPU；可进一步复用状态/速度/条件 buffers，避免每步重新分配。检查 allocator 与 launch profile 后再决定。

## 优先级 4：改变计算预算（会改变方法设置）

1. **降低 NFE。** Stage1 从 2→1 可省约一次 Stage1 Flow；Stage2 从 4→2 可省约两次 Stage2 网络调用（按当前均值约 161 ms CUDA Event 工作量）。质量可能下降，必须在固定 development/Seq08 指标上画质量-延迟曲线，不作为纯工程等价优化。
2. **候选 Parent 数量。** 当前平均约 170K 候选。调高 Stage1 阈值或引入 token budget 会降低 Stage2 成本，但会丢失召回、改变最终 Occupancy。需报告 Candidate Recall/FN ceiling 和完整质量，不可静默截断。
3. **缩小 PTv3 / 分块推理。** 降低宽度、patch size 或层数可加速，但会改变模型容量；空间分块需要 halo 并检查边界连续性，不能假设与全场景推理等价。
4. **Batching。** 批量更适合多场景吞吐。主报告的 batch=1 单帧延迟不能用 batch=4 摊销时间替代；还要处理不同 Parent 数的 ragged packing 与跨帧归一化影响。

## 优先级 5：108 万点输出的独立优化

目前 CPU `native_fine_jitter` 平均约 1.765 s，甚至长于模型生成。可能优化：GPU 向量化体素内 uniform offset、在体素内用 GPU ROI rejection/refill、并行生成每格 count offsets、减少 Python/NumPy 大数组拼接。必须维持原生 Fine Voxel、球形/盒形 ROI、exact 1,080,000 数量、固定 seed 和每点所在源体素合法性。Jitter 优化只改善“生成+输出点云”耗时，不改善模型 Forward。

## 推荐实验顺序

1. 修正并细分阶段计时（同步诊断版 + 原始异步端到端版）。
2. 先实现 GPU candidate selection、GPU observed mask、GPU occupancy decode，并逐帧验证预测一致。
3. 独立测试 Stage1 Encoder autocast FP16；确认候选与指标变化，再纳入端到端速度测量。
4. 缓存 Stage2 固定几何/条件/serialization；四步输出做 parity。
5. 分别评估 `channels_last_3d`、融合 grid_sample、cuDNN/TF32。
6. 最后才尝试 compile/CUDA Graph，并记录捕获/编译启动成本。
7. 将 NFE、候选预算、网络宽度作为质量-速度消融，不与等价工程优化混为一谈。
8. 如关心完整 1.08M PLY 交付速度，另行优化与测量 native jitter。

## 明确不应误解的点

- 当前模型总峰值显存只有约 3 GiB allocated；单纯“多占显存”不会自动让单帧更快。
- Encoder 是 Stage1 与 Stage2 共享的单次条件编码，不应重复运行。
- 四个 Stage2 Euler 步是串行状态依赖，不能把四个速度调用直接并行 batch 化。
- 每帧之间可并行或流水化以提高吞吐，但不等于单帧延迟下降。
- 任何精度、NFE、候选阈值、网络容量变化都需在报告中作为不同设置。
- 本文所有候选项尚未宣称已实现或获得加速。
