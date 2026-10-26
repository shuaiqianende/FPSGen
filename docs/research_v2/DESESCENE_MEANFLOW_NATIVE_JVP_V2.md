# MeanFlow native JVP bridges v2

2026-10-26。独立分支 `research/desescene-meanflow-replay-v2`。原 v1 队列、EF0 评价及其源码未修改。

## 已实现并实际验证

- SpConv SubMConv3d：原生 SpConv 计算 primal；固定当前整数索引，以相同权重、同一原生卷积且 bias=None 计算特征切向量。仅在 detached JVP 内启用；正常反向仍执行原 SpConv。每次使用独立 SparseConvTensor 和 indice_dict 容器，不复用上一 Flow Step 的输出。
- FlashAttention：primal 仍为安装环境的原生 `flash_attn_varlen_qkvpacked_func`，保持 FP16 QKV、scale、序列边界、padding、无 causal/RPE/dropout 的原配置。切向量按 `dP=P*(dS-sum(P*dS))`、`dY=dP*V+P*dV` 解析计算，统计/矩阵计算用 FP32，窗口分块限制临时显存。它是相同 attention 函数的解析 JVP 数值实现，未声称与 CUDA backward 逐位相同。其他 Attention 配置明确拒绝。
- Pooling max：primal 和 argmax 来自原生 torch_scatter，切向量沿原 argmax gather，保留并列值的原生子梯度规则；每次重新计算动态 projection/max，不缓存特征。
- 三个桥接只在 `torch.func.jvp` 的目标计算中启用，不修改 vendor 文件和正常训练反向，不使用有限差分目标。

CPU 测试 **27 passed / 4 GPU skipped**；GPU3 算子与 AMP 回归测试 **6 passed**。Flash JVP 通过 FP64 ragged-window 参考、原生 reverse AD 内积检查；SpConv FP32/FP16 通过原生 primal 逐位等价及 reverse AD 内积检查；Pooling 含并列值和空段。

## 修复实际梯度阻断

完整反向检查发现：同一外部 AMP 作用域内，no_grad 拓扑预热生成的权重 cast 会被后续训练前向复用，导致 time/interval 分支输出没有梯度。预热和 detached JVP 现在继承原 autocast 精度，但关闭各自的 weight cache；正常训练前向保持原 AMP。已添加 CUDA AMP 梯度回归测试，并在实际最大 Batch2 证明 interval 最终投影得到有限非零梯度和参数更新。

## 真实最大 Batch2 GPU 验收

物理 RTX3090 GPU3，原始训练/Development census 17,870/900，无 Parent 截断：seq01/frame1019=192,179；frame1018=191,976，总 **384,155 Parents**。

| 检查 | 实测 |
|---|---|
| 三个 interval 的 Step0 源前向 | 全部 bitwise exact |
| 缓存前向 / Parent 重排 | bitwise exact |
| 九个实际组件 JVP | 9/9 PASS |
| 完整 state/time、固定 endpoint JVP | 有限非零，max abs=125.0625 |
| 全模型 native reverse / JVP 内积 | normalized error=5.47647e-6，门槛3e-4 |
| 完整 Batch2 backward / optimizer update | PASS |
| interval 最终投影梯度 | weight max abs约0.002532；bias约0.003962 |
| Frozen Stage1 参数/状态 | 保持一致 |
| 峰值 allocated | 9,229,540,352 bytes，约8.60 GiB |
| 峰值 reserved | 10,791,944,192 bytes，约10.05 GiB |

全模型随机内积的原始 lhs=17.7185395、rhs=16.4132920；报告的 normalized error 使用两个方向向量范数归一化，不能解释为 scalar 相对误差或 bitwise derivative。FP16 Flash 的反向与 FP32 解析切向量有数值差异，具体单算子 FP64 参考测试另行约束。

FD 仅用于诊断，AMP/Max 的小步长 FD 不作为训练目标和通过依据。训练仍使用完整解析/native JVP。

## Native numerical replay 的明确边界

版本化误差政策写在 `meanflow_replay_bounds.py`，在本轮 GPU gate 前固定。同一张 GPU、相同源权重、同一最大 Batch2 先执行**独立原始 FM 更新恢复对照**，再执行 MeanFlow；两者都必须满足完全相同的政策。Checkpoint 磁盘恢复、所有 RNG、Scheduler、Scaler、Adam step、整数/结构数据仍要求 exact。

浮点上界包括：模型/EMA atol=1e-7, rtol=1e-6；裁剪后梯度 atol=5e-5, rtol=5e-4 且整体 relative L2<=5e-4；Adam 一阶/二阶矩分别使用独立更小绝对界；Prediction/Target/JVP 也单独受界约束。任何 NaN/Inf、状态缺失或超界均失败。这些是局部一次更新的误差界，不代表长训练轨迹或跨训练 Seed 等价。

本轮原始 FM 和 MeanFlow 均通过，梯度整体 relative L2 分别为 **3.88258e-8 / 2.39652e-8**，没有超界项；两次 MeanFlow loss 均为1.856037855，gradient norm均为17.0692234。仍存在非逐位差异，因此分类明确为 `BOUNDED_NATIVE_NUMERICAL_REPLAY_WITH_OWN_FM_CONTROL`，不称 Exact Replay，也没有继承旧 EF 数值例外。

最终工程 `GPU_PASS`、算子日志、误差界细项、源/config/manifest SHA、Frozen 状态、初始化身份和峰值显存见 `meanflow/native_jvp_v2/`。源 SHA：`9a7b41b05892dbb1df1067af99c08e4fc9422f60225fbe39e6f3d31df7812a00`。

## 接续安排

Stage2 工程门槛已通过，接下来在空闲 GPU3 执行单独、丢弃的32 successful updates Smoke。正式20K仍使用新鲜源初始化/优化器与完整10K日程。Stage1 保持独立工程验收，不能借用 Stage2 gate。

v2 控制器保留 EF0/EF1/EF2 和 Flow-only 完整评价依赖，并等待旧 v1 控制器完成自己的已登记工程尝试、释放共享卡锁，再续上 v2 的20K/900帧/seq08-41帧 NFE1/2/4/8与配对测速。已通过且源码/config/物理GPU一致的工程/Smoke证据可严格复用；失败证据不覆盖、不静默跳过。

当前还没有 MeanFlow 正式20K训练质量或速度结果，不能从 JVP PASS 推断模型更好或少步生成已经成功。
