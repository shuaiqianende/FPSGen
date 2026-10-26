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

2026-10-26 15:24 +0800：Stage2 已在 GPU3 完成独立、丢弃的32 successful updates Smoke，`SMOKE_PASS`，64 exposures，34 attempts，其中2次 AMP overflow 明确记录。45个 diagonal / 19个 finite-interval exposures；成功更新内16次实际完整JVP，115.09秒，峰值allocated 7,107,677,184 bytes。Frozen Stage1 exact。Smoke权重不进入正式初始化；正式20K仍使用新鲜源初始化/优化器与完整10K日程。

Stage1 也在 GPU3 完成了独立最大Batch2工程预检，`GPU_PASS`，full JVP/native reverse normalized error=5.76859e-6；自身原FM/MeanFlow replay的梯度整体relative L2为8.46982e-5/4.16412e-5，均在预注册界内。峰值allocated约7.35 GiB。它不能代替正式GPU2的同卡验收；队列将在GPU2释放后重跑其canonical gate，再进行独立Smoke和20K。

v2 控制器保留 EF0/EF1/EF2 和 Flow-only 完整评价依赖，并等待旧 v1 控制器完成自己的已登记工程尝试、释放共享卡锁，再续上 v2 的20K/900帧/seq08-41帧 NFE1/2/4/8与配对测速。已通过且源码/config/物理GPU一致的工程/Smoke证据可严格复用；失败证据不覆盖、不静默跳过。

持久v2控制器已启动，PID3979172，实际状态`WAITING_PREDECESSOR_EVAL`。EF0最新为439/900，GPU2原评价继续；v1控制器PID930632保持原计划。Stage2 Smoke启动PID3979171已正常结束。v2控制器在依赖满足后将严格复用已通过的Stage2工程和Smoke，而不重训Smoke或覆盖输出。Stage1/S2 formal均尚未启动；Stage2监督Encoder仍USER_HOLD。

当前还没有 MeanFlow 正式20K训练质量或速度结果，不能从 JVP PASS 推断模型更好或少步生成已经成功。

## 2026-10-26 15:39 更新：用户要求立即正式开训

用户明确要求空闲GPU3立即开始正式训练，替代上一快照中对EF0完成的全局等待。已核验旧两个CPU调度器均处于WAITING且没有子进程，仅替换这两个等待器；EF0的PID3230215与原评价完整保留。原任务状态和替换收据另存，没有将旧等待状态伪装成完成。

**Stage2 MeanFlow 正式训练已运行：GPU3，PID4037454，新自动控制器PID4037134。** 15:39快照为56 successful updates /112 exposures，58 attempts，2 AMP skips已记录，目标仍为10,000 successful updates /20,000 exposures，真实Batch2。GPU3实际占用约9GiB，采样时GPU利用率100%。正式目录从严格源权重新建，`resume=None, starting_updates=0`；没有使用Smoke的权重。

启动命令中的`--max-updates 100`是第一段注册健康检查，非最终训练预算。控制器按100→500→10,000自动恢复同一个formal目录，保持优化器、EMA、Scaler、Scheduler、Source/Interval RNG与完整10K日程，之后继续900帧Development、seq08/skip100的41帧NFE1/2/4/8评价及配对测速。

新计划为`configs/research_v2/hier_occ_flow/desescene_meanflow_start_now_20261026.yaml`，Python源码SHA和两份模型训练配置没有改变。完成的EF1/EF2和Flow-only控制仍要求真实完成证据；Stage2的工程/Smoke严格复用。Stage1仍在GPU2等待EF0自然释放资源，再进行该卡的canonical工程/Smoke/20K。

实际formal初始化、训练计数、源权重/config/source SHA、等待调度器替换收据分别见`meanflow/native_jvp_v2/formal_start_now_20261026.json`、`scheduling_handoff_20261026.json`。这只是训练启动进度，尚无正式20K最终质量结论。

## 2026-10-26 实际失败与独立接续（取代前面的启动快照）

Stage2 原 formal 在 **210 successful updates /420 exposures、213 attempts** 停止，第3次AMP梯度溢出触发原健康门槛（3/213≈1.41%）。三次失败的Loss有限，其中一次没有JVP调用，尚不能归因于JVP算子。未发生OOM，未放宽溢出门槛，未静默重试。最新完整Checkpoint实际停在100 updates；不能从210无损恢复。原目录、失败、日志和100步Checkpoint均保留。详见 `meanflow/native_jvp_v2/s2_amp_failure210.json`。

用户要求继续后，实际资源核验确认GPU3空闲。仅结束了无训练子进程、等待GPU2的CPU控制器4037134；GPU2的EF0评价PID3230215没有中断。Stage1自身已有同源码/配置、**同物理GPU3**的GPU_PASS，报告按字节原样复制到注册路径；这不是复用Stage2验收。源码SHA和两份原训练配置不变。

Stage1独立丢弃Smoke32通过：32 updates/64 exposures、0 AMP skips、43 diagonal/21 finite-interval exposures、17次真实JVP、29.99秒。随后严格从源模型重新初始化formal，没有使用Smoke参数。**100 updates/200 exposures健康检查PASS，0 skips**，调度器233813已自动恢复同一formal至500健康节点，之后10,000成功更新/20K曝光。训练子进程PID随分段变化，以`single_arm_supervisor.json`为准。实际快照见`continuation_actual_20261026.json`。

Stage1完成后只执行已登记的900帧Development、seq08/skip100固定41帧NFE1/2/4/8对照和无争用配对测速，不追加训练。GPU3每次只运行一个本项目任务。Stage2 AMP稳定性修复在独立分支`research/desescene-meanflow-amp-stability-v3`准备；其任何新源码/config需重新验收，不改本工作树的运行代码。尚无MeanFlow正式20K质量结论。
