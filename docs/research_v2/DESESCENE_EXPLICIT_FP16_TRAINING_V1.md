# 显式FP16网络＋FP32主权重训练

用户明确选择“网络权重和激活也显式FP16，另保留FP32主权重副本”。独立分支`research/desescene-explicit-fp16-training-v1`。这是新的从零20K精度Cohort；旧FP32/AMP结果和Checkpoint均保留，不混入新组曝光或冒充全程FP16。

## 实际精度和更新

- 所有可训练网络参数，包括Norm affine，实际存储FP16；兼容的Conv3d/Conv2d/Linear/DeformConv2d/SpConv/QKV/FFN输入输出FP16。不是只将外层autocast开启或把参数压成Half后又全部转回FP32计算。
- 原始XYZ、ROI、体素索引保持原FP32/FP64规则；噪声、状态、Euler积分和Loss归约保持FP32。
- BN/GN/LN/SceneNorm统计FP32，输出回原激活dtype；BN running buffers、时间frequency保留精度。先保存原buffer再转换，避免先Half再Float已经丢失频率精度。没有全局默认dtype改动或Vendor源码修改。
- 每个网络参数有独立FP32主权重。scaled Half gradient复制到FP32主权重梯度，GradScaler对FP32梯度unscale，Clip1，然后AdamW在FP32主权重/矩上更新，最后同步Half网络。这样不丢弃小于Half ULP的累积更新，也避免Half Adam epsilon/矩下溢。
- EMA权重与浮点累积独立FP32；Shortcut教师的运行副本为FP16。Checkpoint保存`model`Half、`master_parameters`FP32、`ema`FP32、Adam、Scheduler、Scaler、Source/Interval/CUDA/Python/NumPy RNG及Successful计数。
- 模型加载/同步保持原架构、BN分组和child顺序，未降低分辨率、Token数或Batch。V6 inference-only转换器没有被直接当成训练器复用。

主实现：`explicit_fp16_training_policy.py`、`explicit_fp16_training.py`；独立入口：`scripts/research_v2/explicit_fp16/worker.py`。

## 旧运行如何处理

旧Stage1 Encoder FP32/Flow AMP在713成功更新/1426曝光保存STOPPED，旧无EncoderStage2在1904更新/3808曝光保存STOPPED，均0AMP skips。原Source、训练日志、权重和旧等待队列凭证保留。新FP16正式训练重新从Seed20261032随机初始化，20K总曝光；旧参数、EMA、Adam和曝光不继承。Smoke32丢弃。

旧控制器已停止且标为USER_SUPERSEDED_BY_EXPLICIT_FP16，防止旧精度排队继续启动。没有停止其他用户/项目的任务。

## 实际验收

两组最终源码均已运行最大真实Batch2、实际反向、FP32主权重更新、全部Checkpoint状态精确磁盘恢复和下一原生更新的数值重放。S1为完整384×384×32网格，三尺度特征维持原B,C,Z,Y,X；S2为399672＋399671=799343Parents，未裁剪或降Batch。

运行时dtype hooks只在工程检查启用，训练已移除。S1审计实际覆盖73个Conv3d、6个DeformConv2d、12个Conv2d、15个Linear，以及原BN/GN；参数/输入/输出均Half。S2含SparseConv、Attention投影和FFN的Half前向与Half网络梯度，主权重/优化器梯度FP32。记录的是实际dtype与执行，不据此假称已测Tensor Core利用率或完整系统推理加速。

GPU最大场景峰值Allocated：S1约8.41GiB，S2约9.20GiB，属于工程Gate，不替代正式训练峰值或推理Benchmark。

36项CPU回归通过，覆盖真实模型构造禁预训练、主权重小于Half ULP的累积更新、Adam/EMA dtype、精确frequency、Norm统计、Half梯度、严格RNG、队列依赖及已有训练回归。

## 数值恢复政策

原`meanflow_replay_bounds.py`的FP32模型误差政策未修改。显式Half存储采用另行、在GPU验收前写定的`explicit_fp16_replay.py`：网络Half参数8 storage ULP＋1e-6 near-zero绝对容差，FP32主权重/EMA1e-6＋1e-5|x|；其他预测/梯度/矩阈值、全局梯度relative L2≤0.005和全部整数/RNG/计数的exact要求由该版本化文件完整定义。它不是沿用旧FP32容差后偷偷标PASS。

两组均通过该独立政策，梯度重放relative L2约2.96e-5（S1）、8.81e-5（S2），不是Bitwise Replay。这只验收同状态的一步数值重放，不能当成长期轨迹或几何质量保证。若超过新政策仍停止，不自动扩大界限或关闭梯度检查。

## 训练与后续队列

现有两组已通过自己的32成功更新/64曝光Smoke，0AMP skips，正式重新随机初始化后训练中。每组Batch2、Seed20261032、10K成功更新/20K曝光，原AdamW/LR/Clip/EMA，FP32积分。Stage2仍是普通FM，GT粗支持＋原25%邻近空Parent＋原Partial Observed Mask，无384D特征，不调用Encoder/Stage1 Flow/JVP。

后续三组全部使用同一显式FP16/FP32主权重策略：

| GPU | 依赖 | 接续 |
|---|---|---|
| 2 | 新FP16 S1 MeanFlow20K＋41评价 | 新FP16 S1 Shortcut20K＋41评价 |
| 3 | 新FP16无EncoderStage2 20K＋41评价 | 新FP16 S1普通FM匹配对照20K＋41评价 → 新FP16有Encoder Stage2 20K＋41评价 |

有Encoder Stage2冻结的Encoder只来自本队列普通FM从零20K EMA，并转为显式FP16运行；其Stage2参数随机初始化，与无Encoder组共享随机参数和fuse公共列。记录Encoder已有20K与Stage2另20K，不能称整个系统只见过20K。其他成熟Checkpoint不用于初始化。

自动队列入口`explicit_fp16/run_followup_queue.py`及`explicit_fp16_followup_queue.json`。三个后续GPU验收/Smoke尚待资源和依赖，CPU通过不等于已经GPU通过。每个任务仍须自己的最大Batch2、冻结状态、主权重和恢复检查；失败保留证据并停止，不加第二Seed/100K/新架构，不抢占任务。

评价唯一集合仍为seq08/skip100的41帧、NFE1/2/4/8，模型及精确CD使用GPU；物理IoU/JSD保留原实现。Stage1独立Parent中心指标、Stage2 Oracle-Support Native Fine中心指标分别报告，完整部署系统质量与正式推理时延均NOT TESTED。

## 实现失败记录

首次GPU2准入时旧任务还在优雅保存，未运行计算即被拒绝；随后确认旧进程退出再重试。首次GPU3的Half前后向已完成，最后可选冻结Encoder审计错误访问了无Encoder组属性，Gate仍拒绝；已修复并保留原结果。首次Gate晋升流水线的pin局部变量未初始化，Smoke/正式未启动；已修复并新增CPU回归，最终固定源码两卡重新GPU验收通过后启动。上述失败不作为性能/质量数据，日志与凭证均保留。
