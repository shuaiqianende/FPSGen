# 从零20K后续三组：实际自动队列

用户要求：三组均接入队列，前序训练和seq08/skip10041帧评价结束后续跑。独立分支`research/desescene-scratch20k-followups-v1`。不修改当前GPU2/GPU3的工作树、训练源码、权重或评价程序，不抢占其他进程。

## GPU依赖图

| 物理GPU | 现有任务（保持不动） | 后续任务 |
|---|---|---|
| 3 | S2无Encoder普通FM，从零20K＋41评价 | S1普通FM对照，从零20K＋41评价 → S2有384D特征普通FM，Stage2从零20K＋41评价 |
| 2 | S1 MeanFlow，Encoder＋Flow从零20K＋41评价 | S1 Shortcut，Encoder＋Flow从零20K＋41评价 |

优先使用先释放的GPU3补普通FM对照，不必等GPU2 MeanFlow结束。所有新任务Batch2、Seed20261032、10K successful updates/20K successful frame exposures。单卡同时仅一组本项目训练；额外EMA教师前向不计为新曝光。无第二Seed、无100K、无其他架构自动扩展。

## 1. S1普通FM匹配对照

原始Encoder＋Flow随机初始化，全部共同训练；Encoder FP32精度岛、Flow FP16 AMP、确定性cuDNN/关闭TF32与当前MeanFlow对齐。LR Encoder1e-4/Flow2e-4、AdamW wd0.01、500成功更新预热后Cosine、Clip1、EMA0.999、AMP scale2048，原Gaussian-to-binary FM。

实际CPU构造验证：普通FM、Shortcut和正在训练MeanFlow的全部共同初始Encoder/Flow参数逐位一致；新的步长嵌入是唯一额外分支。共享数据顺序与独立Source/Time RNG政策一致。先前成熟权重追加20K对照不能替代这组。

## 2. S1 Shortcut

输入`x_t,t,d`，d=0采用原FM监督；有限步长采用EMA的两次半步速度平均作为停止梯度目标：`v_target=(v_1+v_2)/2`，第二次查询状态为`x_t+d*v_1`、时间`t+d`，在线网络学习步长`2d`。75%场景抽样FM、25%抽样一致性，真Batch2采用逐场景Bernoulli，而不是假称每个Batch恰好25%。half-step范围1/2到1/128，满足`t+2d<=1`。FP32状态更新、Encoder FP32、Flow AMP。只使用FM与一致性监督。

步长是连续嵌入；初态d=0和有限d均与共同随机FM主干输出一致，新增最后投影先收到梯度。EMA教师Encoder与Flow均eval/no_grad，两个半步调用、教师Encoder调用及成本单独记录。在线Encoder仍有训练梯度，教师重复前向不增加曝光。

该实现遵循[Shortcut Models原论文的FM基础与两半步一致性](https://arxiv.org/html/2410.12557v2)，是在原Occupancy网络上的独立适配，需要本组GPU与学习验证；当前不宣称已取得质量或时延收益。

## 3. Stage2有/无Encoder特征

Stage2网络随机初始化，添加384D Parent-center特征；Encoder来自**本队列S1普通FM完整从零20K的EMA**，冻结、FP32 eval、不运行Stage1 Flow。源Checkpoint仅在正式生成后绑定SHA；校验其Scratch身份、代码/config/manifest/geometry/seed和20K计数，拒绝成熟或未完成权重。

Stage2共享随机参数及176个原fuse输入列，逐位复制自同Seed下新构造的无Encoder随机模型；新LiDAR分支及新列随机初始化。没有从训练过的Stage2复制参数。真实CPU检查其参考初始化SHA与当前无Encoder组一致。

支持仍为相同GT coarse＋25%邻近空Parent、统一Score1、原Partial LiDAR Observed Mask，同源Noise/Time和数据顺序。所有Fine GT只用于监督。冻结Encoder不进入优化器或EMA更新，GPU验收检查其参数与Buffer不变。每步重新提取真实输入特征，保存Source SHA，不跨优化步骤缓存可训练投影。

这是“训练了20K的冻结Encoder条件＋随机Stage2训练20K”的两阶段实验，不能称整个系统只见过20K，也不能把Oracle支持质量当成部署预测支持质量。Stage2 Loss更新Encoder仍未启动。

## 工程与接续

`configs/research_v2/hier_occ_flow/scratch20k_followup_queue.json`包含仅三项实验、固定源码/config Hash、前序身份与依赖。控制器`queued20k/run_followup_queue.py`在后台等待：

1. 前序formal summary严格10K成功更新/20K曝光，来源Scratch且代码/config/arm匹配。
2. 原始41帧评价完整EVALUATED，NFE1/2/4/8；900或部分帧不能满足依赖。
3. 对应GPU无其他计算任务、可用显存至少22GiB。
4. 新任务CPU检查后，实际执行本组最大真实Batch2前向/反向、梯度、参数改变、完整Checkpoint/EMA/Adam/Scaler/Scheduler/RNG恢复和固定界限下的原生更新重放。
5. 独立32-update Smoke，丢弃其训练权重；正式重新随机初始化为同一初值与计数0，最多20K曝光。
6. 结束后自动41帧质量评价。当前没有已完成的后续GPU/Smoke/质量成绩；它们的状态必须为PENDING，而不是根据CPU测试宣称通过。

失败即BLOCKED_REVIEW_REQUIRED，不关闭梯度/数值审计、不截断Parent或场景、不自动换Seed或沿用旧数值例外。队列与任务日志均保留，重复启动被锁与已有记录拒绝。

## 评价与时长

仅seq08 skip100固定41帧，NFE1/2/4/8，同输入、原始Surface-hit GT、ROI、Noise政策。模型/精确CUDA KD-tree CD使用GPU，物理IoU和稀疏JSD保持原实现。Stage1为粗Parent中心指标，Stage2为Oracle支持的Native Fine中心指标；全系统部署收益和正式同卡配对时延尚待独立测量。

当前两组剩余训练时间取最近200步的实测均值，不包含质量与Checkpoint开销。三组后续尚无真实步速，因此其时间是计划估算，启动Smoke后必须更新；不得称其为实测性能。最新时间快照与实际队列PID见本目录JSON。

## 已运行的后台等待控制器

GPU2控制器PID1754816，等待当前S1 MeanFlow正式20K和41帧评价；随后启动Shortcut流水线。GPU3控制器PID1754818，等待当前无Encoder Stage2正式20K和41帧评价；随后普通FM对照→其评价→有Encoder Stage2对照→其评价。这是实际进程与可执行依赖队列，不只是YAML计划；当前没有提前开始三个后续正式训练。

CPU28项总回归＋新增1项CLI回归全部通过（共29项）。独立直接CLI check-only对GPU2/GPU3均返回QUEUE_READY_WAITING_DEPENDENCIES。后续三组最大场景GPU验收与Smoke尚未运行，不将其记为PASS。实现期间修复了queue.py遮蔽Python标准库queue的问题，文件已命名run_followup_queue.py，没有因此启动任何训练。

时点估算：当前S1剩余17.55小时，S2无Encoder剩余7.01小时，来自最近200步均值6.624/2.883秒；不包含结尾41帧评价。排队三组步速未测，按现有计算量与依赖关系粗估整个队列约36–48小时，启动后以真实Smoke步速更新。若数据/工程验收阻塞，不能据此承诺完成时刻。
