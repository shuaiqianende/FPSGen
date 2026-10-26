# 更新入口

最新900帧结果及独立GPU预检见 [2026-10-26更新](DESESCENE_ENCODER_900_MEANFLOW_GPU_STATUS.md)。以下为首次排队时点快照，不能当作实时状态。

# MeanFlow 最新实施状态

只读快照：2026-10-26 01:42:58 +0800。该文档是时点记录，不代替服务器实时产物。

本轮实现提交：`5951a83`。独立分支：`research/desescene-meanflow-v1`。

| 项目 | 实际状态 |
|---|---|
| 原版 MeanFlow 数学、Stage1/Stage2适配、独立训练/采样/评价器 | IMPLEMENTED |
| CPU 测试 | CPU_PASS，22/22 |
| 实际 Stage1 EMA / Stage2 online 文件 SHA | 两份均匹配用户注册值 |
| 严格加载与新增参数 | S1共享96、S2共享169 Tensor保持，分别只新增4个interval参数 |
| Stage2源训练曝光 | 99,520；未来为额外20K适配 |
| B0对照 | 元数据/10,005次尝试数据顺序审计通过；仍须新适配器真实GPU前向门槛 |
| GPU JVP / 最大 Batch2 / 原生恢复 | NOT TESTED，待前序评价与GPU准入 |
| MeanFlow正式训练 | NOT STARTED |
| 新 MeanFlow 900 / seq08_41结果、性能收益 | NOT TESTED |
| Stage2 Loss更新Encoder | USER_HOLD |
| Shortcut | 原独立登记保留，尚无完成模型供三方对照 |

## 实际自动续训

控制器 **PID 930632** 已在服务器后台启动并核验存活，状态 **WAITING_PREDECESSOR_EVAL**。这不是只有YAML的待办登记。它目前只占CPU等待，不新增GPU计算进程。

等待 EF1/EF2 20K及900评价、EF0 900评价、GPU0 S1-FM20K 20K及900评价的实际产物。满足后，GPU2先验收S1-MF20K，GPU3先验收S2-MF20K；各自通过后自动32-update独立Smoke → 正式100/500/10K同日程精确恢复 → 900 Development → seq08/skip10041帧NFE1/2/4/8 → 独占时的三轮配对时间测试。

Stage2若缺原生JVP，留下算子堆栈并记BLOCKED_JVP；不回传假零、不用FD代替、不拖住通过的Stage1。原生恢复若不严格通过也不自动继承旧EF数值例外。任何未通过门槛的方案均不强行启动20K。

实时状态：`experiments/desescene_meanflow_v1/queue/supervisor.json`；各Arm启动后看对应的`queue_status.json`。源码SHA固定为 `1293e8bef20b2684859fac32db4b228eaed0b604a93b8143fc6274c1ab617391`。运行源文件未改动，历史结果未覆盖。

## 前序训练快照

| 任务 | Successful updates | Successful exposures | AMP skips | 状态 |
|---|---:|---:|---:|---|
| EF1 | 7311/10,000 | 14622/20,000 | 0 | 训练中，之后原队列900评价 |
| EF2 | 7305/10,000 | 14610/20,000 | 5 | 训练中，之后原队列900评价 |
| S1-FM20K | 7314/10,000 | 14628/20,000 | 5 | 训练中，之后原队列900评价 |

完整协议、可复现入口与停止条件见 [实施计划](DESESCENE_MEANFLOW_PLAN.md)。本次没有启动Stage2监督Encoder、100K或额外Seed。
