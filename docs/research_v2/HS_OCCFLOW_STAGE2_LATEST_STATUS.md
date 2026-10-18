# HS-OccFlow：最新状态与 GPT 讨论材料

> 快照：2026-10-18T16:58:41（北京时间）。本文件是当前阅读入口；其他报告中的 RUNNING、PENDING 和 ETA 是各自记录时的历史状态。
> 私有分支：`research/hs-occflow-384-stage2`。本次只整理 HS-OccFlow；主工作区已有的其他修改保持原样。

## 1. 当前结论

- Stage1 已完成自己的 **100,000 次成功 training-frame exposures**，最终为 34,275 次 optimizer updates。
- Stage2 仍在训练：本快照 **21,905 updates / 87,140 个成功训练样本曝光**，目标为 100,000 个样本曝光，预计结束于 update 25,120。
- 当前 Stage2 为纯 FM；Stage1 冻结，两张卡共同训练同一 Stage2。没有追加 BCE、Shortcut、Chamfer、Teacher、Sinkhorn 或 PointFlow 训练。
- 已完成的 seq08/skip100 测试覆盖 41 个固定帧。13K 相比 10K 的部分 CD 改善，但高 NFE 的 0.5m IoU 略降。**尚不能宣称 Stage2 提高整体系统质量。**
- 16K 的 4+4 步复测目前只完成 37/41 帧，GPU0 后出现其他计算任务，保护逻辑退出；已配置空闲后自动断点续测。完整指标尚未生成，不将部分平均值当作 41 帧结果。

## 2. 实际模型和监督

| 项目 | 实际配置 |
|---|---|
| 父网格 / 子网格 | 384×384×32 / 768×768×64 |
| 子体素细分 | 2×2×2，一个 Parent 对应 8 维状态 |
| XYZ 范围 | [-50,50]×[-50,50]×[-4,4.4] m |
| 有效空间 | 半径50m球形 ROI，严格高度/点边界；体素有效性采用几何交集 |
| Parent 尺寸 | 0.2604167×0.2604167×0.2625 m |
| Child 尺寸 | 0.1302083×0.1302083×0.13125 m |
| 标签定义 | 原始局部地图、Poisson之前的 **surface-hit occupancy**；不是完整实体内部占用 |
| 输入 | Partial LiDAR XYZ；不使用第4列语义 |
| Stage1 | VoxDet Encoder + 3D Occupancy FM U-Net；实际 widths=[8,16,32,64] |
| Stage1 条件 | 三层128维金字塔，在父体素世界坐标采样后合为384维 |
| Stage2 | Local State MLP + 条件融合 + Sparse PTv3 + Local Velocity Decoder |
| Stage2 参数量 | 2,677,128 |
| PTv3 | width64；enc channels64/128、depth2/2；decoder64、depth2；patch64/128 |
| Token | 每个 Parent 一个 Token，Child 不展开成独立 Token |

Stage2 训练源为 `x0 ~ N(0,I_8)`；二值目标映射 `x1=2*O_GT-1`。
每场景采样一个 t，`xt=(1-t)*x0+t*x1`，目标速度 `x1-x0`。
唯一损失是有效 Child 上的 velocity MSE：先按每个场景平均，再平均场景和 DDP ranks。
没有额外正负体素重加权、occupancy/BCE loss、mass loss、endpoint loss 或 shortcut loss。

条件包括当前8维状态、实际Stage1金字塔特征384维、归一化Parent XYZ、Parent Score、输入点生成的8维Observed Child Mask、32维时间嵌入。
时间进入 PTv3 内部 block。Parent绝对网格、序列化/逆排序、pooling/unpooling保留对应关系；不同场景的batch/归一化/时间彼此隔离。
模型生成的是稀疏细占用，不是最终固定180K点的 PointFlow 输出。

## 3. 训练协议与真实部署之间的差异

**当前训练是 Oracle Parent：** 全部GT非空Parent + 数量约为正Parent25%的邻近空Parent。正负候选的Score都为常数1。
GT membership仅决定Oracle候选，未作为特征输入。条件仍来自真实Partial LiDAR的冻结Stage1 Encoder。
这个训练模式不需要每个训练样本都执行Stage1 Occupancy Flow采样。

**联合推理是 Predicted Parent：** Partial LiDAR先经过冻结Stage1 Encoder和Occupancy Flow，阈值0.5选择Parent，Score来自真实生成结果；随后Stage2预测8维细占用。
没有GT补候选、候选膨胀、Top-K截断或测试时阈值调优。GT只进入评价计算。
Stage1 Score是截断线性终点代理，尚未校准为概率。

因此目前明确存在 **Oracle/Predicted候选分布差异** 和 **常数训练Score/实际推理Score差异**。
这两项是已确认的协议风险；没有实验能证明它们就是IoU下降的唯一或主要原因。尚未完成Predicted Parent fine-tuning和Development候选策略校准。
Stage1漏检的Parent区域不能由只在候选内运行的Stage2恢复。

Stage1自身从scratch训练完成后冻结。Stage2从本轮自己的随机参数起步，后续仅恢复本实验的模型/EMA/优化器；这两阶段是分别训练的。

## 4. 数据、预算与恢复

- Train：17,870帧；Development：900帧；Guard：360帧。seq08不进入训练。
- 当前周期监控只用8个固定Development帧×2个随机seed；并非每次测900帧。
- 用户要求的seq08测试为 `000000,000100,...,004000`，共41帧，不据此修改超参数或选择方法。
- GPU2/3训练，每卡2个完整变长场景，真实global batch=4；无Padding、随机删Parent或降分辨率。
- FP16外层AMP，SpConv内部FP32，FlashAttention FP16，activation checkpointing；未使用Inductor compile。
- AdamW，base LR=2e-4，weight decay=0.01，grad clip=1，cosine，EMA decay=0.999。
- 每500次成功更新保存latest，每2500次保存milestone；每1000次更新做固定监控。
- 当前累计AMP跳步 **11**；这些尝试不计入成功更新或成功样本曝光。
- 预算换算：早期160个B1样本，其后GB4，`successful_frames=4*successful_updates-480`。100K个样本曝光对应25,120次更新，不能写成100K次optimizer updates。
- 前572次更新来自早期小样本诊断，之后完整训练集续训。400步Oracle overfit曾未达到预设质量门槛，该失败保留；用户随后明确要求直接进行100K样本训练，质量门槛转为监控，工程/数值/恢复检查没有豁免。

最近1000步训练计算均值约 **2.692s/update**，不含周期验证开销。训练仍在进行，本报告不把100K预算写成已完成。
最新已保存checkpoint：update **21,500** / **85,520** 个成功样本；报告读取时模型已继续到上述更高实时步数。

## 5. 每2000次成功更新的平均FM Loss

统计使用当前模型真实连续日志链，排除失败尝试、校验成功step无缺失/重复。第一窗口包含早期小样本和batch变更。

| Step区间 | 成功更新数 | 平均FM Loss |
|---|---:|---:|
|1–2,000|2000|1.098285|
|2,001–4,000|2000|0.612026|
|4,001–6,000|2000|0.574730|
|6,001–8,000|2000|0.559054|
|8,001–10,000|2000|0.557709|
|10,001–12,000|2000|0.537543|
|12,001–14,000|2000|0.541338|
|14,001–16,000|2000|0.530731|
|16,001–18,000|2000|0.523513|
|18,001–20,000|2000|0.518397|
|20,001–21,905（不足2000步）|1905|0.518753|

Loss整体下降；最近下降较慢。训练loss变小不能代替真实Predicted Parent推理的CD/IoU验证。
CSV：[每2000步统计](stage2_384_audit/stage2_loss_per2000_20261018.csv)。

## 6. 完整seq08/41帧联合推理结果

Stage1固定最终EMA；Stage2使用所列checkpoint的online权重。每行是41帧等权macro平均。
Stage1和Stage2分别执行表中的Euler NFE；每帧LiDAR Encoder只执行一次，实际特征被复用。
Stage1各NFE使用相同source seed；Stage2用按Parent ID索引的高斯表，确保共享Parent/不同NFE的初始噪声一致。

**CD目标是每帧完整的原始地图点云，通常百万级，不是细占用GT体素中心，也未抽样到180K点。**
CD = `0.5*(mean NN distance P→G + mean NN distance G→P)`，欧氏L2距离，单位m，未平方；所有点参与exact cKDTree查询。
预测P是Stage2生成的细占用体素中心。
**IoU₀.₅ₘ是0.5米物理网格分辨率，不是IoU@0.5判定或占用概率阈值。**
GT由原始地图点直接体素化；预测由细占用中心体素化。原点[-50,-50,-4]，网格200×200×17，最高Z格截到4.4。
Stage1漏掉区域的GT完整计入FN及G→P距离。没有附加噪点过滤。

| S1 NFE | S2 NFE | 10K CD(m) | 10K IoU₀.₅ₘ | 13K CD(m) | 13K IoU₀.₅ₘ |
|---:|---:|---:|---:|---:|---:|
|1|1|0.3366|40.69%|0.3344|40.95%|
|1|2|0.3187|42.96%|0.3012|43.28%|
|1|4|0.3136|43.87%|0.3010|43.90%|
|2|1|0.3242|39.29%|0.3315|39.43%|
|2|2|0.2540|45.68%|0.2427|45.44%|
|2|4|0.2426|47.16%|0.2321|46.93%|
|4|1|0.2793|40.28%|0.2776|40.48%|
|4|2|0.2382|45.94%|0.2333|45.66%|
|4|4|0.2325|47.09%|0.2277|46.82%|

4+4：CD从0.2325m改善到0.2277m（约2.04%），IoU从47.09%下降到46.82%（约0.273个百分点）。
2+4：CD从0.2426m改善到0.2321m，IoU从47.16%下降到46.93%。
这些是观测结果，不是统计显著性结论，也不足以宣称Stage2已有整体收益。

### 同条件Stage1-only对照

另行实际测了相同最终Stage1权重/噪声/41帧/0.5m定义，移除Stage2，直接使用Parent中心。

| Stage1 NFE | Stage1-only IoU₀.₅ₘ | Stage2@10K，NFE4 | Stage2@13K，NFE4 |
|---:|---:|---:|---:|
|1|46.28%|43.87%|43.90%|
|2|48.81%|47.16%|46.93%|
|4|47.55%|47.09%|46.82%|

10K的2+4对照：Precision由75.33%升到86.27%，Recall由58.57%降到51.22%。减少FP的同时损失了更多真实覆盖。
简单将Stage1候选内8个有效Child全部填满、再计算0.5m IoU的无网络对照分别为50.78%/51.40%/49.76%。
这个对照提示粗物理网格覆盖及中心栅格化会影响指标；不能据此推断细网格几何正确，应另测原生Fine Occupancy。

16K/4+4：**仍是37/41帧未完成结果**，GPU0新任务触发保护退出，等待空闲续测最后003700/003800/003900/004000帧。
自动续测tmux：`fpsgen_hs_stage2_eval16000_4x4_gpu0`。不将其37帧均值与前面41帧正式表混比。

## 7. 最近的Oracle Development监控（单独口径）

checkpoint update **21,000**，Stage2 EMA，8个固定Development帧×2seed=16条scene-seed记录。
候选来自GT Parent+附近空Parent，Fine Recall Ceiling=1；这里的IoU是原生768×768×64细网格，不能和上面Predicted seq08的0.5m IoU直接比较。

| Stage2 NFE | Native Fine IoU | Native Fine F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
|1|61.00%|75.75%|76.21%|75.33%|
|2|55.79%|71.60%|71.29%|71.92%|
|4|53.15%|69.38%|69.05%|69.73%|
|8|51.83%|68.24%|68.08%|68.42%|

本次Oracle监控中，增加NFE反而降低原生Fine IoU；这是需要分析的实际现象。它与Predicted seq08/0.5m指标处在不同候选和评价口径，尚不能仅凭二者趋势定位bug或证明FM优劣。

## 8. 工程验证与尚未完成的研究结论

已保存真实GPU2/GPU3全场景forward/backward、AMP、FlashAttention/SpConv、DDP恢复证据。
真实最大Oracle候选数399,672；单场景FM峰值allocated/reserved=7.619/8.975GiB；两个最大场景packed batch峰值=16.058/19.130GiB。
21步连续执行和20步保存+恢复1步的模型/EMA/AdamW/scaler/RNG/sampler一致性通过；这些是工程正确性，不能代替质量提升证据。
最近评测入口相关13项CPU测试通过。CPU测试不会被写成GPU测试通过。

尚未建立：公平匹配预算的BCE vs FM质量比较、正式Predicted Parent fine-tuning、Development候选校准、预测候选全训练集最大资源审计、100K完成后的最终结果、Stage2对Stage1的稳定增益。
本轮没有启动新的Teacher/Sinkhorn/PointFlow或额外方法长训。

## 9. 供GPT讨论的问题（假设与后续建议，未当作已验证结论）

1. 在Oracle support和真实Predicted support之间，如何用Development诊断分别量化候选漏检、Score分布差异和局部细分错误？
2. Oracle Fine IoU随NFE增加下降，应怎样检查速度场、时间覆盖、二值终点分布和少步/多步rollout，而不靠调整测试阈值掩盖问题？
3. 是否应先用训练/Development完成Predicted Parent适配与neutral-score一致性消融，再判断网络容量或增加其他损失？
4. 如何联合报告原生Fine IoU、0.5m物理覆盖和原始地图CD，避免将稀疏点密度、中心量化或覆盖扩张误认为细结构改善？
5. 哪些结果需要等预算BCE对照和多seed验证，才能支持FM本身的研究价值？

这些只用于研究讨论。本次整理没有修改训练loss、候选、阈值、空间定义或正在运行的源码。

## 10. 身份与阅读索引

- 本报告生成前Git HEAD：`67a93fa80c409b9804f4e142b35703797d0eb9bf`；提交本报告后的HEAD以GitHub分支为准。
- 当前训练源码SHA256：`01ef37195ef50b412016dd5bae481afb2421d4986b64618c153ed5cf5929a3f7`。
- 最新保存Stage2 checkpoint SHA256：`4c7842510f0415fcb0656146bce539fdc49123138e724541c8277c77380bfa79`。
- 配置 SHA256：`36f9659d7f3951b8ce49bb4869e9b9ac2e5291a9b6af473f2185a17a36037390`。
- Geometry SHA256：`c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b`。
- Stage1 checkpoint SHA256：`b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2`。
- Train manifest SHA256：`8dc632d341ffc2c6f0c139322cd24767616d570cf84c91fb4e516a26d7bab064`。
- Development manifest SHA256：`6447076563b6c0efffbd3a07c5f695ebedf8aef1fa270a5cda9d99df73b30d53`。

建议阅读顺序：

1. 本文件：当前状态、损失、不同评价口径、实际问题。
2. [训练配置](../../configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_100kframes.yaml)。
3. [Stage2模型](../../fpsgen/hier_occ_flow/stage2_model.py)、[条件与候选](../../fpsgen/hier_occ_flow/stage2_condition.py)、[冻结Stage1适配器](../../fpsgen/hier_occ_flow/stage1_occ_adapter.py)。
4. [10K完整41帧及Stage1-only对照](HS_OCCFLOW_STAGE2_SEQ08_41_GRID124.md)。
5. [13K完整41帧报告](HS_OCCFLOW_STAGE2_SEQ08_41_STEP013000.md)。
6. [真实GPU审计](HS_OCCFLOW_STAGE2_384_GPU_AUDIT.md)、[真实GB4与恢复](HS_OCCFLOW_STAGE2_384_GB4.md)。
7. [100K样本曝光预算与历史启动记录](HS_OCCFLOW_STAGE2_384_FM_100KFRAMES.md)。其中早期ETA是历史估计，以新快照及实测速度为准。
8. [16K未完成复测](HS_OCCFLOW_STAGE2_SEQ08_STEP016000_4PLUS4.md)。完整结果以服务器输出目录中results.json及summary.csv是否生成、是否41帧为准。
9. [机器可读状态](stage2_384_audit/latest_status_20261018.json)、[21K Oracle监控摘要](stage2_384_audit/oracle_monitor_latest_20261018.json)。

PLY示例已在服务器保存（不上传GitHub）：
`/data-12/M2024-HWZ/FPSGen/experiments/hs_stage2_ply/step013500_seq08_003923/`。
有Stage1粗占用、Stage2 NFE1/2/4/8、Partial，以及另行导出的GT raw map和GT fine centers。
该帧原始GT997,208点，细占用中心496,747点；GT在推理完成后导出，未参与生成。3923不在41个skip100帧中。

仓库只提交代码、配置、测试、Markdown和小型JSON/CSV。Checkpoint、PLY、原始地图、数据集、环境、编译产物保持本地。
