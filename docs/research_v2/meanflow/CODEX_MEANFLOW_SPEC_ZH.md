# DeSeScene MeanFlow 接入执行规格

## 0. 工作性质与边界
这是待实现、待真实 GPU 验证的方案，不是已完成的仓库补丁。
参考分支：research/desescene-shortcut-pilot-v1
本次审阅提交：b9ad22e7f77e0b1ca9d9bdf7db8479798f55ac70。
先检查实际 Git、进程、GPU、源文件及实验产物。保护 Encoder/EF/Shortcut 现有作业，不因本方案重启或覆盖它们。
独立工作树/私有分支建议 research/desescene-meanflow-v1。
只做原版 MeanFlow 的前向时间改写，不混用 Shortcut、Reflow 或 iMF 的训练目标。
各正式方案最多 20,000 successful frame exposures，真单卡 Batch2，10,000 successful updates。保持 Seed20261032，不启动100K，不自动增加训练Seed。
本提示词要求先实施工程门槛；未通过真实JVP、最大Batch2和恢复检查时，不启动正式训练。

## 1. 已审阅的基础
- Stage1: stage1_occupancy.py, stage1_network.py:OccupancyUNet。
  x1=2*occ-1, x0~N(0,I), xt=(1-t)x0+t*x1, v=x1-x0。
  t=0为噪声，t=1为数据，推理沿正时间更新。
- Stage2: stage2_batching.py 的 packed_training_pair / scene_mean_mse；
  stage2_model.py:Stage2OccupancyModel，真实训练器导入该类而非只使用legacy model.py。
  每Parent一个Token，状态[M,8]，每场景共享一个时间，场景间独立。
- Stage2包含SpConv、FlashAttention、Point对象和activation checkpointing；普通backward成功不代表JVP成功。
- 现有Shortcut队列是EMA两步自举，不是MeanFlow；不得仅改名字。
- 已登记900帧Development用于学习/主要质量；seq08 skip100的41帧用于冻结后NFE对比，不用于调参。
- S1-FM20K已经注册为Flow-only对照；EF1/EF2不能代替它。
- B0仅在完整初始化/数据/精度/时间分布/优化器/随机数协议匹配时复用。

## 2. 数学合同
定义 U(x,t,s,c)=(Phi_{t->s}(x)-x)/(s-t)，0<=t<=s<=1。
保持endpoint s不变，对当前时刻t沿流线求全导数：
    U = v + (s-t)*(J_x U v + partial_t U)
将样本条件速度v=x1-x0代入原版MeanFlow训练式：
    D = jvp(U, (xt,t,s), (v,1,0))
    target = stopgrad(v + (s-t)*D)
    loss = per_scene_masked_MSE(U(xt,t,s), target)
重要：
- 这里是加号；原论文采用数据0->噪声1的时间定义，因此其目标里是减号。
- 网络内部若输入(t,d)，其中d=s-t，则直接对(t,d)做JVP必须使用(v,1,-1)，不能(v,1,0)。
- 优先对endpoint形式的closure做JVP，由closure计算d=s-t，自动包含链式法则。
- 不是仅对time求导；必须包括state Jacobian作用在样本速度v上的项。
- target全部detach；不计算该target的参数反向图。
- JVP使用当前online模型，同一参数/精度/条件；不是EMA教师目标。
- 条件c、Parent IDs、valid mask、坐标及拓扑在JVP中固定。
- d=0时严格退化为原FM loss。
- 不把gt-current_state直接当作平均速度监督，否则退回端点回归。
- 原版MeanFlow与iMF分别登记，不将预测切向速度或其他目标静默混入。

## 3. 模型接入
新增独立模块，避免更改活动实验的源码：
- meanflow_math.py
- meanflow_stage1.py
- meanflow_stage2.py
- meanflow_jvp.py
- meanflow_training.py
- meanflow_sampling.py
名称是新文件建议，不代表当前仓库已有。

对Stage1/Stage2统一新增time_end: [B]。
内部d=time_end-time。
沿用原time embedding和主干，新增连续interval embedding：
    embedding_new = embedding_time(t) + d * interval_mlp(fourier(d))
interval_mlp隐藏层正常随机初始化，仅最后投影置零；不要再乘一个零初始化可学习gate导致双零。
Stage1输出embedding保持128D；Stage2保持32D，用于early fusion、六个原时间调制Block及原decoder。
不改变Stage2的240D融合、64/128D PTv3、8D输出、Child顺序x+2y+4z。
禁用M1、L0/L1、B1和静态PTv3新变量。
普通forward、cached inference、AMP/显式FP16路径均传递time_end，不能只改legacy model.py。
Step0新模型在所有时间间隔的输出与源FM数值一致；d=0时结构上保持旧模型边界。
严格加载所有共有旧tensor，仅允许显式列出的interval新参数缺失。
保持父网格384x384x32、子网格768x768x64、ROI和原始surface-hit GT。

## 4. 时间采样与损失
主实验使用仓库匹配采样：
1. 保留原occupancy_flow_pair / packed_training_pair的x0与t抽样顺序。
2. 用独立interval RNG为每个场景采样s=t+(1-t)*rand。
3. 每场景以0.75概率令s=t，0.25保留有限区间。
4. 使用Bernoulli per scene，不用int(B*0.75)：B=2时后者会变成50%。
5. 保存interval RNG state到checkpoint；记录实际成功曝光中的比例。
6. 这是为了匹配原FM current-time边缘分布的迁移配置，不宣称与官方sorted-pair时间采样完全相同。
7. 主实验先用原有场景等权masked MSE，adaptive_power=0；保留可选adaptive版本，但不自动启动。
8. 若采用官方式adaptive weighting，先按场景有效元素均值得到q_b，再使用stopgrad((q_b+eps)^-p)*q_b。
   这不同于图像固定尺寸的sum reduction，应明确登记；相同权重的FM对照才可归因。
9. 不额外添加BCE/Shortcut/Chamfer/EMA consistency损失。
10. invalid状态、速度、导数及loss都按同一个mask处理，不能将非法Child噪声泄漏到网络。

## 5. JVP工程门槛
第一层：纯MLP/时间嵌入的CPU双精度解析测试。
第二层：Stage1真实Conv3d/GroupNorm/三线性插值JVP，冻结Encoder在JVP外执行。
第三层：Stage2局部MLP -> 时间Block -> attention -> SpConv embedding/CPE -> pooling/unpooling -> 全模型，逐级测试。

推荐分离：
    with torch.no_grad():
        _, D = torch.func.jvp(u_fn, (xt,t,s), (v,ones,zeros))
        target = ...
    prediction = u_fn(xt,t,s)
    loss.backward()

- no_grad不等于inference_mode；不要在JVP外套inference_mode。
- JVP分支关闭activation checkpointing；普通梯度分支可保留，必须验证前向数值一致。
- 不通过切换train/eval改变dropout、normalization或其他数学行为。
- 将排序、整数坐标、场景长度、拓扑等预处理移出可微closure；不要对动态激活detach。
- 冻结Encoder不代表Stage2内的LiDAR投影MLP冻结。可缓存原始冻结特征，不能跨optimizer steps缓存依赖训练参数的投影。
- torch.autograd.functional.jvp使用double-backward，不是SpConv/FlashAttention的保证可用替代品。
- 原环境与官方PyTorch实现环境不同，不在现有环境原地升级。
- Native JVP不支持时，先定位算子，必要时独立实现保持同样权重、padding、邻域、mask的参考数学路径，并证明前向及导数一致。
- 不能简单enable_flash=False后宣称无变化；vendor非Flash路径会改变小场景patch大小逻辑。
- SparseConv在固定索引下对输入features线性，可为输入切向实现验证过的规则；禁止绕过缺失导数返回0。
- 有限差分只能作为单独MF-FD近似研究方案或导数诊断。不得把它静默叫作exact MeanFlow；不得跳过任何验收。
- 若Stage2暂时不通过，保留BLOCKED_JVP，先推进Stage1；不强行开始20K。

## 6. 两个独立实验
### S1-MF20K
- 源Stage1 EMA SHA:
  b60ab632b26e79140ace4cdb94577db6aba43b02f44f3393da84a6a94622e3d2
- 冻结Stage1 Encoder及整个Stage2；训练Stage1 Flow和interval embedding。
- 对照S1-FM20K；以现场已注册配置为准。
- 当前已核验对照Flow LR=1e-5，seed20261032，warmup500，AdamW wd0.01，clip1，EMA0.999。
- Encoder只执行一次，放在no_grad外部条件准备；不要使用TensorRT可训练Flow。
- Stage1状态[B,1,32,384,384]，time/end[B]。
- 推理最终score保留clip((state+1)/2,0,1)，strict score>0.5选择Parent。

### S2-MF20K
- Stage1保持成熟冻结原始FM/NFE2/threshold0.5，无GT补候选。
- 源Stage2 online SHA:
  eaf70f8e2cbc068add3b744b1988299197c787b8f15567fd19fd06b0d5f3f74c
- 源训练曝光是99,520；本轮是额外20K微调，不写成scratch20K。
- 训练原Dynamic PTv3、原输入/输出模块和interval embedding；不改Encoder。
- Stage2 LR=2e-4，仅在完整对照合同匹配时沿用；新参数同LR。
- 真packed Batch2，state[M,8]，time/end[B]；row_to_scene=batch。
- JVP和正常forward都包括完整动态PTv3，不只对local MLP求导。
- 最终state>0解码Fine Occupancy，invalid Child保持屏蔽。

配对控制优先复用通过身份审计的已有源；若需重新训练控制，必须显式登记为独立20K，
不能以EF1/EF2或未追加预算的旧checkpoint替代。
已有GPU0控制作业保持原授权；新MF默认GPU2/GPU3空闲后独立运行，不自动扩大GPU授权。
MF不是取代当前Shortcut队列的授权；两者分别保留并按实测可用资源安排。

## 7. 采样
同一个模型评NFE1/2/4/8，不按NFE重复训练。
正向时间schedule: [0,...,1]。
每一步:
    u = model(x, time=t_i, time_end=t_{i+1}, condition=fixed)
    x = x + (t_{i+1}-t_i)*u
NFE1: x_final = epsilon + U(epsilon,0,1,c)。
NFE2: 0->0.5->1；第二步使用实际更新状态。
不在每个中间时间都time_end=1然后乘固定1/NFE。
无JVP参与推理，不提前阈值/二值化中间state；累计保持FP32。
Stage1 Parent集合在Stage1完整采样结束后选择，Stage2整个rollout固定Parent集合。

## 8. 验收
附带纯数学参考模块有CPU tests，但不是服务器GPU通过证据。
必须新增/执行：
1. 正负号解析测试、状态依赖速度场测试。
2. endpoint JVP(v,1,0)和duration JVP(v,1,-1)一致。
3. d=0 loss/gradient与旧FM等价。
4. interval零初始化输出等价；最后投影先学习、上游随后收到梯度。
5. target无参数反向图，prediction保留梯度。
6. 原顺序/打乱顺序、packed两场景与各自单场景对照。
7. Parent拓扑与条件在JVP内不变；无跨场景污染。
8. 与中心有限差分作数值诊断，合理区分max-pooling折点/浮点误差。
9. 真最大Batch2 forward/backward、AMP和VRAM。
10. 原始/参考JVP后端数值一致性；部署后端恢复后的质量验证。
11. checkpoint保存/恢复含所有RNG、optimizer、EMA、scaler、scheduler和计数。
12. 数值非确定性例外只能由同后端未修改对照支持，不自动继承旧例外。

工程smoke32独立权重丢弃。正式100->500->10000更新保持同一优化器、EMA和总日程。
1K exposures只做学习健康检查，不因少帧质量波动自动换方法。
出现NaN、断梯度、候选泄漏、JVP缺失或显存不够时停止该实验，不缩小场景冒充同协议。

## 9. 评价
- 900帧Development：主要质量与机制评价。
- 已登记seq08 skip100固定41帧(0,100,...,4000)：冻结checkpoint和协议后做NFE比较。
- seq08不进入训练、目标构建、阈值或sampler调参。
- S1扫1/2/4/8时Stage2固定原始NFE4；
  S2扫1/2/4/8时Stage1固定原始NFE2。
- 每个NFE同时比较原FM、已完成且身份匹配的Shortcut、MF。
- 固定输入/GT/ROI/noise；共享绝对Parent IDs使用相同Stage2 source noise。
- 报告同NFE效果，以及原部署预算FM2+4与更少步MF的质量/速度权衡。
- 单独成功后才测MF1+MF1组合，不预先宣称组合质量。

指标：
Parent precision/recall/IoU、fixedK ceiling、native Fine IoU/F1/P/R、
双向CD分量与总CD、physical IoU0.5/0.2/0.1、JSD、
Parent/Child数量、距离分组、最差帧/最差十分位、
Encoder/Stage1Flow/Stage2/完整generation wall和峰值VRAM。
输出均使用native Fine centers；jitter另报。
同GPU同后端交替多轮计时；不可将新MF裸PyTorch与旧TRT结果直接拼接。
记录JVP次数、额外forward、GPU训练时间，不将JVP重复计算记成额外样本曝光。

建议预注册筛选，不作为既成结论：
- Stage2 MF1相对匹配FM4：native Fine IoU非劣界-0.5pp，CD相对回归不超过2%，
  物理IoU/JSD无明显退化，Stage2实际延迟有清楚下降。
- Stage1 MF1相对匹配FM2：Parent IoU非劣界-0.3pp，Fine-GT ceiling非劣界-0.5pp，
  下游Fine质量不明显恶化。
- 单纯一NFE IoU增加却CD/覆盖显著变差不算成功。
- 不达1步但2步有效也如实记录。
- 多步的一致性残差只做诊断，不静默加Shortcut损失。
- 不直接用MF loss数值与FM loss数值排名；另报diagonal velocity MSE。

## 10. 提交与报告
报告准确区分REGISTERED/IMPLEMENTED/CPU_PASS/GPU_PASS/TRAINING/EVALUATED/BLOCKED。
保存源checkpoint、source/config/manifest SHA、运行环境、真实point/token数、
JVP后端、precision、normalization和checkpoint策略。
只提交私有仓库代码、配置、单测与紧凑JSON/CSV/MD；不上传数据、权重和大点云，不force-push。
已有Shortcut是独立强对照，不能将MeanFlow接入本身声称为原创方法。
最后返回真实完成清单、失败/阻塞证据、实验身份、私有分支和commit。
