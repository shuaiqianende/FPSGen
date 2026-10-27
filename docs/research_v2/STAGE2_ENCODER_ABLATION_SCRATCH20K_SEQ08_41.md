# Stage2 Encoder注入消融：Scratch20K /seq08 skip10041帧

两组Stage2均从零训练20K成功曝光、Batch2、10000成功更新、Seed20261032、同Manifest/原Surface-hit GT/原Gaussian FM/原动态PTv3/原子体素顺序。共享初始张量及Fusion公共列有exact匹配审计；含Encoder组仅新增LiDAR分支/列随机初始化。

含Encoder组使用先前普通FM从零20K得到的冻结Stage1 Encoder；不是成熟100K初始化，也不是Encoder联合训练。无Encoder组去掉384D Pyramid输入和对应投影，保留GT Parent坐标、Observed Mask、状态、时间和动态PTv3。

## 适用范围

本次训练和评价均为 **GT粗占用+25%邻近空Parent**、Score常数1、原Partial LiDAR Observed。两组相同输入/候选算法/Seed；41帧Parent数逐帧一致，平均230459.0244。Oracle Support覆盖全部GT Parent，Ceiling=100%。不是Stage1实际预测Support，不代表部署质量；也不是完全不使用LiDAR。

|NFE|无Encoder Fine IoU %|带Encoder Fine IoU %|增加pp|无Encoder CD m|带Encoder CD m|
|---:|---:|---:|---:|---:|---:|
|1|57.7755|57.8016|+0.0261|0.077531|0.077919|
|2|52.0979|52.3671|+0.2692|0.075826|0.075623|
|4|49.6867|49.9463|+0.2596|0.077486|0.077238|
|8|48.4631|48.6924|+0.2294|0.078504|0.078257|

完整Native P/R/F1、CD、三档物理IoU、两种JSD、Observed/Not-directly-hit IoU和输出数在CSV；逐帧原记录及配置/模型身份全部保留。

NFE2/4的41帧Native Fine IoU全部略微受益于Encoder；NFE1仅21/41改善且宏平均仅+0.0261pp。NFE4 IoU增加约0.2596pp，CD下降约0.32%，Occupied Child仅增加约315/帧，输出数几乎不变。结论是小幅且部分NFE一致的收益，不能说完全无用，也不能说强依赖。单个训练Seed，不代表跨Seed稳定性。

## 部署含义与缺测

真实Predicted Support上的这对模型配对41帧评测：NOT TESTED。Oracle粗支撑本身提供强几何线索，可能降低对深层Encoder条件的依赖；这只是待验证解释。不能把本表57.8%与Predicted Support历史31%直接横比。

正常两阶段部署中，Stage1依然需要Encoder产生粗Support；Stage2通常复用其Pyramid。去掉Stage2注入不等于删除整个Stage1 Encoder，不能直接声称节省全部Encoder延迟。新两组真实端到端速度：NOT TESTED。训练实耗分别约5.75h与7.84h，负载/时段不同，非配对性能证据。

下一项有价值的验证是保持同一冻结Stage1、同一真实Predicted Support，在41帧上对比这两个已训练模型；不需要新训练。本报告未自动启动该评测或额外训练。
