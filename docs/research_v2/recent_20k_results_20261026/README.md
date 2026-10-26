# 最近20K实验结果：2026-10-26

来自实际完成的900帧Development输出；Stage1 NFE2、Stage2 NFE4、Native Fine Voxel Centers。一个训练Seed20261032，真实Batch2。不同Cohort分别比较。完整源文件路径/SHA、7组逐帧指标、时间块Bootstrap见summary.json和per_frame_quality.csv。

## Stage1成熟权重继续微调

| 方案 | Parent IoU % | Fine IoU % | CD m | IoU0.1 % | JSD3D |
|---|---:|---:|---:|---:|---:|
| EF0 | 46.25250 | 31.04702 | 0.207024 | 14.48553 | 0.400118 |
| S1-FM20K | 46.23987 | 31.03243 | 0.209893 | 14.47827 | 0.400488 |
| EF1 | 46.27681 | 31.03794 | 0.208172 | 14.47532 | 0.399729 |
| EF2 | 46.24059 | 31.03469 | 0.206825 | 14.47892 | 0.399781 |

EF0是成熟100K原始EMA、不追加训练；S1-FM20K只继续训练原Flow；EF1只更新Encoder，EF2同时更新Encoder与Flow，均额外20K曝光。Frozen Stage2相同。

EF1的Parent IoU相对EF0仅+0.0243个百分点，分组95%CI跨0；Fine IoU略降，CD增加1.147mm（当前逐帧分组CI为+0.395至+2.128mm）。EF2 Parent IoU -0.0119个百分点、Fine IoU略降，CD减少0.199mm且CI跨0。两组没有达到预注册质量GO门槛，不推荐替换成熟原Encoder。

固定Teacher Support下，EF1/EF2的Fine IoU分别31.03901%/31.04677%，EF0为31.04702%；特征本身未显示明显Fine收益。新Encoder预测Support改变时，当前旧评价采用按行随机噪声，公共Parent不保证绝对ID对齐；这些部署结果应解释为当前协议的整体输出差异。

**重要身份限制**：EF1/EF2是直接Numerical Pilot，Resume记录为NOT_CERTIFIED_NUMERICAL_REPLAY，不能称逐位重放认证；EF0报告中的arm=EF1是复用reference配置标签，其reference_only=true且实际使用原源EMA。Source与状态审计另见各protocol文件。

EF0完成前，EF1/EF2相对Flow-only对照的CD分别好约1.72/3.07mm；这不能替代相对成熟EF0的结论。现已补齐原始参照。

## Stage2位置编码和局部Attention

| 方案 | Fine IoU % | 相对B0 pp | CD m | IoU0.1 % |
|---|---:|---:|---:|---:|
| B0 | 31.74501 | +0.00000 | 0.217099 | 14.84159 |
| L0 | 31.75935 | +0.01433 | 0.217203 | 14.84813 |
| L1 | 31.70565 | -0.03936 | 0.217775 | 14.82418 |

L0是8 Child局部XYZ位置编码，L1在此基础上加入2-head局部Child Attention，B0为同协议原始Dynamic FM。三组均完整10K成功更新/20K曝光/900帧。全部900帧输入SHA与Parent支持SHA严格一致。

L0 Fine IoU仅+0.0143个百分点；当前单Seed分组CI为+0.00166至+0.02417个百分点，幅度远低于1个百分点目标，CD反而增加0.104mm。L1 Fine IoU -0.0394个百分点且CD增加0.676mm。均未证明值得增加结构复杂度的几何收益；保留负结果，不继续扩展该Attention分支。

配对Bootstrap按sequence和floor(frame/100)分组，共9个时间块、4000次重采样，只有一个训练Seed。CI不代表跨训练Seed稳定性。

## 尚未获得正式结论的任务

- Stage1 MeanFlow：已完成10K成功更新/20K曝光，正在900帧评价；最终900帧及seq08/skip100 41帧NFE1/2/4/8质量尚未完成，不根据Loss宣称少步加速成功。
- Stage2 MeanFlow修复试验：最大Batch2/native JVP/恢复与Smoke通过，正式359步/718曝光、0 AMP skips后用户要求暂停，完整Checkpoint保留；尚无正式质量结果。此前16384初始scale运行210步触发AMP门槛的失败记录另存。
- Shortcut Models：本体未启动，完成的是S1-FM20K普通FM对照。
- 用户已确认Stage2用GT粗占用、保留原始Observed Mask、去掉384D特征，并进一步明确使用普通FM。旧MeanFlow原型等待器已停止、未开formal；后继分支research/desescene-stage2-no-encoder-fm-v1已重新通过最大799343 Parent Batch2普通FM/梯度/原生更新恢复，目标仅y-x0、单次网络前向、无interval/JVP/预热目标。新的FM Smoke及正式队列单独登记，正式仍等待Stage1训练评价完成；尚无该组20K质量结论。

这些训练消融尚无已完成的独占、同后端三轮端到端性能结论，不能使用共享评测内的计时宣称推理加速。
