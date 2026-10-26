# 2026-10-26：用户纠正初始化，改为从零20K对照

先前Stage1 FM/MeanFlow实际使用成熟Stage1权重、冻结Encoder，追加训练Flow20K；Stage2无Encoder普通FM同样从成熟Stage2权重初始化。它们均为适配实验，不能表述成总计20K的从零训练，也不能与Scratch Cohort混比。保留原始实验身份、结果、Checkpoint和负结果，不改写成Scratch。

此前采用微调的配置来源包括项目`docs/CODEX_MEANFLOW_SPEC_ZH.md`第6节（S1明确源EMA并冻结Encoder；S2明确额外20K）。用户本次明确纠正后，该初始化安排不再是后续正式对照的依据；最新用户要求优先。此前汇报应明确披露总曝光与训练模块，不能仅回答“都是20K”。

我方GPU2普通FM控制器1121977及子进程1320258已核验身份后优雅结束；训练在809次成功更新、1618曝光、0AMP跳步保存，summary为STOPPED。关闭自动适配续训，不删除产物。GPU3旧适配质量/测速在本次纠正前已结束，仍是适配结果。

后续必须独立注册Scratch20K：Stage1 Encoder+Flow从随机初始化共同训练；Stage2训练网络随机初始化，保留本组已指定GT粗支持与Observed Mask、不用384D Encoder特征。共同参数同初值、同Seed20261032、Batch2、10K成功更新/20K总曝光；评价仅seq08 skip10041。任何冻结的另一阶段历史模型仅作显式评价参照，不能宣称全系统均从零20K。不得通过给旧模型计数清零或重建优化器伪装Scratch。

新Scratch正式训练尚未启动，须有独立Runner/配置/初始化审计及最大真实Batch2、梯度、AMP和恢复验收。本记录不宣称已完成这些工程检查。
