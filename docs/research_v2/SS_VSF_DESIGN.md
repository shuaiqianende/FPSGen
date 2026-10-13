# Structured-Source Voxel Shortcut Flow (SS-VSF)

独立的 Stage 1 研究实现。已有 BEVFlow、PointFlow、Teacher checkpoint 和评测核心均不修改；Stage 2 不训练，也不接收本路线的隐藏特征。代码、CPU 测试和短 GPU pilot 用于验证实现，不能替代正式实验。

## 几何与数据

表示范围是 `[-50,50]×[-50,50]×[-4,4.4]`，网格 `256×256×32`，共 **2,097,152** 格；尺度 `0.390625×0.390625×0.2625 m`。实际 GT 合同额外满足三维半径 `<50 m`。有效体素通过轴对齐盒与圆盘/球的交集判断，保留中心在 ROI 外的边界相交体素。

公共 count、source、velocity 和条件为 `[B,C,Z,Y,X]`。上游 VoxDet 保留其既有内部 XYZ tensor ABI，在唯一 geometry adapter 中转换；新 velocity U-Net 的 Conv3d 使用 ZYX。flatten ID 为 `(iz*256+iy)*256+ix`。

训练序列 `00–07,09,10`，时间块 development 和 guard 从训练集冻结划出。SS-VSF pilot 不读取 seq08；正式测试须冻结协议后独立执行。配对复用已有 numeric frame pairing，输入仅保留前三列 XYZ；第4列不参与网络。

GT 文件必须原始180,000点、全部有限。全17,870训练帧审计发现2,567帧合计4,226点在球边界外，最大超出0.585309 mm，占全部点1.31 ppm。依照此前允许过滤少量越界点的授权，正式配置显式采用 `filter_audited`，上限每帧32点且最大距离1 mm。每帧记录实际保留点数、过滤点数和距离；超限拒绝，非有限值拒绝。**GT不复制、不补点、不修复总质量**，实际 target mass 为179,987–180,000。推理生成预算始终180,000。可切换 `policy: strict` 复现严格审计。

## 冻结基础密度

固定官方 VoxDet LiDAR tokenizer、128宽3D backbone、GeneralizedLSSFPN、SpatiallyDecoupledFPN、六方向引导和四次 attention refinement。沿用先前已审计的 standalone compatibility adapter，不通过替代 U-Net 实现 VoxDet。

六维 LiDAR 输入是 relative XYZ、absolute XY、`norm(XYZ)/50`；最后一个特征是输入适配，不是语义标签或未经对齐的 intensity。六方向监督由 GT occupied run 派生，不是官方语义 VoxNT。上游训练期 `use_gt_refine` 相关语义过滤不进入新推理接口。

基础输出为独立 occupancy head 的概率乘 conditional positive count。新 occupancy classifier 接收官方 refined64通道特征，`Conv3d(64,1,1)`共65参数；count head保留此前1通道 density classifier。原density参数22,082,470，SS-VSF base为22,082,535。短 pilot 从旧25k density checkpoint warmstart，但新球形数据合同与 occupancy head 要重新验证。

Base loss 是此前7项 density loss 加平衡 occupied/background BCE 和 occupied conditional-log-count SmoothL1。全部权重在 YAML 中公开。source gate只依据development选择，门限冻结后才允许flow训练。门限后的非负密度归一化至180,000；总质量无效直接失败。

## Source 与 count mapping

`phi(n)=2*log1p(n)/log1p(n_ref)-1`，训练全帧审计得到 `n_ref=24`。大于24的目标仍编码至大于1，绝不饱和裁剪。FP32 inverse只在最终计数解码时限制指数至 `log1p(1e6)`，记录限制比例；无每步裁剪、无每步整数化。

S0独立Gaussian、S1确定性base、S2固定Gaussian扰动base、S3观测命中/未命中双档扰动base。S3 known mask只由输入XYZ rasterize产生，并不声称它能表示全部可见空间。source噪声默认observed0.01、unknown0.05、floor0.01；development诊断0.02/0.05/0.10。所有正式source来自同一冻结base，不使用GT occupancy或GT density。

## 原始 voxel-space velocity

pilot U-Net widths `[8,16,32,64]`，参数528,081，time/step Fourier+MLP和FiLM。保持 full-resolution state和output，不是latent FM。三尺度128通道条件只在coarse层投影融合，base每场景编码一次，多次velocity调用共享条件。

FM用配对直线插值和 `target-source` MSE，instantaneous分支 `h=0`。Shortcut使用同一EMA模型两次half-step，no-grad且eval，不重采样噪声，平均速度为bootstrap target。大步由student监督；endpoint样本一半显式包含 `(t=0,h=1)`。

实施适配：bootstrap半步扩展至 `{1/16,1/8,1/4,1/2}`，使正式8NFE对应步长也有监督。最小positive step `h=1/16`增加解析FM anchor（权重0.25），并在consistency warmup前就训练；仅h=0分支不足以约束positive-h EMA叶子。稳定后以小比例使用EMA实际rollout起点，记录 off-path 标记。

Surface在真实endpoint上加balanced BCE+Dice occupancy proxy以及4尺度normalized mass L1。下采样用sum pooling。独立比较occupancy-only、multiscale-only、both和相同额外更新数的无surface continuation。该设计不声称中间轨迹非负或质量守恒。

## 推理与接口

每帧 **1次 VoxDet base forward + N次velocity forward**，NFE实际计数。一次修正不等于整个Stage1只调用一次网络。标准CFM调用instantaneous分支；Shortcut调用对应正h。

最终FP32 decode、有效ROI mask、非负化、归一化，之后精确整数allocation。Gaussian anchors从体素与球/圆盘的交集截断采样；不坐标clamp，不复制补数。source-indexed顺序不变，导出 density、allocated_counts、source_anchors、voxel_ids、voxel_centers及manifest。不导出条件feature，不运行PointFlow/Teacher/Sinkhorn。

## 实验边界

Gaussian CFM、S2 CFM、S3 CFM、S1 residual和匹配source的Shortcut使用相同冻结base、mapping、frame cohort、网络容量和source seed。NFE曲线包含1/2/4/8/16；路径长度比从16步实际rollout计算，不能用单步R=1证明直线性。必须同时报告GT误差、raw mass calibration以及180k anchor几何诊断。

`--smoke`最多16成功更新，`--pilot`最多512。非pilot正式训练需要对应stage gate report `READY`。当前代码验证不自动授权多组长训，报告不将短pilot结果解释成研究结论。
