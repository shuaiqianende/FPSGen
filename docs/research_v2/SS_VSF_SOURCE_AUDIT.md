# SS-VSF source audit

固定来源：

| 来源 | Commit | 用途 |
|---|---|---|
| [VoxDet](https://github.com/vita-epfl/VoxDet/tree/e10d8280de96d4f03f54b616e57e7c6754044fa4) | `e10d8280de96d4f03f54b616e57e7c6754044fa4` | 官方LiDAR tokenizer、backbone、necks、head/refinement |
| [Shortcut Models](https://github.com/kvfrans/shortcut-models/tree/601004348667094e1b71f30942199759412d4432) | `601004348667094e1b71f30942199759412d4432` | step conditioning、two-half bootstrap、EMA/FM联合训练 |
| PWM（同所有者私有仓库） | `c6e809486db3d0b6f8503318dfea246c06c7a50e` | symmetric-log mapping、linear path和voxel mass工具参考 |

逐文件SHA和来源在 `third_party/voxel_shortcut/SOURCE_MANIFEST.json`。Shortcut Models保留MIT license。PWM快照未包含license，仅保留在所有者私有研究仓库，禁止向public origin发布。参考快照不作为直接运行的JAX/PWM训练依赖。

VoxDet当前代码复用 `fpsgen/voxel_point/vendor/voxdet` 的已固定官方模块与独立compatibility adapters。新路线只新增 `fpsgen/voxel_shortcut` 和脚本/config/test/document，不改BEVFlow或Stage2核心。正式架构审计JSON记录preserved modules、新65参数occupancy head和legacyXYZ→publicZYX转换。

相对PWM：去除encode的上界截断及inverse的[-1,1]裁剪，超出n_ref的真实GT计数不丢失。相对Shortcut JAX参考：新PyTorch raw3D网络、球ROI和LiDAR条件，按本研究合同不在中间state/velocity上采用[-4,4]裁剪。正步长leaf的解析FM anchor和8NFE bootstrap覆盖属于本实现适配，应在论文中单独记录。

上游 `use_gt_refine=True` 的训练目标过滤与语义run逻辑经过审计；新base/velocity forward没有GT参数。六方向loss改为density-derived几何run，不能宣称原论文语义VoxNT原封不动复现。新的occupancy×conditional-count head也属于明确的任务适配。

本研究不提前主张原创性或实验优势；来源审计用于明确代码来源、实现差异和待验证假设。

最终compatibility检查修复了继承checkpoint wrapper在重算时重复更新BatchNorm buffers的问题。新encoder仅在recompute时临时使用buffer clones，保留batch-stat前向/梯度，原VoxDet/voxel_point文件不变。CPU测试比较普通与checkpoint执行的输出、梯度和buffer逐项一致；GPU2+3 DDP4步验证每个BN buffer仅更新4次。现有冻结base pilot在修复前生成，保留原checkpoint供所有对照共享；不能把它标为修复后的训练结果。正式base训练须使用该修复版。
