"""384-only canonical token math and B1 instance reductions, unchanged capacity.

Legacy HierarchicalOccupancyModel and vendor PointFlow normalization stay intact.
Canonicalizing BEFORE the input MLPs also stabilizes mixed-precision GEMM row
packing. A scene mean/variance reduction replaces atomic index_add for the
single-scene-per-rank instance normalization used by this experiment.
"""
import torch
from .model import HierarchicalOccupancyModel


def single_scene_norm(base):
    class SceneNorm(base):
        def forward(self,point):
            features=point.feat
            if len(features)==0:return point
            if bool((point.batch!=0).any()):raise ValueError('Stage2 B1 normalization expects one full scene per rank')
            if len(features)==1:normalized=features
            else:
                centered=features-features.mean(0,keepdim=True)
                normalized=centered*torch.rsqrt(centered.square().mean(0,keepdim=True)+self.eps)
            if self.affine:normalized=normalized*self.weight+self.bias
            point.feat=normalized
            if 'sparse_conv_feat' in point:point.sparse_conv_feat=point.sparse_conv_feat.replace_feature(normalized)
            return point
    return SceneNorm


class Stage2OccupancyModel(HierarchicalOccupancyModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.backbone_kind=='ptv3':
            from fpsgen.voxel_point.vendor.pwm.ptv3_reference.model import PointInstanceNorm,SerializedPooling
            norm_type=single_scene_norm(PointInstanceNorm)
            for module in self.trunk.modules():
                if isinstance(module,PointInstanceNorm):module.__class__=norm_type
                # Official top-level shuffle_orders=False is NOT forwarded to
                # SerializedPooling, whose default stays True even in eval.
                if isinstance(module,SerializedPooling):module.shuffle_orders=False

    def forward(self,parent_ids,**kwargs):
        order=torch.argsort(parent_ids);inverse=torch.argsort(order)
        if len(parent_ids)>1 and bool((parent_ids[order][1:]==parent_ids[order][:-1]).any()):raise ValueError('Duplicate parent tokens')
        conditions={key:value[order] if value is not None and key!='time' else value for key,value in kwargs.items()}
        return super().forward(parent_ids[order],**conditions)[inverse]
