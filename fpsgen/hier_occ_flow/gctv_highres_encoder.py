"""H1-H6 high-resolution GCTV variants, independently registered."""
import torch
from torch import nn
from torch.nn import functional as F
from .gctv_encoder import Compact3DEncoder,Coupler,lift,project


class AdaptiveProjection(nn.Module):
    """Softmax-weighted observed features; empty rays remain exactly zero."""
    def __init__(self,width):
        super().__init__();self.logits=nn.Conv3d(width,1,1);nn.init.zeros_(self.logits.weight);nn.init.zeros_(self.logits.bias)
    def forward(self,x,mask,axis):
        # Compute normalized weights in FP32, avoiding NaN on all-empty rays.
        scores=self.logits(x).float();mask=mask.bool();scores=scores.masked_fill(~mask,-1e4)
        weights=torch.softmax(scores,dim=axis)*mask.float();weights=weights/weights.sum(dim=axis,keepdim=True).clamp_min(1e-12)
        observed=(x.float()*weights).sum(dim=axis).to(x.dtype);context=x.mean(dim=axis);coverage=mask.float().mean(dim=axis).to(x.dtype)
        return observed,context,coverage


class DeformDepthwise2D(nn.Module):
    """CUDA torchvision deformable 3x3 depthwise convolution, learnable offsets."""
    def __init__(self,width):
        super().__init__();self.offset=nn.Conv2d(width,18,3,padding=1);nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        self.weight=nn.Parameter(torch.empty(width,1,3,3));nn.init.kaiming_uniform_(self.weight,a=5**.5)
    def forward(self,x):
        from torchvision.ops import deform_conv2d
        offset=self.offset(x)
        return deform_conv2d(x,offset,self.weight.to(x.dtype),padding=(1,1))


class HighResCoupler(Coupler):
    def __init__(self,width,feedback=False,adaptive=False,dcn=False):
        super().__init__(width,feedback=False);self.adaptive=adaptive
        if adaptive:self.projectors=nn.ModuleList([AdaptiveProjection(width) for _ in range(3)])
        if dcn:
            for plane in self.planes:plane.depth=DeformDepthwise2D(width)
        if feedback:self.add_feedback(width)
    def forward(self,x,mask,*,enable_feedback=True):
        args=[(self.projectors[i](x,mask,axis) if self.adaptive else project(x,mask,axis)) for i,axis in enumerate((2,3,4))]
        planes=[block(*v) for block,v in zip(self.planes,args)]
        first=x+self.update(lift(planes)/3)
        if not self.feedback or not enable_feedback:return first
        delta=first-x;updated=[p+r(delta.mean(dim=axis)) for p,r,axis in zip(planes,self.refine,(2,3,4))]
        return first+self.update2(lift(updated)/3)


class HighResGCTVEncoder(Compact3DEncoder):
    MODES={'h1':(True,False,False,False),'h2':(True,True,False,False),
           'h3':(True,True,True,False),'h4':(True,False,False,True),
           'h5':(False,False,False,False),'h6':(True,True,True,True)}
    def __init__(self,spec,variant='h1',activation_checkpointing=False):
        if variant not in self.MODES:raise ValueError('H1-H6 only')
        super().__init__(spec,variant='g0',activation_checkpointing=activation_checkpointing)
        self.variant=variant;use_f2,adaptive,dcn,feedback=self.MODES[variant]
        # Construct both common F1/F2 blocks before feedback-only parameters.
        self.f1=HighResCoupler(32,adaptive=adaptive,dcn=dcn)
        if use_f2:self.f2=HighResCoupler(64,adaptive=adaptive,dcn=dcn)
        if feedback:
            self.f1.add_feedback(32)
            if use_f2:self.f2.add_feedback(64)
    def forward_statistics(self,x,mask,*,enable_feedback=True):
        x=x.to(self.stem[0].weight.dtype);a=self._run(self.stem,x)
        a=self.f1(a,mask,enable_feedback=enable_feedback)
        b=self._run(self.level1,a);m1=F.max_pool3d(mask.float(),2)>0
        if hasattr(self,'f2'):b=self.f2(b,m1,enable_feedback=enable_feedback)
        c=self._run(self.level2,b)
        a,b,c=[p(v) for p,v in zip(self.fpn,(a,b,c))]
        b=b+F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False)
        a=a+F.interpolate(b,size=a.shape[2:],mode='trilinear',align_corners=False)
        return a,b,c
