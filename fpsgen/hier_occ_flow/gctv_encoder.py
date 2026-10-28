"""Compact 3D and voxel/plane coupling controls using input-only observations."""
import torch
from torch import nn
from torch.nn import functional as F
from .triplane_encoder import voxel_statistics


class Res3D(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.Conv3d(width,width,3,padding=1,bias=False),nn.GroupNorm(8,width),nn.SiLU(),nn.Conv3d(width,width,3,padding=1,bias=False),nn.GroupNorm(8,width))
    def forward(self,x):return F.silu(x+self.net(x))


class PlaneBlock(nn.Module):
    def __init__(self,width,kernel):
        super().__init__();self.input=nn.Conv2d(2*width+1,width,1)
        self.depth=nn.Conv2d(width,width,kernel,padding=(kernel[0]//2,kernel[1]//2),groups=width,bias=False)
        self.point=nn.Conv2d(width,width,1,bias=False);self.norm=nn.GroupNorm(8,width)
    def forward(self,observed,context,coverage):
        x=self.input(torch.cat((observed,context,coverage),1));return x+F.silu(self.norm(self.point(self.depth(x))))


def project(volume,mask,axis):
    m=mask.to(volume.dtype);coverage=m.mean(dim=axis)
    return (volume*m).sum(dim=axis)/m.sum(dim=axis).clamp_min(1),volume.mean(dim=axis),coverage


def lift(planes):
    xy,xz,yz=planes
    return xy.unsqueeze(2)+xz.unsqueeze(3)+yz.unsqueeze(4)


class Coupler(nn.Module):
    """G1: plane->voxel; G2: voxel update->plane feedback->voxel update."""
    def __init__(self,width,feedback=False):
        super().__init__();self.feedback=False
        self.planes=nn.ModuleList([PlaneBlock(width,(7,7)),PlaneBlock(width,(3,7)),PlaneBlock(width,(3,7))])
        self.update=nn.Conv3d(width,width,1);nn.init.zeros_(self.update.weight);nn.init.zeros_(self.update.bias)
        if feedback:self.add_feedback(width)
    def add_feedback(self,width):
        if self.feedback:raise ValueError('Feedback already registered')
        self.feedback=True
        self.refine=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,1,bias=False),nn.GroupNorm(8,width),nn.SiLU()) for _ in range(3)])
        self.update2=nn.Conv3d(width,width,1);nn.init.zeros_(self.update2.weight);nn.init.zeros_(self.update2.bias)
    def forward(self,x,mask,*,enable_feedback=True):
        planes=[block(*project(x,mask,axis)) for block,axis in zip(self.planes,(2,3,4))]
        first=x+self.update(lift(planes)/3)
        if not self.feedback or not enable_feedback:return first
        # Refined planes enter the second lift and therefore the final output.
        delta=first-x
        updated=[p+ref(delta.mean(dim=axis)) for p,ref,axis in zip(planes,self.refine,(2,3,4))]
        return first+self.update2(lift(updated)/3)


class Compact3DEncoder(nn.Module):
    def __init__(self,spec,variant='g0',activation_checkpointing=False):
        super().__init__()
        if variant not in ('g0','g1','g2'):raise ValueError('G0/G1/G2 only')
        self.spec=spec;self.variant=variant;self.activation_checkpointing=activation_checkpointing
        self.stem=nn.Sequential(nn.Conv3d(12,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.SiLU(),Res3D(32))
        self.level1=nn.Sequential(nn.Conv3d(32,64,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,64),nn.SiLU(),Res3D(64),Res3D(64))
        self.level2=nn.Sequential(nn.Conv3d(64,96,3,stride=2,padding=1,bias=False),nn.GroupNorm(8,96),nn.SiLU(),Res3D(96),Res3D(96))
        self.fpn=nn.ModuleList([nn.Conv3d(w,128,1) for w in (32,64,96)])
        if variant!='g0':
            # Build both shared single-pass modules before G2-only parameters,
            # so same-seed G1/G2 have identical common initialization.
            self.couplers=nn.ModuleList([Coupler(64),Coupler(96)])
            if variant=='g2':
                for width,coupler in zip((64,96),self.couplers):coupler.add_feedback(width)
    def _run(self,module,*args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(module,*args,use_reentrant=False)
        return module(*args)
    def forward_statistics(self,x,mask,*,enable_feedback=True):
        # Statistics are constructed in FP32; explicit FP16 weights or AMP
        # start only at the neural-network boundary.
        x=x.to(self.stem[0].weight.dtype);a=self._run(self.stem,x);b=self._run(self.level1,a)
        m1=F.max_pool3d(mask.float(),2)>0
        if self.variant!='g0':b=self.couplers[0](b,m1,enable_feedback=enable_feedback)
        c=self._run(self.level2,b);m2=F.max_pool3d(m1.float(),2)>0
        if self.variant!='g0':c=self.couplers[1](c,m2,enable_feedback=enable_feedback)
        a,b,c=[p(x) for p,x in zip(self.fpn,(a,b,c))]
        b=b+F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False)
        a=a+F.interpolate(b,size=a.shape[2:],mode='trilinear',align_corners=False)
        return (a,b,c)
    def forward(self,xyz):
        x,mask=voxel_statistics(xyz,self.spec,index_precision='float64')
        return self.forward_statistics(x,mask)
