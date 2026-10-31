"""Depth-moment residual lift; a hypothesis, not a validated innovation."""
import torch
from torch import nn
from .patch_unet_condition import PatchLidarCondition

class MomentResidual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.mix=nn.ModuleList([nn.Conv2d(width*2,width*2,1) for _ in range(3)])
        for m in self.mix:
            nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x):
        result=torch.zeros_like(x)
        for axis,layer in zip((2,3,4),self.mix):
            n=x.shape[axis];shape=[1]*5;shape[axis]=n
            t=(2*(torch.arange(n,device=x.device,dtype=torch.float32)+.5)/n-1).reshape(shape).to(x.dtype)
            b2=t.square()-t.square().mean()
            m1=(x*t).mean(axis);m2=(x*b2).mean(axis)
            a,b=layer(torch.cat((m1,m2),1)).chunk(2,1)
            result=result+a.unsqueeze(axis)*t+b.unsqueeze(axis)*b2
        return result/3

class DepthMomentEncoder(PatchLidarCondition):
    def __init__(self,spec,activation_checkpointing=True):
        super().__init__(spec,'u1a',activation_checkpointing)
        # All base modules are built first, preserving shared scratch tensors.
        for m,w in zip(list(self.plane_down)+list(self.plane_up),(32,64,64,32)):
            m.depth_moment=MomentResidual(w)
