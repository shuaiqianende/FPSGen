"""Raw-resolution 3D velocity U-Net, low-resolution additive VoxDet condition."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def batch_scalar(value, state):
    value = torch.as_tensor(value,device=state.device,dtype=torch.float32).reshape(-1)
    if value.numel()==1:
        return value.expand(len(state))
    if value.shape != (len(state),):
        raise ValueError("One time/step per scene required")
    return value


class TimeStepEmbedding(nn.Module):
    def __init__(self, width=128, bands=16):
        super().__init__()
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),bands)),persistent=False)
        self.mlp = nn.Sequential(nn.Linear(4*bands+1,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,t,h,instantaneous):
        t,h=t[:,None]*self.frequencies[None],h[:,None]*self.frequencies[None]
        features=torch.cat((t.sin(),t.cos(),h.sin(),h.cos(),torch.full_like(t[:,:1],float(instantaneous))),1)
        return self.mlp(features)


class FilmBlock(nn.Module):
    def __init__(self, channels, embed_width):
        super().__init__()
        groups = min(8,channels)
        self.norm1=nn.GroupNorm(groups,channels);self.norm2=nn.GroupNorm(groups,channels)
        self.conv1=nn.Conv3d(channels,channels,3,padding=1);self.conv2=nn.Conv3d(channels,channels,3,padding=1)
        self.film=nn.Linear(embed_width,2*channels)

    def forward(self,x,embedding):
        scale,shift=self.film(embedding).to(x.dtype).chunk(2,1)
        y=self.norm1(x)*(1+scale[:,:,None,None,None])+shift[:,:,None,None,None]
        y=self.conv1(F.silu(y));y=self.conv2(F.silu(self.norm2(y)))
        return x+y


class ShortcutUNet(nn.Module):
    def __init__(self, widths=(8,16,32,64), condition_channels=128, embed_width=128, activation_checkpointing=True):
        super().__init__()
        if len(widths)!=4 or any(w<8 or w%8 for w in widths):
            raise ValueError("Four widths divisible by8 required")
        self.widths=tuple(widths);self.activation_checkpointing=activation_checkpointing
        self.embedding=TimeStepEmbedding(embed_width)
        self.stem=nn.Conv3d(1,widths[0],3,padding=1)
        self.down=nn.ModuleList(nn.Conv3d(a,b,3,stride=2,padding=1) for a,b in zip(widths[:-1],widths[1:]))
        self.enc=nn.ModuleList(FilmBlock(w,embed_width) for w in widths)
        self.condition=nn.ModuleList(nn.Conv3d(condition_channels,w,1) for w in widths[1:])
        self.up=nn.ModuleList(nn.Conv3d(widths[i+1],widths[i],1) for i in range(3))
        self.dec=nn.ModuleList(FilmBlock(w,embed_width) for w in widths[:-1])
        self.output=nn.Conv3d(widths[0],1,3,padding=1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def _block(self,module,x,embedding):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(module,x,embedding,use_reentrant=False)
        return module(x,embedding)

    def forward(self,state,t,h,condition,*,instantaneous=False):
        if state.ndim!=5 or state.shape[1]!=1 or len(condition)!=3:
            raise ValueError("Full[B,1,Z,Y,X] and three coarse conditions required")
        t,h=batch_scalar(t,state),batch_scalar(h,state)
        if bool(((t<0)|(t>1)|(h<0)|(t+h>1.000001)).any()) or (not instantaneous and bool((h<=0).any())):
            raise ValueError("Valid t,h witht+h<=1;h=0 only instantaneous")
        if instantaneous and bool((h!=0).any()):
            raise ValueError("Instantaneous branch requires h=0")
        embedding=self.embedding(t,h,instantaneous)
        x=self.stem(state.to(self.stem.weight.dtype));skips=[]
        for i in range(4):
            if i:
                x=self.down[i-1](x)
                c=self.condition[i-1](condition[i-1].to(x.dtype))
                if c.shape != x.shape:
                    raise ValueError("Condition grid must match U-Net coarse grid; no full128-channel expansion")
                x=x+c
            x=self._block(self.enc[i],x,embedding)
            skips.append(x)
        for i in reversed(range(3)):
            x=self.up[i](F.interpolate(x,size=skips[i].shape[-3:],mode='trilinear',align_corners=False))+skips[i]
            x=self._block(self.dec[i],x,embedding)
        return self.output(x).float()
