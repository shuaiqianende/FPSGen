"""PatchScene-inspired 3D velocity U-Net for forward-time occupancy FM."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .stage1_network import batch_scalar

class TimeResidual(nn.Module):
    def __init__(self,cin,cout,dim):
        super().__init__();self.norm1=nn.GroupNorm(8,cin);self.conv1=nn.Conv3d(cin,cout,3,padding=1)
        self.time=nn.Sequential(nn.SiLU(),nn.Linear(dim,cout));self.norm2=nn.GroupNorm(8,cout);self.conv2=nn.Conv3d(cout,cout,3,padding=1)
        nn.init.zeros_(self.conv2.weight);nn.init.zeros_(self.conv2.bias)
        self.skip=nn.Conv3d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,t):
        h=self.conv1(F.silu(self.norm1(x)));h=h+self.time(t).to(h.dtype)[:,:,None,None,None]
        return self.skip(x)+self.conv2(F.silu(self.norm2(h)))

class SpatialAttention(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__();self.heads=heads;self.norm=nn.GroupNorm(8,width);self.qkv=nn.Conv1d(width,3*width,1);self.out=nn.Conv1d(width,width,1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):
        b,c,z,y,w=x.shape;q,k,v=self.qkv(self.norm(x).reshape(b,c,-1)).chunk(3,1);n=z*y*w
        q,k,v=(a.reshape(b,self.heads,c//self.heads,n).transpose(2,3).contiguous() for a in (q,k,v))
        h=F.scaled_dot_product_attention(q,k,v).transpose(2,3).reshape(b,c,n)
        return x+self.out(h).reshape_as(x)

class PatchFlowUNet(nn.Module):
    """Same input/condition interface; FM t=0 noise, t=1 data, FP32 output."""
    def __init__(self,objective='fm',widths=(8,16,32,64),activation_checkpointing=True,embed_width=128):
        super().__init__()
        if objective!='fm' or tuple(widths)!=(8,16,32,64):raise ValueError('Registered FM widths only')
        self.objective=objective;self.widths=tuple(widths);self.checkpointing=activation_checkpointing
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,6.9,16)),persistent=False)
        self.embedding=nn.Sequential(nn.Linear(32,embed_width),nn.SiLU(),nn.Linear(embed_width,embed_width))
        self.stem=nn.Conv3d(2,8,3,padding=1);self.downsample=nn.ModuleList([nn.Conv3d(a,b,3,stride=2,padding=1) for a,b in zip(widths[:-1],widths[1:])])
        self.down=nn.ModuleList([TimeResidual(w,w,embed_width) for w in widths]);self.condition=nn.ModuleList([nn.Conv3d(128,w,1) for w in widths[1:]])
        self.mid1=TimeResidual(64,64,embed_width);self.attn=SpatialAttention(64);self.mid2=TimeResidual(64,64,embed_width)
        self.upconv=nn.ModuleList([nn.Conv3d(widths[i+1],widths[i],3,padding=1) for i in range(3)])
        self.up=nn.ModuleList([TimeResidual(2*widths[i],widths[i],embed_width) for i in range(3)])
        self.out_norm=nn.GroupNorm(8,8);self.output=nn.Conv3d(8,1,3,padding=1);nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)
    def _run(self,m,*args):
        if self.training and self.checkpointing and torch.is_grad_enabled():return checkpoint(m,*args,use_reentrant=False)
        return m(*args)
    def forward(self,state,observed,condition,time=None):
        if state.ndim!=5 or state.shape[1]!=1 or observed.shape!=state.shape or condition is None or len(condition)!=3 or time is None:raise ValueError('Expected state, input-only mask, three scales and FM time')
        t=batch_scalar(time,state)
        if not bool(((t>=0)&(t<=1)).all()):raise ValueError('FM time must be in [0,1]')
        angle=t[:,None]*self.frequencies[None];emb=self.embedding(torch.cat((angle.sin(),angle.cos()),1).to(self.embedding[0].weight.dtype))
        x=self.stem(torch.cat((state,observed.float()),1).to(self.stem.weight.dtype));skips=[];conditions=[]
        for i,block in enumerate(self.down):
            if i:
                x=self.downsample[i-1](x);c=self.condition[i-1](condition[i-1].to(x.dtype))
                if c.shape!=x.shape:raise ValueError('Condition shape mismatch')
                conditions.append(c);x=x+c
            x=self._run(block,x,emb);skips.append(x)
        x=self._run(self.mid1,x,emb);x=self._run(self.attn,x);x=self._run(self.mid2,x,emb)
        for i in reversed(range(3)):
            x=self.upconv[i](F.interpolate(x,size=skips[i].shape[2:],mode='nearest'))
            x=self._run(self.up[i],torch.cat((x,skips[i]),1),emb)
            if i>0:x=x+conditions[i-1]
        return self.output(F.silu(self.out_norm(x))).float()
