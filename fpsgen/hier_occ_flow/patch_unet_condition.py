"""Scratch LiDAR-only PatchScene-style condition encoders; not official weights."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SpatialResBlock(nn.Module):
    """PatchScene-style GN/SiLU/conv residual block without time embedding."""
    def __init__(self, cin, cout, dropout=.1):
        super().__init__()
        self.first=nn.Sequential(nn.GroupNorm(32,cin),nn.SiLU(),nn.Conv3d(cin,cout,3,padding=1))
        self.second=nn.Sequential(nn.GroupNorm(32,cout),nn.SiLU(),nn.Dropout(dropout),nn.Conv3d(cout,cout,3,padding=1))
        nn.init.zeros_(self.second[-1].weight);nn.init.zeros_(self.second[-1].bias)
        self.skip=nn.Conv3d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x):return self.skip(x)+self.second(self.first(x))


class SpatialAttention(nn.Module):
    """Single-head F3 spatial attention; SDPA avoids materializing NxN weights."""
    def __init__(self,channels):
        super().__init__();self.outer=nn.GroupNorm(32,channels);self.inner=nn.GroupNorm(32,channels)
        self.qkv=nn.Conv1d(channels,3*channels,1);self.output=nn.Conv1d(channels,channels,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        x=F.silu(self.outer(x));shape=x.shape;b,c=shape[:2];v=x.reshape(b,c,-1)
        q,k,z=self.qkv(self.inner(v)).chunk(3,dim=1)
        q,k,z=(y.transpose(1,2).unsqueeze(1).contiguous() for y in (q,k,z))
        h=F.scaled_dot_product_attention(q,k,z).squeeze(1).transpose(1,2)
        return (v+self.output(h)).reshape(shape)


class PatchLidarCondition(nn.Module):
    """P0: PatchScene-like LiDAR-only down path or U1 adapted full U-Net."""
    def __init__(self,spec,variant='p0',activation_checkpointing=True):
        super().__init__()
        if variant not in ('p0','u1'):raise ValueError('Unknown registered variant')
        self.spec=spec;self.variant=variant;self.activation_checkpointing=activation_checkpointing
        self.input=nn.Conv3d(2,32,3,padding=1)
        self.down_blocks=nn.ModuleList([SpatialResBlock(32,32),SpatialResBlock(32,64),SpatialResBlock(64,128)])
        self.downsample=nn.ModuleList([nn.Conv3d(32,32,3,stride=2,padding=1),nn.Conv3d(64,64,3,stride=2,padding=1)])
        self.f3_attention=SpatialAttention(128)
        # Construct every P0/U1 shared tensor before adding U1-only layers.
        self.outputs=nn.ModuleList([nn.Conv3d(c,128,1) for c in (32,64,128)])
        if variant=='u1':
            self.downsample.append(nn.Conv3d(128,128,3,stride=2,padding=1))
            self.deep=SpatialResBlock(128,256)
            self.middle=nn.Sequential(SpatialResBlock(256,256),SpatialResBlock(256,256))
            self.up_blocks=nn.ModuleList([SpatialResBlock(256+128,128),SpatialResBlock(128+64,64),SpatialResBlock(64+32,32)])
            self.upsample=nn.ModuleList([nn.Conv3d(256,256,3,padding=1),nn.Conv3d(128,128,3,padding=1),nn.Conv3d(64,64,3,padding=1)])

    def _run(self,module,*args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():return checkpoint(module,*args,use_reentrant=False)
        return module(*args)

    def forward(self,xyz):
        scenes=[xyz] if isinstance(xyz,torch.Tensor) else list(xyz)
        # Input-only occupancy, in B,C,Z,Y,X, at the registered tokenizer grid.
        observed=torch.cat([self.spec.observed_mask(x) for x in scenes],0)
        observed=F.max_pool3d(observed.float(),2)
        x=self.input(torch.cat((observed,observed),dim=1));features=[]
        for i,block in enumerate(self.down_blocks):
            x=self._run(block,x)
            if i==2:x=self._run(self.f3_attention,x)
            features.append(x)
            if i<2:x=self.downsample[i](x)
        if self.variant=='u1':
            x=self._run(self.deep,self.downsample[2](x));x=self._run(self.middle,x)
            up=[]
            for i,skip in enumerate(reversed(features)):
                x=self.upsample[i](F.interpolate(x,size=skip.shape[2:],mode='nearest'))
                x=self._run(self.up_blocks[i],torch.cat((x,skip),1));up.append(x)
            features=list(reversed(up))
        return tuple(layer(x) for layer,x in zip(self.outputs,features))
