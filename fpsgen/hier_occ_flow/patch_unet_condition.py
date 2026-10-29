"""Scratch LiDAR-only PatchScene-style condition encoders; not official weights."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .gctv_encoder import PlaneBlock,project,lift


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


class TriPlaneInteraction(nn.Module):
    """Input-mask-guided three-plane interaction, with reusable plane features."""
    def __init__(self,width):
        super().__init__()
        self.planes=nn.ModuleList([PlaneBlock(width,(7,7)),PlaneBlock(width,(3,7)),PlaneBlock(width,(3,7))])
        self.update=nn.Conv3d(width,width,1)
        nn.init.zeros_(self.update.weight);nn.init.zeros_(self.update.bias)
    def forward(self,x,mask,memory=None,fusion=None):
        planes=tuple(block(*project(x,mask,axis)) for block,axis in zip(self.planes,(2,3,4)))
        if memory is not None:
            coverage=tuple(mask.float().mean(dim=axis) for axis in (2,3,4))
            planes=fusion(memory,planes,coverage)
        return x+self.update(lift(planes)/3),planes


class PlaneMemory(nn.Module):
    def __init__(self,width):
        super().__init__();self.down=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(3)]);self.up=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(3)])
        self.norm=nn.GroupNorm(8,width);self.gates=nn.ModuleList([nn.Conv2d(3*width+1,1,1) for _ in range(3)])
    def forward(self,down,up,coverage):
        result=[]
        for a,b,c,da,ub,g in zip(down,up,coverage,self.down,self.up,self.gates):
            a,b=da(a),ub(b);diff=(self.norm(a)-self.norm(b)).abs();w=torch.sigmoid(g(torch.cat((a,b,diff,c.to(a.dtype)),1)));result.append(w*a+(1-w)*b)
        return tuple(result)

class DirectionalContext(nn.Module):
    def __init__(self,width,adaptive=True):
        super().__init__();self.adaptive=adaptive;self.ops=nn.ModuleList([nn.Conv3d(width,width,k,padding=tuple(x//2 for x in k),groups=width,bias=False) for k in ((1,7,7),(3,1,9),(3,9,1))])
        self.gate=nn.Conv3d(2*width,3,1) if adaptive else None
        self.output=nn.Conv3d(width,width,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,planes):
        if self.adaptive:w=torch.softmax(self.gate(torch.cat((x,lift(planes)/3),1)).float(),dim=1).to(x.dtype)
        else:w=x.new_full((x.shape[0],3,*x.shape[2:]),1/3)
        out=sum(op(x)*w[:,i:i+1] for i,op in enumerate(self.ops))
        return x+self.output(out)

class PatchLidarCondition(nn.Module):
    """P0: PatchScene-like LiDAR-only down path or U1 adapted full U-Net."""
    def __init__(self,spec,variant='p0',activation_checkpointing=True):
        super().__init__()
        if variant not in ('p0','u1','u1a','u1ab','u1abc','u1ac','u1abctl','u1acctl'):raise ValueError('Unknown registered variant')
        self.spec=spec;self.variant=variant;self.activation_checkpointing=activation_checkpointing
        self.input=nn.Conv3d(2,32,3,padding=1)
        self.down_blocks=nn.ModuleList([SpatialResBlock(32,32),SpatialResBlock(32,64),SpatialResBlock(64,128)])
        self.downsample=nn.ModuleList([nn.Conv3d(32,32,3,stride=2,padding=1),nn.Conv3d(64,64,3,stride=2,padding=1)])
        self.f3_attention=SpatialAttention(128)
        # Construct every P0/U1 shared tensor before adding U1-only layers.
        self.outputs=nn.ModuleList([nn.Conv3d(c,128,1) for c in (32,64,128)])
        if variant!='p0':
            self.downsample.append(nn.Conv3d(128,128,3,stride=2,padding=1))
            self.deep=SpatialResBlock(128,256)
            self.middle=nn.Sequential(SpatialResBlock(256,256),SpatialResBlock(256,256))
            self.up_blocks=nn.ModuleList([SpatialResBlock(256+128,128),SpatialResBlock(128+64,64),SpatialResBlock(64+32,32)])
            self.upsample=nn.ModuleList([nn.Conv3d(256,256,3,padding=1),nn.Conv3d(128,128,3,padding=1),nn.Conv3d(64,64,3,padding=1)])
        if variant not in ('p0','u1'):
            # Four separate down-F1/F2 and up-F2/F1 modules after all shared
            # U1 parameters: scratch initialization of the base is preserved.
            self.plane_down=nn.ModuleList([TriPlaneInteraction(32),TriPlaneInteraction(64)])
            self.plane_up=nn.ModuleList([TriPlaneInteraction(64),TriPlaneInteraction(32)])
        if variant in ('u1ab','u1abc','u1abctl'):self.direction=DirectionalContext(64,adaptive=variant!='u1abctl')
        if variant in ('u1ac','u1abc'):self.memory=nn.ModuleList([PlaneMemory(64),PlaneMemory(32)])

    def _run(self,module,*args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():return checkpoint(module,*args,use_reentrant=False)
        return module(*args)

    def forward(self,xyz):
        scenes=[xyz] if isinstance(xyz,torch.Tensor) else list(xyz)
        # Input-only occupancy, in B,C,Z,Y,X, at the registered tokenizer grid.
        observed=torch.cat([self.spec.observed_mask(x) for x in scenes],0)
        observed=F.max_pool3d(observed.float(),2)
        masks=(observed>0,F.max_pool3d(observed,2)>0)
        x=self.input(torch.cat((observed,observed),dim=1));features=[];downplanes=[];has_a=self.variant not in ('p0','u1')
        for i,block in enumerate(self.down_blocks):
            x=self._run(block,x)
            if has_a and i<2:
                x,planes=self._run(self.plane_down[i],x,masks[i]);downplanes.append(planes)
                if i==1 and hasattr(self,'direction'):x=self._run(self.direction,x,planes)
            if i==2:x=self._run(self.f3_attention,x)
            features.append(x)
            if i<2:x=self.downsample[i](x)
        if self.variant!='p0':
            x=self._run(self.deep,self.downsample[2](x));x=self._run(self.middle,x)
            up=[]
            for i,skip in enumerate(reversed(features)):
                x=self.upsample[i](F.interpolate(x,size=skip.shape[2:],mode='nearest'))
                x=self._run(self.up_blocks[i],torch.cat((x,skip),1))
                if has_a and i>0:
                    if hasattr(self,'memory'):x,_=self._run(self.plane_up[i-1],x,masks[2-i],downplanes[2-i],self.memory[i-1])
                    else:x,_=self._run(self.plane_up[i-1],x,masks[2-i])
                up.append(x)
            features=list(reversed(up))
        return tuple(layer(x) for layer,x in zip(self.outputs,features))
