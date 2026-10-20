"""Independent occupancy network. Pinned VoxDet trunk; no density/checkpoint path.

The classification TPV neck is retained; unused regression neck and detection
heads are removed exactly as in the earlier E2E encoder, without importing that
experiment's missing module. Construction only initializes random parameters.
"""
import torch
from torch import nn
import torch.nn.functional as F
from fpsgen.voxel_point.voxdet_density import VoxDetDensity
from fpsgen.voxel_shortcut.geometry import ShortcutGeometry
from fpsgen.voxel_shortcut.checkpointing import checkpoint_once_batchnorm
from fpsgen.voxel_shortcut.shortcut_unet import FilmBlock, batch_scalar


class OccupancyGridSpec(ShortcutGeometry):
    def valid_points(self, xyz):
        p=xyz.double()
        return (torch.isfinite(p).all(-1) & (p>p.new_tensor(self.origin)).all(-1)
                & (p<p.new_tensor(self.maximum)).all(-1) & (p.square().sum(-1)<self.sphere_radius**2))


def occupancy_lidar_features(xyz,spec):
    """Same six features, with exact configured float64 coarse-cell indexing."""
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not bool(torch.isfinite(xyz).all()):raise ValueError('Finite XYZ only')
    points=xyz[spec.valid_points(xyz)].float()
    if len(points)<2:raise ValueError('Too few valid partial LiDAR points')
    origin=points.new_tensor(spec.origin,dtype=torch.float64)
    size=points.new_tensor(spec.size,dtype=torch.float64)*2
    indices=torch.floor((points.double()-origin)/size).int()
    centers=((indices.double()+.5)*size+origin).float()
    features=torch.cat((points-centers,points[:,:2],points.norm(dim=-1,keepdim=True)/spec.xy_radius),-1)
    return features[None,None],[indices[None]]


class OccupancyConditionEncoder(VoxDetDensity):
    def __init__(self, geometry, activation_checkpointing=True):
        super().__init__(geometry, activation_checkpointing=activation_checkpointing)
        del self.main_head, self.aux_head
        for name in ('lateral_convs_reg','fpn_convs_reg','sp_tpvs_reg'):
            delattr(self.local_neck,name)

    def _run(self,module,*args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint_once_batchnorm(module,*args)
        return module(*args)

    def _classification_neck(self,inputs):
        neck=self.local_neck; laterals=list(inputs)
        for i in range(len(laterals)-2,-1,-1):
            up=F.interpolate(laterals[i+1],size=laterals[i].shape[2:],**neck.upsample_cfg)
            x=neck.sp_tpvs_cls[i](laterals[i])
            laterals[i]=neck.fpn_convs_cls[i](neck.lateral_convs_cls[i](torch.cat((x,up),1)))
        return laterals[0]

    def forward(self,lidar_xyz):
        scenes=[lidar_xyz] if isinstance(lidar_xyz,torch.Tensor) else list(lidar_xyz)
        token=torch.cat([self.lidar_tokenizer(*occupancy_lidar_features(x,self.spec)) for x in scenes],0)
        pyramid=self._run(self.voxel_backbone,token)
        voxel=self._run(self.voxel_neck,pyramid)[0]
        local=self._run(self.local_backbone,voxel)
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            cls=checkpoint(self._classification_neck,local,use_reentrant=False)
        else: cls=self._classification_neck(local)
        return tuple(self.spec.condition_to_field(x) for x in (cls,pyramid[1],pyramid[2]))


class OccupancyUNet(nn.Module):
    """State + input-hit mask; time for FM, learned constant embedding for BCE."""
    def __init__(self,objective,widths=(8,16,32,64),activation_checkpointing=True,embed_width=128):
        super().__init__()
        if len(widths)!=4 or any(w<8 or w%8 for w in widths): raise ValueError('Four widths divisible by8 required')
        self.objective=objective; self.widths=tuple(widths); self.checkpointing=activation_checkpointing
        if objective=='fm':
            self.register_buffer('frequencies',torch.exp(torch.linspace(0,6.9,16)),persistent=False)
            self.embedding=nn.Sequential(nn.Linear(32,embed_width),nn.SiLU(),nn.Linear(embed_width,embed_width))
        else: self.constant_embedding=nn.Parameter(torch.randn(1,embed_width)*.01)
        self.stem=nn.Conv3d(2,widths[0],3,padding=1)
        self.down=nn.ModuleList(nn.Conv3d(a,b,3,stride=2,padding=1) for a,b in zip(widths[:-1],widths[1:]))
        self.enc=nn.ModuleList(FilmBlock(w,embed_width) for w in widths)
        self.condition=nn.ModuleList(nn.Conv3d(128,w,1) for w in widths[1:])
        self.up=nn.ModuleList(nn.Conv3d(widths[i+1],widths[i],1) for i in range(3))
        self.dec=nn.ModuleList(FilmBlock(w,embed_width) for w in widths[:-1])
        self.output=nn.Conv3d(widths[0],1,3,padding=1)
        nn.init.normal_(self.output.weight,std=.001); nn.init.zeros_(self.output.bias)

    def _block(self,block,x,embedding):
        if self.training and self.checkpointing and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(block,x,embedding,use_reentrant=False)
        return block(x,embedding)

    def prepare_inference_context(self, observed, condition):
        """Project the fixed condition pyramid once for a multi-step FM sample.

        This inference-only context is deliberately separate from ``forward``:
        training and the default sampling path retain the original execution.
        The condition projection input dtype mirrors the first stem output under
        the active autocast policy (CUDA autocast uses its configured dtype).
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Stage1 condition caching is inference-only")
        if observed.ndim != 5 or observed.shape[1] != 1 or len(condition) != 3:
            raise ValueError("Scalar ZYX mask and three condition scales required")
        # Under CUDA autocast the stem produces the autocast dtype. Without
        # autocast (including CPU) it produces FP32 from the FP32 input/weights.
        if observed.device.type == "cuda" and torch.is_autocast_enabled():
            projection_input_dtype = torch.get_autocast_gpu_dtype()
        else:
            projection_input_dtype = self.stem.weight.dtype
        projected = tuple(
            layer(feature.to(projection_input_dtype))
            for layer, feature in zip(self.condition, condition)
        )
        return {"observed": observed, "condition_projected": projected}

    def forward_with_inference_context(self, state, context, time=None):
        return self(state, context["observed"], None, time,
                    inference_context=context)

    def forward(self,state,observed,condition,time=None,*,inference_context=None):
        if inference_context is not None:
            if inference_context.get("observed") is not observed:
                raise ValueError("Stage1 inference context belongs to a different observed mask")
            return self._forward(state, observed, None, time,
                                 condition_projected=inference_context["condition_projected"])
        return self._forward(state, observed, condition, time)

    def _forward(self,state,observed,condition,time=None,condition_projected=None):
        if (state.ndim!=5 or state.shape[1]!=1 or observed.shape!=state.shape
                or (condition_projected is None and (condition is None or len(condition)!=3))
                or (condition_projected is not None and len(condition_projected)!=3)):
            raise ValueError('Scalar ZYX state, input-only mask and three conditions required')
        if self.objective=='fm':
            if time is None: raise ValueError('FM requires time')
            t=batch_scalar(time,state)
            if not bool(((t>=0)&(t<=1)).all()): raise ValueError('t outside [0,1]')
            angle=t[:,None]*self.frequencies[None]
            embedding=self.embedding(torch.cat((angle.sin(),angle.cos()),1))
        else:
            if time is not None: raise ValueError('BCE does not accept time')
            embedding=self.constant_embedding.expand(len(state),-1)
        x=self.stem(torch.cat((state,observed.float()),1).to(self.stem.weight.dtype)); skips=[]
        for i in range(4):
            if i:
                x=self.down[i-1](x)
                c=(self.condition[i-1](condition[i-1].to(x.dtype))
                   if condition_projected is None else condition_projected[i-1])
                if c.shape!=x.shape: raise ValueError('VoxDet condition ZYX pyramid mismatch')
                x=x+c
            x=self._block(self.enc[i],x,embedding); skips.append(x)
        for i in reversed(range(3)):
            x=self.up[i](F.interpolate(x,size=skips[i].shape[-3:],mode='trilinear',align_corners=False))+skips[i]
            x=self._block(self.dec[i],x,embedding)
        return self.output(x).float()
