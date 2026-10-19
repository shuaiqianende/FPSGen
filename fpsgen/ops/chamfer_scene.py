"""Independent scene-scale CUDA Chamfer: tiled exact search or Morton BVH.

Original fpsgen.ops.chamfer and chamfer_3D extension remain unchanged. Both modes
return squared distances and original-order nearest indices, with native CUDA
coordinate backward. No dense pair matrix, sampling or approximate visit budget.
"""
from pathlib import Path
import math
import torch
from torch import nn
from torch.autograd.function import once_differentiable

_EXTENSION=None
BUILD=Path('/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/chamfer_scene_v1')


def load_scene_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        BUILD.mkdir(parents=True,exist_ok=True)
        _EXTENSION=load(name='fpsgen_chamfer_scene_v1',sources=[str(Path(__file__).with_name('chamfer_scene_cuda.cu'))],
            build_directory=str(BUILD),extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)
    return _EXTENSION


def _validate(p):
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or len(p)>=2**31-1 or p.device.type!='cuda' or p.dtype!=torch.float32 or not bool(torch.isfinite(p).all()):
        raise ValueError('Nonempty finite CUDA FP32 [N,3], N<INT_MAX required')


class SceneIndex:
    """Single GPU Morton sort followed by bounded-depth balanced BVH build."""
    @torch.no_grad()
    def __init__(self,p,extension,leaf_size=32):
        depth=max(0,(math.ceil(len(p)/leaf_size)-1).bit_length())
        if depth>28:raise ValueError('BVH exceeds validated stack depth')
        self.leaves=1<<depth
        codes=extension.morton(p,p.amin(0).contiguous(),p.amax(0).contiguous())
        self.ids=torch.argsort(codes)
        self.points=p[self.ids].contiguous()
        self.boxes,self.minima=extension.build_bvh(self.points,self.ids,self.leaves)


class _SceneDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,q,mode,leaf_size):
        extension=load_scene_extension()
        if mode=='tiled':
            dp,ip=extension.tiled(p,q);dq,iq=extension.tiled(q,p)
        else:
            pi,qi=SceneIndex(p,extension,leaf_size),SceneIndex(q,extension,leaf_size)
            dp,ip=extension.bvh_query(pi.points,pi.ids,qi.points,qi.ids,qi.boxes,qi.minima,qi.leaves)
            dq,iq=extension.bvh_query(qi.points,qi.ids,pi.points,pi.ids,pi.boxes,pi.minima,pi.leaves)
        ctx.save_for_backward(p,q,ip,iq)
        ctx.mark_non_differentiable(ip,iq)
        return dp,dq,ip,iq

    @staticmethod
    @once_differentiable
    def backward(ctx,gdp,gdq,_ip,_iq):
        p,q,ip,iq=ctx.saved_tensors
        if gdp is None:gdp=p.new_zeros(len(p))
        if gdq is None:gdq=q.new_zeros(len(q))
        gp,gq=load_scene_extension().backward(p,q,ip,iq,gdp.contiguous(),gdq.contiguous())
        return gp,gq,None,None


class SceneChamferDistance(nn.Module):
    """B1/denseB/ragged-list distances. FP32-only, first-order coordinate gradients.

    Auto selection is purely point-count based, never uses GT membership/labels.
    ``bvh`` is exact spatial acceleration; ``tiled`` is independently optimized
    all-pairs CUDA. Original caller can retain its loss reduction convention.
    """
    def __init__(self,mode='auto',*,leaf_size=32,brute_force_pair_limit=2**30):
        super().__init__()
        if mode not in ('auto','tiled','bvh'):raise ValueError('Use auto/tiled/bvh')
        if leaf_size<8 or leaf_size>256:raise ValueError('leaf_size8..256 required')
        if brute_force_pair_limit<1:raise ValueError('Positive pair budget required')
        self.mode,self.leaf_size,self.brute_force_pair_limit=mode,leaf_size,brute_force_pair_limit

    def _single(self,p,q):
        _validate(p);_validate(q)
        if p.device!=q.device:raise ValueError('One admitted CUDA device required')
        mode=self.mode
        if mode=='auto':mode='tiled' if len(p)*len(q)<=self.brute_force_pair_limit else 'bvh'
        return _SceneDistance.apply(p.contiguous(),q.contiguous(),mode,self.leaf_size)

    def forward(self,p,q):
        if isinstance(p,(list,tuple)) or isinstance(q,(list,tuple)):
            if not isinstance(p,(list,tuple)) or not isinstance(q,(list,tuple)) or not p or len(p)!=len(q):
                raise ValueError('Equal nonempty scene lists required')
            return [self._single(x,y) for x,y in zip(p,q)]
        if p.ndim==3 or q.ndim==3:
            if p.ndim!=3 or q.ndim!=3 or len(p)!=len(q) or not len(p):raise ValueError('Matching dense batch sizes required')
            rows=[self._single(x,y) for x,y in zip(p,q)]
            return tuple(torch.stack([row[j] for row in rows]) for j in range(4))
        return self._single(p,q)

    def sqrt_loss(self,p,q):
        distances=self(p,q)
        if isinstance(distances,list):
            return torch.stack([self._reduce(d[0],d[1]) for d in distances]).mean()
        return self._reduce(distances[0],distances[1])

    @staticmethod
    def _reduce(dp,dq):
        # Zero-safe derivative0 at exact matches, rather than sqrt'(0)=Inf.
        def norm(d):return torch.where(d>0,d.clamp_min(1e-30).sqrt(),torch.zeros_like(d))
        return .5*(norm(dp).mean()+norm(dq).mean())
