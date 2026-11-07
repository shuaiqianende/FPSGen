"""GPU rejection sampling of the continuous squared B-spline amplitude.

FP32 hierarchy of 256-way CDFs avoids one huge low-resolution prefix sum.
Each output has its own Philox stream and completes its own rejection loop;
there is no first-finished atomic compaction bias. No extra noise or threshold.
"""
from functools import lru_cache
from pathlib import Path
import time
import torch
import torch.nn.functional as F

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    folder=Path(__file__).resolve().parents[1]/'runs/field_sampler_cuda_v1/cuda_build'
    folder.mkdir(parents=True,exist_ok=True)
    return load(name='vggi_field_sampler',sources=[str(Path(__file__).with_suffix('.cu'))],build_directory=str(folder),extra_cuda_cflags=['-O3','--fmad=false'])

class CUDAFieldSampler:
    @torch.no_grad()
    def __init__(self,amplitude,origin,spacing,device='cuda'):
        a=torch.as_tensor(amplitude,device=device,dtype=torch.float32).contiguous()
        self.origin=torch.as_tensor(origin,device=a.device,dtype=torch.float32).contiguous()
        self.spacing=torch.as_tensor(spacing,device=a.device,dtype=torch.float32).contiguous()
        if a.ndim!=3 or min(a.shape)<1 or a.numel()>=2**31 or self.origin.shape!=(3,) or self.spacing.shape!=(3,):raise ValueError('Invalid XYZ geometry')
        if not bool(torch.isfinite(a).all()&torch.isfinite(self.origin).all()&torch.isfinite(self.spacing).all()&(self.spacing>0).all()):raise ValueError('Finite amplitude and positive spacing required')
        scale=a.abs().max()
        if not bool(scale>0):raise ValueError('Cannot sample empty field')
        # One global scalar cancels in acceptance and categorical probabilities.
        self.a=(a/scale).contiguous()
        self.upper=F.max_pool3d(self.a.abs()[None,None],5,1,2).flatten().square()*(1+1e-5)
        def level(weights):
            return F.pad(weights,(0,(-weights.numel())%256)).reshape(-1,256).cumsum(1)
        self.c0=level(self.upper);self.c1=level(self.c0[:,-1]);self.c2=self.c1[:,-1].cumsum(0)
        self.envelope_mass=float(self.c2[-1])*float(scale)**2*float(self.spacing.prod())
        extension()

    @torch.no_grad()
    def sample(self,count=180000,*,seed=20261107,max_attempts_per_point=100000):
        if not isinstance(count,int) or not 0<count<2**31 or not isinstance(max_attempts_per_point,int) or not 0<max_attempts_per_point<2**31:raise ValueError('Positive count/attempt limit required')
        if not isinstance(seed,int) or not 0<=seed<2**63:raise ValueError('Seed must be nonnegative int64')
        with torch.cuda.device(self.a.device):
            torch.cuda.synchronize();start=time.perf_counter()
            out,tries,errors=extension().sample(self.a,self.upper,self.c0,self.c1,self.c2,self.origin,self.spacing,count,seed,max_attempts_per_point)
            error=int(errors.item());proposals=int(tries.long().sum().item())
            if error:raise RuntimeError(f'{error} samples exhausted attempts or violated envelope; no partial output returned')
            if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite sample coordinates')
            return out,dict(points=count,proposals=proposals,acceptance=count/proposals,seconds=time.perf_counter()-start,seed=seed,envelope_mass=self.envelope_mass,
                method='cuda_continuous_squared_amplitude_rejection_v1',arithmetic='fp32',extra_noise=False,threshold=None)
