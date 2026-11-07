"""Linear-memory CUDA CSR transpose. Graph edges and costs are preserved."""
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    build=Path(__file__).resolve().parents[1]/'runs/point_transport_soft_graph_opt_v1/cuda_build'
    build.mkdir(parents=True,exist_ok=True)
    return load(name='vggi_soft_graph',sources=[str(Path(__file__).with_suffix('.cu'))],
                build_directory=str(build),extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)


def decode(keys,x,y):
    s=torch.empty_like(keys);t=torch.empty_like(keys);c=torch.empty(len(keys),device=x.device,dtype=x.dtype)
    extension().decode(keys,x.contiguous(),y.contiguous(),s,t,c)
    return s,t,c


def transpose(s,t,cost,offsets,n):
    sc=torch.empty_like(s);ct=torch.empty_like(cost)
    cursors=torch.zeros(n,device=s.device,dtype=torch.int32)
    extension().transpose(s,t,cost,offsets,cursors,sc,ct)
    return sc,ct
