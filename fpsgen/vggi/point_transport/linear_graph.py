"""Exact KNN union without a global edge sort. Unused capacity is never read."""
from functools import lru_cache
from pathlib import Path
import torch

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    folder=Path(__file__).resolve().parents[1]/'runs/point_transport_linear_v1/cuda_build'
    folder.mkdir(parents=True,exist_ok=True)
    return load(name='vggi_linear_graph',sources=[str(Path(__file__).with_suffix('.cu'))],build_directory=str(folder),extra_cuda_cflags=['-O3','--fmad=false'])

def graph(x,y,xi,yi,k,cost_dtype=torch.float32,hints=None):
    if cost_dtype!=torch.float32:
        raise ValueError('Linear graph uses FP32 geometry and costs')
    f=yi.query(xi,k,None if hints is None else hints.get("f")).sort(dim=-1).values
    r=f if xi is yi else xi.query(yi,k,None if hints is None else hints.get("r"))
    if hints is not None:hints.update(f=f,r=r)
    ro,co,t,c,s,ct=extension().graph(f,r,x.contiguous(),y.contiguous())
    return dict(keys=None,s=None,t=t,cost=c,offsets=ro,offsets_t=co,sc=s,cost_t=ct)
