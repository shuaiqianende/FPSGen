"""Exact multi-neighbor adaptation of the project's Scene BVH CUDA index.

Morton order is only for search locality; returned IDs address original points.
No search radius, visit budget, or transport partitions are imposed.
"""
from functools import lru_cache
from pathlib import Path
import torch
from .local_neighbors import validate


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    build=Path(__file__).resolve().parents[1]/'runs/point_transport_bvh_knn_v1/cuda_build'
    build.mkdir(parents=True,exist_ok=True)
    return load(name='vggi_bvh_neighbors',sources=[str(Path(__file__).with_suffix('.cu'))],
                build_directory=str(build),extra_cuda_cflags=['-O3','--fmad=false'],verbose=False)


class BVHIndex:
    @torch.no_grad()
    def __init__(self,points,leaf_size=512):
        validate(points)
        if not isinstance(leaf_size,int) or leaf_size<1:raise ValueError('positive integer leaf size required')
        self.points=points.detach().contiguous()
        depth=min(len(points).bit_length()-1,max(0,((len(points)+leaf_size-1)//leaf_size-1).bit_length()))
        if depth>28:raise ValueError('BVH exceeds validated stack depth')
        self.leaves=1<<depth
        ext=extension()
        codes=ext.morton(self.points,self.points.amin(0),self.points.amax(0))
        self.ids=torch.argsort(codes)
        self.sorted_points=self.points[self.ids].contiguous()
        self.boxes,self.minima=ext.build_bvh(self.sorted_points,self.ids,self.leaves)

    @torch.no_grad()
    def query_index(self,queries,k=32):
        if queries.points.device!=self.points.device or k not in (1,32,64) or k>len(self.points):
            raise ValueError('same device and K=1/32/64 <= support count required')
        d,ids=extension().knn(queries.sorted_points,queries.ids,self.sorted_points,self.ids,
                             self.boxes,self.minima,self.leaves,k)
        return ids,d

    @torch.no_grad()
    def query(self,queries,k=32):
        return self.query_index(BVHIndex(queries),k)


@torch.no_grad()
def bidirectional_knn(x,y,k=32,leaf_size=None):
    if leaf_size is None:leaf_size=16 if k==1 else 512
    xi,yi=BVHIndex(x,leaf_size),BVHIndex(y,leaf_size)
    return yi.query_index(xi,k),xi.query_index(yi,k)
