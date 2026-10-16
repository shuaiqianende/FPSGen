"""XYZ world/index contract, X-fastest IDs, open sphere/height ROI.

Voxel validity means positive-volume intersection with the ROI, not center inclusion.
World indexing is evaluated in float64 on both CPU and CUDA. Float32 input points
are interpreted as their stored values; no snapping or boundary epsilon is applied.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np


def is_torch(value):
    return value.__class__.__module__.startswith("torch")


@dataclass(frozen=True)
class SparseHierarchyGeometry:
    parent_resolution: tuple = (256, 256, 32)
    child_split: tuple = (3, 3, 2)
    origin: tuple = (-50., -50., -4.)
    maximum: tuple = (50., 50., 4.4)
    sphere_radius: float = 50.

    def __post_init__(self):
        for name in ("parent_resolution", "child_split", "origin", "maximum"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(len(v) != 3 for v in (self.parent_resolution, self.child_split, self.origin, self.maximum)):
            raise ValueError("Geometry requires three XYZ components")
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in self.parent_resolution + self.child_split):
            raise ValueError("Positive integer resolutions/splits required")
        if self.k > 32 or self.num_parents > np.iinfo(np.uint32).max:
            raise ValueError("Packed format supports K<=32 and uint32 parent IDs")
        if not np.isfinite([*self.origin, *self.maximum, self.sphere_radius]).all() or self.sphere_radius <= 0 or any(a >= b for a, b in zip(self.origin, self.maximum)):
            raise ValueError("Invalid world bounds")

    @property
    def k(self): return int(np.prod(self.child_split))
    @property
    def num_parents(self): return int(np.prod(self.parent_resolution))
    @property
    def fine_resolution(self): return tuple(n*s for n,s in zip(self.parent_resolution,self.child_split))
    @property
    def parent_size(self): return tuple((b-a)/n for a,b,n in zip(self.origin,self.maximum,self.parent_resolution))
    @property
    def fine_size(self): return tuple((b-a)/n for a,b,n in zip(self.origin,self.maximum,self.fine_resolution))
    @property
    def mask_dtype(self): return np.dtype("u1" if self.k <= 8 else "<u4")
    @property
    def sha256(self): return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    def to_dict(self): return asdict(self)

    def _array(self, values, like, *, integer=False):
        if is_torch(like):
            import torch
            return torch.tensor(values,device=like.device,dtype=torch.long if integer else torch.float64)
        return np.asarray(values,dtype=np.int64 if integer else np.float64)

    def voxelize_points(self, points, *, strict=False):
        if points.ndim != 2 or points.shape[1] != 3: raise ValueError("XYZ only [N,3] required")
        if is_torch(points):
            import torch
            p=points.double(); valid=torch.isfinite(p).all(-1)
        else:
            p=np.asarray(points,dtype=np.float64); valid=np.isfinite(p).all(-1)
        valid &= (p > self._array(self.origin,p)).all(-1) & (p < self._array(self.maximum,p)).all(-1)
        valid &= (p*p).sum(-1) < self.sphere_radius**2
        if strict and not bool(valid.all()): raise ValueError(f"{int((~valid).sum())} invalid GT points")
        scaled=(p[valid]-self._array(self.origin,p))/self._array(self.fine_size,p)
        if is_torch(p):
            import torch
            coord=torch.floor(scaled).long()
        else: coord=np.floor(scaled).astype(np.int64)
        return coord,valid

    def flatten(self, coord, *, fine=False):
        shape=self.fine_resolution if fine else self.parent_resolution
        if coord.shape[-1] != 3 or bool(((coord<0)|(coord>=self._array(shape,coord,integer=True))).any()):
            raise ValueError("XYZ grid coordinates out of range")
        x,y,z=coord[...,0],coord[...,1],coord[...,2]
        return x+shape[0]*(y+shape[1]*z)

    def unflatten(self, ids, *, fine=False):
        shape=self.fine_resolution if fine else self.parent_resolution
        if bool(((ids<0)|(ids>=int(np.prod(shape)))).any()): raise ValueError("Voxel ID out of range")
        values=(ids%shape[0],(ids//shape[0])%shape[1],ids//(shape[0]*shape[1]))
        if is_torch(ids):
            import torch
            return torch.stack(values,-1).long()
        return np.stack(values,-1).astype(np.int64)

    def split_fine(self, fine_coord):
        split=self._array(self.child_split,fine_coord,integer=True)
        parent=fine_coord//split; local=fine_coord%split
        child=local[...,0]+self.child_split[0]*(local[...,1]+self.child_split[1]*local[...,2])
        return parent,child

    def child_offsets(self, like=None):
        ids=np.arange(self.k,dtype=np.int64)
        sx,sy,_=self.child_split
        offsets=np.stack((ids%sx,(ids//sx)%sy,ids//(sx*sy)),-1)
        return self._array(offsets,like,integer=True) if like is not None else offsets

    def compose(self, parent_coord, child_id):
        if bool(((child_id<0)|(child_id>=self.k)).any()): raise ValueError("Child ID out of range")
        return parent_coord*self._array(self.child_split,parent_coord,integer=True)+self.child_offsets(parent_coord)[child_id]

    def centers(self, coords, *, fine=False):
        size=self.fine_size if fine else self.parent_size
        return self._array(self.origin,coords)+(coords+.5)*self._array(size,coords)

    def valid_children(self, parent_ids):
        coord=self.unflatten(parent_ids)
        fine=coord[:,None,:]*self._array(self.child_split,coord,integer=True)+self.child_offsets(coord)[None]
        lo=self.centers(fine,fine=True)-self._array(self.fine_size,fine)*.5
        hi=lo+self._array(self.fine_size,fine)
        if is_torch(parent_ids):
            import torch
            closest=torch.clamp_min(lo,0)+torch.clamp_max(hi,0)
        else: closest=np.maximum(lo,0)+np.minimum(hi,0)
        return (closest*closest).sum(-1)<self.sphere_radius**2
