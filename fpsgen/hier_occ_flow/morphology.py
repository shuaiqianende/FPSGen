"""Parent-preserving monotone fine closing with tiled, bounded bool workspace.

The output is an augmented label version, not additional measured map hits.
No coarse parent or original hit is removed or added. Halo=2*radius reproduces
global dilation-then-erosion before restriction to the original parent domain.
"""
from dataclasses import asdict,dataclass
import hashlib,json
import numpy as np
from scipy.ndimage import binary_closing,generate_binary_structure
from .voxelization import unpack_children,pack_children,occupied_fine_coords

MORPH_DATA_VERSION='hs-occ-morph-close-v1'


@dataclass(frozen=True)
class FineClosingSpec:
    connectivity: int=26
    iterations: int=1
    tile_parent_xy: int=64
    method: str='parent_preserving_monotone_closing'

    def __post_init__(self):
        if self.connectivity not in (6,26) or self.iterations!=1 or type(self.tile_parent_xy) is not int or self.tile_parent_xy<1:
            raise ValueError('Registered connectivity6/26, one closing and positive tile size required')
        if self.method!='parent_preserving_monotone_closing':raise ValueError('Unknown target transform')

    def to_dict(self):return asdict(self)
    @property
    def sha256(self):return hashlib.sha256(json.dumps(self.to_dict(),sort_keys=True).encode()).hexdigest()
    @property
    def footprint(self):return generate_binary_structure(3,1 if self.connectivity==6 else 3)


def close_fine_masks(parent_ids,masks,geometry,spec):
    if geometry.child_split!=(2,2,2):raise ValueError('This target version is8-way only')
    values=np.asarray(masks)
    if values.dtype.kind not in 'iu' or np.any(values<1) or np.any(values>255):raise ValueError('Valid nonzero8-bit masks required')
    ids=np.asarray(parent_ids,dtype=np.int64);old=values.astype(np.uint8)
    if ids.ndim!=1 or old.shape!=ids.shape or not len(ids) or np.any(ids[1:]<=ids[:-1]) or np.any(old==0):
        raise ValueError('Sorted unique occupied parents and nonzero masks required')
    parents=geometry.unflatten(ids);original=unpack_children(old,geometry)
    valid=geometry.valid_children(ids)
    if np.any(original&~valid):raise ValueError('Original GT contains ROI-invalid children')
    fine=occupied_fine_coords(ids,old,geometry)
    result=original.copy();tile=spec.tile_parent_xy
    tiles_x=(geometry.parent_resolution[0]+tile-1)//tile
    keys=parents[:,0]//tile+tiles_x*(parents[:,1]//tile)
    offsets=geometry.child_offsets();nx,ny,nz=geometry.fine_resolution
    peak_cells=0
    for key in np.unique(keys):
        tx=int(key%tiles_x);ty=int(key//tiles_x)
        # radius1 for each of dilation and erosion requires two input cells.
        xlo=max(0,tx*tile*2-2);xhi=min(nx,(tx+1)*tile*2+2)
        ylo=max(0,ty*tile*2-2);yhi=min(ny,(ty+1)*tile*2+2)
        selected=(fine[:,0]>=xlo)&(fine[:,0]<xhi)&(fine[:,1]>=ylo)&(fine[:,1]<yhi)
        points=fine[selected]
        block=np.zeros((nz,yhi-ylo,xhi-xlo),dtype=bool)
        block[points[:,2],points[:,1]-ylo,points[:,0]-xlo]=True
        closed=binary_closing(block,structure=spec.footprint,iterations=1,border_value=0)
        query=np.flatnonzero(keys==key)
        coord=parents[query,None,:]*2+offsets[None]
        result[query]|=closed[coord[...,2],coord[...,1]-ylo,coord[...,0]-xlo]&valid[query]
        peak_cells=max(peak_cells,block.size)
    packed=pack_children(result,geometry)
    if not np.array_equal(packed&old,old) or np.any(packed==0):raise RuntimeError('Original hits/parents changed')
    old_count=int(original.sum());new_count=int(result.sum())
    return packed,{'parents':len(ids),'original_fine':old_count,'new_fine':new_count,
        'added_fine':new_count-old_count,'added_fraction':(new_count-old_count)/old_count,
        'original_hits_preserved':True,'parent_ids_unchanged':True,'invalid_children_added':0,
        'peak_bool_tile_cells':peak_cells,'transform_sha256':spec.sha256}
