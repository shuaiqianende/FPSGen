"""Index FPSGen's map, then reuse its exact pre-Poisson candidate function."""
from argparse import Namespace
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import cKDTree


class RawMapSource:
    def __init__(self,data_root,*,max_range=50.,min_z=-4.,max_z=4.4,viewpoint_voxel_size=10.,fast_viewpoint=False):
        self.root=Path(data_root); self.sequence=None
        self.fast_viewpoint=fast_viewpoint
        self.args=Namespace(max_range=max_range,min_z=min_z,max_z=max_z,viewpoint_voxel_size=viewpoint_voxel_size)

    def _load(self,sequence):
        scripts=Path(__file__).resolve().parents[2]/'scripts'
        if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
        from prepare_semantickitti import load_map,load_poses
        if sequence!=self.sequence:
            self.tree=None; self.xyz=None; self.labels=None
            directory=self.root/sequence
            self.xyz,self.labels=load_map(directory/'map_clean.npy')
            self.poses=load_poses(directory)
            self.tree=cKDTree(self.xyz,leafsize=64)
            self.sequence=sequence

    def load(self,sequence,frame,*,reference_full_scan=False):
        self._load(sequence)
        from generate_semantickitti_poisson_gt import _full_candidate
        pose=self.poses[frame]
        if reference_full_scan: xyz,labels=self.xyz,self.labels
        else:
            idx=self.tree.query_ball_point(pose[:3,3],self.args.max_range)
            # Reference preserves original map row ordering before sampling.
            idx=np.sort(np.asarray(idx,dtype=np.int64)); xyz,labels=self.xyz[idx],self.labels[idx]
        if self.fast_viewpoint:
            # Process-local, single-threaded adapter. Keep every other operation
            # in the original function, including map row order and float casts.
            from .viewpoint_lookup import viewpoint_mask_lookup
            import generate_semantickitti_poisson_gt as module
            original=module.viewpoint_mask
            try:
                module.viewpoint_mask=viewpoint_mask_lookup
                raw=_full_candidate(self.root/sequence/'velodyne'/f'{frame:06d}.bin',pose,xyz,labels,self.args)
            finally:module.viewpoint_mask=original
        else:raw=_full_candidate(self.root/sequence/'velodyne'/f'{frame:06d}.bin',pose,xyz,labels,self.args)
        return np.ascontiguousarray(raw[:,:3])
