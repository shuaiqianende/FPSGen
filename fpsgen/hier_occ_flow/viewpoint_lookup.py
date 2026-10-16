"""Exact integer-cell membership with a small lookup table, no map-row sorting."""
import numpy as np


def viewpoint_mask_lookup(scan_xyz,map_local_xyz,voxel_size):
    scan=np.floor(scan_xyz/voxel_size).astype(np.int64)
    points=np.floor(map_local_xyz/voxel_size).astype(np.int64)
    if not len(scan) or not len(points):return np.zeros(len(points),dtype=bool)
    lower=np.minimum(scan.min(0),points.min(0));upper=np.maximum(scan.max(0),points.max(0));shape=upper-lower+1
    if np.prod(shape,dtype=object)>1000000:raise ValueError('Unexpected unbounded viewpoint lattice')
    def flatten(coord):
        q=coord-lower;return q[:,0]+shape[0]*(q[:,1]+shape[1]*q[:,2])
    occupied=np.zeros(int(np.prod(shape)),dtype=bool);occupied[flatten(scan)]=True
    return occupied[flatten(points)]
