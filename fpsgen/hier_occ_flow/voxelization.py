"""Sparse surface hits and bitmasks; never allocate a dense fine field."""
import numpy as np


def pack_children(occupancy, geometry):
    if occupancy.ndim != 2 or occupancy.shape[1] != geometry.k: raise ValueError("Expected [M,K]")
    return np.bitwise_or.reduce(occupancy.astype(np.uint32)<<np.arange(geometry.k,dtype=np.uint32),axis=1).astype(geometry.mask_dtype)


def unpack_children(masks, geometry):
    return ((np.asarray(masks,dtype=np.uint32)[:,None]>>np.arange(geometry.k,dtype=np.uint32))&1).astype(bool)


def voxelize_surface(points, geometry):
    fine,valid=geometry.voxelize_points(points)
    if not len(fine): raise ValueError("No valid surface points")
    fine=geometry.unflatten(np.unique(geometry.flatten(fine,fine=True)),fine=True)
    parent,child=geometry.split_fine(fine)
    ids,inverse=np.unique(geometry.flatten(parent),return_inverse=True)
    masks=np.zeros(len(ids),dtype=np.uint32)
    np.bitwise_or.at(masks,inverse,np.left_shift(np.uint32(1),child.astype(np.uint32)))
    return ids.astype('<u4'),masks.astype(geometry.mask_dtype),{
        'input_points':len(points),'valid_points':int(valid.sum()),'filtered_points':int((~valid).sum()),
        'active_parents':len(ids),'fine_occupied':len(fine)}


def lookup_masks(label_ids, label_masks, candidates):
    """Missing candidates are true all-empty labels, never omitted from loss."""
    candidates=np.asarray(candidates,dtype=np.int64)
    if len(label_ids)==0: return np.zeros(len(candidates),dtype=label_masks.dtype)
    idx=np.searchsorted(label_ids,candidates)
    match=(idx<len(label_ids)) & (label_ids[np.minimum(idx,len(label_ids)-1)]==candidates)
    out=np.zeros(len(candidates),dtype=label_masks.dtype); out[match]=label_masks[idx[match]]
    return out


def occupied_fine_coords(ids,masks,geometry):
    rows,child=np.nonzero(unpack_children(masks,geometry))
    return geometry.compose(geometry.unflatten(ids.astype(np.int64))[rows],child)
