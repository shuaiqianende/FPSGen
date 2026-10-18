"""Read-only diagnostics: empty child slots are not proof of missing surface."""
import numpy as np


POPCOUNT8=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)


def mask_histogram(masks):
    values=np.asarray(masks)
    if values.ndim!=1 or values.dtype.kind not in 'iu' or np.any(values<1) or np.any(values>255):
        raise ValueError('Nonempty packed8-way GT masks required')
    return np.bincount(values.astype(np.int64),minlength=256)


def summarize_histogram(pattern_histogram):
    h=np.asarray(pattern_histogram)
    if h.shape!=(256,) or h.dtype.kind not in 'iu' or np.any(h<0) or h[0]!=0:
        raise ValueError('256 nonnegative pattern bins; GT parent zero mask forbidden')
    parents=int(h.sum())
    if parents==0:raise ValueError('No occupied parents')
    children=int(np.dot(h,POPCOUNT8.astype(np.int64)))
    child_hist=np.zeros(9,dtype=np.int64)
    np.add.at(child_hist,POPCOUNT8,h)
    return {'occupied_parents':parents,'occupied_fine_children':children,
        'occupied_children_per_parent_mean':children/parents,
        'occupied_children_per_parent_histogram':child_hist.tolist(),
        'fine_fill_fraction_inside_occupied_parents_nominal8':children/(8*parents),
        'all8_children_occupied_parent_fraction':int(h[255])/parents,
        'denominator_policy':'Nominal8 child slots per occupied parent; ROI-partial validity measured separately',
        'interpretation':'Surface-hit labels; unhit children are not proven free space or automatically missing surface'}
