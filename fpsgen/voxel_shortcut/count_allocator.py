"""Exact180k Stage2 bridge: density/counts/XYZ/IDs only; sphere-truncated anchors."""

from fpsgen.voxel_point.count_allocator import allocate_counts
from .anchor_sampler import sample_spherical_anchors
from .count_mapping import normalize_density


def allocate_anchors(density, geometry, *, generator, noise_std_ratio=.25):
    if density.shape != (1,1,*geometry.field_shape):
        raise ValueError("Export one full-resolution scene at a time")
    valid = geometry.field_mask(device=density.device)
    density = normalize_density(density,valid)
    counts = allocate_counts(geometry.to_xyz(density[0,0]).cpu(),geometry,
                             generator=generator,total_points=geometry.point_budget)
    anchors,ids = sample_spherical_anchors(geometry,counts,generator=generator,noise_std_ratio=noise_std_ratio)
    if anchors.shape != (180000,3) or int(counts.sum())!=180000 or not geometry.valid_points(anchors).all():
        raise ValueError("Stage2 interface cardinality/ROI violation")
    return {'counts':geometry.to_field(geometry.flat_to_dense(counts)),
            'anchors':anchors,'voxel_ids':ids,'voxel_centers':geometry.voxel_centers(ids)}
