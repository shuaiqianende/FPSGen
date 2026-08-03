import numpy as np
import MinkowskiEngine as ME
import torch
import open3d as o3d

def bev_resample(pcd, grid_size=256, target_pts=180000, noise_std_xy=1.0, noise_std_z=1.0):
    r"""Construct the training-time BEV-supported point source.

    This function implements the BEV source sampler
    :math:`\mathcal{R}(\bar{B}; N, \Sigma)`, where the ground-truth BEV prior
    :math:`\bar{B}=\Phi(\mathcal{P}^{gt})` is obtained directly from the
    complete training scene. Cells are sampled with replacement according to
    their empirical density, converted to zero-height metric anchors, and
    perturbed with independent Gaussian coordinate noise.

    Args:
        pcd: Batched point clouds with shape ``[B, N, 3]`` in metres.
        grid_size: Number of cells per BEV axis over the fixed ``[-50, 50]`` m
            range.
        target_pts: Number of points returned for every batch item.
        noise_std_xy: Standard deviation of the independent X/Y jitter.
        noise_std_z: Standard deviation of the Z jitter around the ground plane.

    Returns:
        ``(anchors, source, independent_source)``, each with shape
        ``[B, target_pts, 3]``. ``anchors`` contains the sampled zero-height
        cell centers. The remaining tensors are independent realizations from
        the same conditional Gaussian mixture and are used by the coupling
        construction in the point-flow stage.

    Important:
        FPSGen stores BEV cells as ``flat_index = x_index * G + y_index``.
        Thus tensor dimension 2 is X and dimension 3 is Y, unlike the usual
        image convention ``[row=y, column=x]``. All BEV rasterization and
        reconstruction code must preserve this convention.
    """
    if pcd.ndim != 3 or pcd.shape[-1] != 3:
        raise ValueError(f"pcd must have shape [B, N, 3], got {tuple(pcd.shape)}")
    if pcd.shape[1] == 0:
        raise ValueError("pcd must contain at least one point per batch item")
    if grid_size < 1 or target_pts < 1:
        raise ValueError("grid_size and target_pts must be positive")
    if noise_std_xy < 0 or noise_std_z < 0:
        raise ValueError("noise standard deviations must be non-negative")

    b, _, _ = pcd.shape
    device = pcd.device

    xy = pcd[:, :, :2]

    # Values outside the modelled range are assigned to the nearest boundary
    # cell. The training dataset is range-filtered, so clamping mainly protects
    # against small numerical overshoots at exactly +/-50 m.
    xy_norm = (xy + 50.0) / 100.0

    xy_pixels = (xy_norm * grid_size).long()
    xy_pixels = xy_pixels.clamp(0, grid_size - 1)

    # Axis 0 is X and axis 1 is Y throughout FPSGen's BEV representation.
    flat_indices = xy_pixels[:, :, 0] * grid_size + xy_pixels[:, :, 1]

    img_flat = torch.zeros((b, grid_size * grid_size), dtype=torch.float32, device=device)
    ones = torch.ones_like(flat_indices, dtype=torch.float32)
    img_flat.scatter_add_(1, flat_indices, ones)

    # epsilon_w is the numerical stabilizer in the density-weighted categorical
    # distribution w_D(q). It also guarantees a valid distribution in finite
    # precision.
    weights = img_flat + 1e-8
    sampled_indices = torch.multinomial(weights, num_samples=target_pts, replacement=True)

    sampled_x_idx = (sampled_indices // grid_size).float()
    sampled_y_idx = (sampled_indices % grid_size).float()

    # Map discrete cells to metric cell centers, as defined by the BEV source
    # sampler. The half-cell offset is 0.1953125 m for a 256 x 256 grid.
    sampled_x = ((sampled_x_idx + 0.5) / grid_size) * 100.0 - 50.0
    sampled_y = ((sampled_y_idx + 0.5) / grid_size) * 100.0 - 50.0

    # The BEV anchor is defined at zero height. Vertical structure is introduced
    # by the Gaussian perturbation and recovered by the learned point transport.
    sampled_z = torch.zeros_like(sampled_x)
    new_pcd = torch.stack([sampled_x, sampled_y, sampled_z], dim=-1)

    std_tensor = torch.tensor([noise_std_xy, noise_std_xy, noise_std_z], dtype=torch.float32, device=new_pcd.device)
    noise_ot = torch.randn_like(new_pcd) * std_tensor
    noise_rand = torch.randn_like(new_pcd) * std_tensor
    noisy_pcd_ot = new_pcd + noise_ot
    noisy_pcd_rand = new_pcd + noise_rand
    return new_pcd, noisy_pcd_ot, noisy_pcd_rand

def feats_to_coord(p_feats, resolution, mean, std):
    """Quantize metric XYZ features into MinkowskiEngine voxel coordinates.

    ``mean`` and ``std`` are accepted for compatibility with older normalized
    LiDiff call sites. FPSGen operates in metric coordinates, so they only
    provide the batch size needed to restore ``[B, N, 3]``.
    """
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def normalize_pcd(points, mean, std):
    """Apply per-scene affine point-cloud normalization."""
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    """Invert :func:`normalize_pcd` for batched or unbatched statistics."""
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse_refine(p_full, p_part, n_full, n_part, resolution, filename):
    """Legacy refinement preprocessor retained for compatibility.

    It repeats and randomly permutes full/partial point sets to fixed
    cardinalities, then returns statistics computed from the full scene.
    ``resolution`` is retained in the public signature for older call sites.
    """
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])

    p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    """Legacy viewpoint-aware fixed-cardinality point-set preprocessor.

    The partial observation defines a coarse 10 m viewpoint grid. Full-scene
    points outside those occupied view cells are removed before random
    fixed-cardinality sampling.
    """
    concat_part = np.ceil(n_part / p_part.shape[0])
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)

    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    p_part = torch.tensor(np.array(pcd_part.points))

    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint]
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)

    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    """Convert coordinate/feature lists to a MinkowskiEngine sparse tensor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()

        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseSegmentCollation:
    """Stack temporal point-cloud samples into the FPSGen batch contract."""
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        """Preserve metadata while batching full and partial clouds along dimension zero."""
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'pcd_part': torch.stack(batch[1]).float(),
            'filename': batch[2],
            'pos_tran': torch.stack(batch[3]).float(),
            }
