import numpy as np
import MinkowskiEngine as ME
import torch
import open3d as o3d

def feats_to_coord(p_feats, resolution, mean, std):
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def normalize_pcd(points, mean, std):
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse_refine(p_full, p_part, n_full, n_part, resolution, filename):
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])


    p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    print('p_part', p_part.shape)
    concat_part = np.ceil(n_part / p_part.shape[0])
    p_part = p_part.repeat(concat_part, 0)
    print('p_part2', p_part.shape)
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    print('pcd_part', pcd_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)


    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    print('pcd_part2', pcd_part)
    p_part = torch.tensor(np.array(pcd_part.points))

    print('p_full', p_full.shape)
    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint]
    concat_full = np.ceil(n_full / p_full.shape[0])

    print('p_full2', p_full.shape)
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]
    print('p_full3', p_full.shape)

    p_full = torch.tensor(p_full)

    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
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
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'pcd_part': torch.stack(batch[1]).float(),
            'filename': batch[2],
            'pos_tran': torch.stack(batch[3]).float(),
            }


class SparseSegmentCollationGen:
    """Collate fixed-size generation samples without sparse quantization.

    Sparse ``TensorField`` objects are created inside each model because they
    must live on the model's CUDA device. The DataLoader therefore only stacks
    dense point arrays and their point-aligned labels here.
    """
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        """Return the batch dictionary shared by all three training stages.

        ``pcd_full`` and ``full_label`` must have the same point ordering;
        likewise for ``pcd_part`` and ``part_label``. Fixed preprocessing
        cardinality is required because ``torch.stack`` does not pad samples.
        """
        if not data:
            raise ValueError("Cannot collate an empty batch")
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'pcd_part': torch.stack(batch[1]).float(),
            'filename': batch[2],
            'pos_tran': torch.stack(batch[3]).float(),
            'part_label': torch.stack(batch[4]).float(),
            'full_label': torch.stack(batch[5]).float(),
            }
