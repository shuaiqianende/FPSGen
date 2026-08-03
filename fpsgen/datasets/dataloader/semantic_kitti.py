import torch
from torch.utils.data import Dataset
from fpsgen.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
from fpsgen.utils.pcd_transforms import *
from fpsgen.utils.data_map import learning_map
from natsort import natsorted
import os
import numpy as np
import yaml
import open3d as o3d
from PIL import Image

import warnings

warnings.filterwarnings('ignore')

def point_set_to_sparse(p_full, p_part, filename, pos_tran, part_label, full_label):
    p_full = torch.tensor(p_full)
    pos_tran = torch.tensor(pos_tran)

    return [p_full, p_part, filename, pos_tran, part_label, full_label]

class TemporalKITTISet(Dataset):
    """Load SemanticKITTI scans, labels, and temporal partial-point conditions."""
    def __init__(self, data_dir, seqs, split, resolution, num_points, max_range, dataset_norm=False, std_axis_norm=False, HW=[64, 1024]):
        super().__init__()
        self.data_dir = data_dir

        self.n_clusters = 50
        self.resolution = resolution
        self.num_points = num_points
        self.max_range = max_range

        self.HW = HW
        self.split = split
        self.seqs = seqs
        self.cache_maps = {}

        self.datapath_list()
        self.data_stats = {'mean': None, 'std': None}

        if os.path.isfile(f'utils/data_stats_range_{int(self.max_range)}m.yml') and dataset_norm:
            stats = yaml.safe_load(open(f'utils/data_stats_range_{int(self.max_range)}m.yml'))
            data_mean = np.array([stats['mean_axis']['x'], stats['mean_axis']['y'], stats['mean_axis']['z']])
            if std_axis_norm:
                data_std = np.array([stats['std_axis']['x'], stats['std_axis']['y'], stats['std_axis']['z']])
            else:
                data_std = np.array([stats['std'], stats['std'], stats['std']])
            self.data_stats = {
                'mean': torch.tensor(data_mean),
                'std': torch.tensor(data_std)
            }

        self.nr_data = len(self.points_datapath)
        self.cnt = 0
        print('The size of %s data is %d'%(self.split,len(self.points_datapath)))

    def transforms(self, points):
        """Apply the dataset's geometry preprocessing to one point cloud."""
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    def datapath_list(self):
        """Enumerate frames and keep every frame aligned with its own pose.

        Pose lookup uses the numeric frame stem instead of a global concatenated
        index. This remains correct when a sequence has missing preprocessed
        frames or when only a subset of sequences is selected.
        """
        self.points_datapath = []
        self.point_poses = []
        self.seq_poses = {}
        data_dir = self.data_dir
        if not data_dir:
            raise ValueError(
                "data.data_dir is empty; set the TRAIN_DATABASE environment variable"
            )

        for seq in self.seqs:
            seq_path = os.path.join(data_dir, seq)
            if not os.path.isdir(seq_path):
                raise FileNotFoundError(f"Sequence directory does not exist: {seq_path}")
            poses = load_poses(
                os.path.join(seq_path, 'calib.txt'),
                os.path.join(seq_path, 'poses.txt'),
            )
            self.seq_poses[seq] = poses

            point_seq_path = os.path.join(self.data_dir, seq, 'gt_')
            if not os.path.isdir(point_seq_path):
                raise FileNotFoundError(
                    f"Preprocessed ground-truth directory does not exist: {point_seq_path}"
                )
            point_seq_gt = [f for f in os.listdir(point_seq_path) if f.endswith('.npy')]
            point_seq_gt = natsorted(point_seq_gt)

            for file_name in point_seq_gt:
                try:
                    frame_index = int(os.path.splitext(file_name)[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Expected a numeric frame name, got {file_name!r}"
                    ) from exc
                if frame_index >= len(poses):
                    raise IndexError(
                        f"Frame {file_name} has no matching pose in sequence {seq}"
                    )
                self.points_datapath.append(os.path.join(point_seq_path, file_name))
                self.point_poses.append(poses[frame_index])

        if not self.points_datapath:
            raise FileNotFoundError(
                f"No .npy ground-truth frames found for sequences {list(self.seqs)}"
            )

    def cart2sphere_proj(self,
                         xyz,
                         H: int = 96,
                         W: int = 2048,
                         fov_up: float = 8.0,
                         fov_down: float = -30.0):
        """
        xyz: [N, 3]  torch.Tensor  (x, y, z)
        :
            proj_img: [H, W, 4]  torch.float32
                      channel 0:  r
                      channel 1~3: xyz
        """
        xyz = xyz.float()
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        theta = (torch.atan2(y, x) + torch.pi) * 180.0 / torch.pi
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2).clamp(min=1e-8)
        beta = torch.asin(z / r) * 180.0 / torch.pi

        fov = abs(fov_up) + abs(fov_down)
        u = (theta / 360.0 * W).round().long() % W
        v = (1.0 - (beta - fov_down) / fov) * H
        v = torch.clamp(v.long(), 0, H - 1)

        order = torch.argsort(r, descending=True)
        u_sorted = u[order]
        v_sorted = v[order]
        r_sorted = r[order].float()
        xyz_sorted = xyz[order]

        proj_img = torch.zeros((H, W, 4), dtype=torch.float32, device=xyz.device)
        proj_img[v_sorted, u_sorted, 0] = r_sorted
        proj_img[v_sorted, u_sorted, 1:4] = xyz_sorted

        return proj_img

    def save_color_depth(self, depth_map, save_path):
        """（）"""
        valid_depth = depth_map[depth_map > 0]
        if len(valid_depth) == 0:
            print("No valid depth values!")
            return
        depth_log = np.log1p(depth_map)
        depth_normalized = ((depth_log - depth_log.min()) /
                            (depth_log.max() - depth_log.min() + 1e-8) * 255).astype(np.uint8)
        import matplotlib.pyplot as plt
        colored = plt.cm.viridis(depth_normalized / 255.0)
        colored = (colored[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(colored).save(save_path)

    def cart2sphere_proj_np(self,
                            xyz: np.ndarray,
                            H: int = 64,
                            W: int = 1024,
                            fov_up: float = 3.0,
                            fov_down: float = -25.0):
        """
        xyz: [N, 3]  NumPy float32  (x, y, z)
        :
            proj_img: [H, W, 4]  NumPy float32
                      channel 0:  r
                      channel 1~3: xyz
        """
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        theta = (np.arctan2(y, x) + np.pi).astype(np.float32) * 180.0 / np.pi
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2).astype(np.float32).clip(min=1e-8)
        beta = np.arcsin(z / r).astype(np.float32) * 180.0 / np.pi

        fov = abs(fov_up) + abs(fov_down)
        u_float = theta / 360.0 * W
        u = np.floor(u_float).astype(np.int64) % W
        v_float = (1.0 - (beta - fov_down) / fov) * H
        v = np.floor(v_float).astype(np.int64)
        v = np.clip(v, 0, H - 1)

        order = np.argsort(r)[::-1]
        u_sorted = u[order]
        v_sorted = v[order]
        r_sorted = r[order]
        xyz_sorted = xyz[order]

        proj_img = np.zeros((H, W, 4), dtype=np.float32)
        proj_img[v_sorted, u_sorted, 0] = r_sorted
        proj_img[v_sorted, u_sorted, 1:4] = xyz_sorted

        depth = proj_img[:, :, 0]
        return proj_img

    def __getitem__(self, index):
        """Return one aligned full/partial cloud, labels, pose, and source path.

        Both clouds receive the same random flip so semantic labels remain
        aligned point-for-point. Preprocessed arrays must store XYZ followed by
        at least one label column.
        """
        full_path = self.points_datapath[index]
        part_path = full_path.replace('gt_', 'input_')
        if not os.path.isfile(part_path):
            raise FileNotFoundError(f"Partial point cloud does not exist: {part_path}")
        p_part = np.load(part_path)
        # Test-mode loss is evaluated against the full ground truth as well.
        # Generation-only inference uses ``fpsgen.inference`` and does not
        # route unlabeled inputs through this supervised Dataset.
        p_full = np.load(full_path)
        if p_part.ndim != 2 or p_full.ndim != 2:
            raise ValueError(
                f"Point arrays must be two-dimensional: {part_path}, {full_path}"
            )
        if p_part.shape[1] < 4 or p_full.shape[1] < 4:
            raise ValueError(
                "Preprocessed point arrays must contain XYZ plus a label column"
            )
        if len(p_part) == 0 or len(p_full) == 0:
            raise ValueError(f"Point arrays must not be empty: {part_path}, {full_path}")
        part_label = p_part[:, 3:]
        full_label = p_full[:, 3:]
        p_part = p_part[:, :3]
        p_full = p_full[:, :3]
        p_part = p_part.reshape((-1,3))
        p_full = p_full.reshape((-1,3))
        if self.split == 'train':
            p_concat = np.concatenate((p_full, p_part), axis=0)
            p_concat = self.transforms(p_concat)

            p_full = p_concat[:-len(p_part)]
            p_part = p_concat[-len(p_part):]

        n_part = int(self.num_points / 10.)
        p_part = torch.tensor(p_part)
        part_label = torch.tensor(part_label)
        full_label = torch.tensor(full_label)

        return point_set_to_sparse(
            p_full,
            p_part,
            full_path,
            self.point_poses[index],
            part_label,
            full_label,
        )

    def __len__(self):
        return self.nr_data
