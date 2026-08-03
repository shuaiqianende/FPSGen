import numpy as np
import open3d as o3d
from scipy.spatial.distance import jensenshannon
from fpsgen.utils.metrics import ChamferDistance, PrecisionRecall
import matplotlib.pyplot as plt
import torch

def histogram_point_cloud_torch(pcd, resolution, max_range, bev=False):
    """
    （ torch ）

    ：
        pcd: torch.Tensor， (N, 3)，
        resolution: float，
        max_range: float， [-max_range, max_range]
        bev: bool， True，（ 0  1）

    ：
        hist: torch.Tensor， (bins, bins, bins)
    """
    bins = int(2 * max_range / resolution)
    delta_bin = int(bins - 2 * max_range)
    indices = torch.floor((pcd + max_range) / resolution).long()

    valid_mask = (indices >= 0+delta_bin//2) & (indices < bins-delta_bin//2)
    valid_mask = valid_mask.all(dim=1)
    indices = indices[valid_mask]

    hist = torch.zeros((bins, bins, bins), dtype=torch.float32, device=pcd.device)

    if indices.shape[0] > 0:
        flat_indices = indices[:, 0] * (bins * bins) + indices[:, 1] * bins + indices[:, 2]
        flat_hist = hist.view(-1)
        ones = torch.ones(flat_indices.size(0), dtype=flat_hist.dtype, device=pcd.device)
        flat_hist.index_add_(0, flat_indices, ones)

    if bev:
        hist = (hist > 0).float()

    return hist


def compute_jsd_torch(P, Q, bev):
    """
     P  Q  Jensen-Shannon Divergence（JSD）， torch

    ：
        P, Q: torch.Tensor，
        bev: bool，，

    ：
        jsd: torch.Tensor，， JSD
    """
    eps = 1e-8
    P_norm = P / (P.sum() + eps)
    Q_norm = Q / (Q.sum() + eps)
    M = (P_norm + Q_norm) / 2.0
    jsd = 0.5 * (P_norm * (torch.log(P_norm + eps) - torch.log(M + eps))).sum() \
          + 0.5 * (Q_norm * (torch.log(Q_norm + eps) - torch.log(M + eps))).sum()
    return jsd


def compute_hist_metrics_torch(pcd_gt, pcd_pred, bev=False):
    """
     ground truth ，，
     Jensen-Shannon Divergence （ torch ）

    ：
        pcd_gt: torch.Tensor，ground truth ， (N, 3)
        pcd_pred: torch.Tensor，， (M, 3)
        bev: bool， True，（）

    ：
        JSD ， torch.Tensor （ .item() ）
    """
    hist_pred = histogram_point_cloud_torch(pcd_pred, resolution=0.5, max_range=50.0, bev=bev)
    hist_gt = histogram_point_cloud_torch(pcd_gt, resolution=0.5, max_range=50.0, bev=bev)

    return compute_jsd_torch(hist_gt, hist_pred, bev)

def histogram_point_cloud(pcd, resolution, max_range, bev=False):
    bins = int(2 * max_range / resolution)

    hist = np.histogramdd(pcd, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range]))

    return np.clip(hist[0], a_min=0., a_max=1.) if bev else hist[0]

def compute_jsd(hist_gt, hist_pred, bev=False, visualize=False):
    bev_gt = hist_gt.sum(-1) if bev else hist_gt
    norm_bev_gt = bev_gt / bev_gt.sum()
    norm_bev_gt = norm_bev_gt.flatten()

    bev_pred = hist_pred.sum(-1) if bev else hist_pred
    norm_bev_pred = bev_pred / bev_pred.sum()
    norm_bev_pred = norm_bev_pred.flatten()

    if visualize:
        grid = np.meshgrid(np.arange(len(hist_gt)), np.arange(len(hist_gt)))
        points = np.concatenate((grid[0].flatten()[:,None], grid[1].flatten()[:,None]), axis=-1)
        points = np.concatenate((points, np.zeros((len(points),1))),axis=-1)

        norm_hist_gt = bev_gt / bev_gt.max()
        colors_gt = plt.get_cmap('viridis')(norm_hist_gt)
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(points)
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt.reshape(-1,4)[:,:3])

        norm_hist_pred = bev_pred / bev_pred.max()
        colors_pred = plt.get_cmap('viridis')(norm_hist_pred)
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(points)
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred.reshape(-1,4)[:,:3])

    return jensenshannon(norm_bev_gt, norm_bev_pred)


def _as_xyz_array(point_cloud):
    """Return XYZ coordinates from either Open3D clouds or numeric arrays.

    The unified evaluator keeps ground truth and generated predictions as
    ``numpy.ndarray`` objects, while the legacy metric helpers accepted only
    Open3D ``PointCloud`` instances.  Normalizing both representations here
    keeps the metric implementation independent of the caller's container.
    """
    if hasattr(point_cloud, "points"):
        points = np.asarray(point_cloud.points)
    else:
        points = np.asarray(point_cloud)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"point cloud must have shape [N, >=3], got {points.shape}")
    return points[:, :3]


def compute_hist_metrics_multirange(pcd_gt, pcd_pred, bev=False, dist_bins=[(0, 20), (20, 35), (35, 50)]):
    """Compute range-wise JSD for Open3D point clouds or XYZ arrays."""
    points_gt = _as_xyz_array(pcd_gt)
    points_pred = _as_xyz_array(pcd_pred)
    dist_gt = np.linalg.norm(points_gt[:, :2], axis=1)
    dist_pred = np.linalg.norm(points_pred[:, :2], axis=1)

    range_results = {}

    for d_min, d_max in dist_bins:
        sub_gt = points_gt[(dist_gt >= d_min) & (dist_gt < d_max)]
        sub_pred = points_pred[(dist_pred >= d_min) & (dist_pred < d_max)]

        if len(sub_gt) == 0 or len(sub_pred) == 0:
            range_results[f"{d_min}_{d_max}m"] = 0.0
            continue

        hist_pred = histogram_point_cloud(sub_pred, 0.5, 50., bev)
        hist_gt = histogram_point_cloud(sub_gt, 0.5, 50., bev)

        jsd_val = compute_jsd(hist_gt, hist_pred, bev)

        range_results[f"{d_min}_{d_max}m"] = jsd_val

    return range_results

def compute_hist_metrics(pcd_gt, pcd_pred, bev=False):
    hist_pred = histogram_point_cloud(np.array(pcd_pred.points), 0.5, 50., bev)
    hist_gt = histogram_point_cloud(np.array(pcd_gt.points), 0.5, 50., bev)

    return compute_jsd(hist_gt, hist_pred, bev)

def compute_chamfer(pcd_pred, pcd_gt):
    chamfer_distance = ChamferDistance()
    chamfer_distance.update(pcd_gt, pcd_pred)
    cd_pred_mean, cd_pred_std = chamfer_distance.compute()

    return cd_pred_mean

def compute_precision_recall(pcd_pred, pcd_gt):
    precision_recall = PrecisionRecall(0.05,2*0.05,100)
    precision_recall.update(pcd_gt, pcd_pred)
    pr, re, f1 = precision_recall.compute_auc()

    return pr, re, f1

def preprocess_pcd(pcd):
    points = np.array(pcd.points)
    dist = np.sqrt(np.sum(points**2, axis=-1))
    pcd.points = o3d.utility.Vector3dVector(points[dist < 30.])

    return pcd

def compute_metrics(pred_path, gt_path):
    pcd_pred = preprocess_pcd(o3d.io.read_point_cloud(pred_path))
    points_pred = np.array(pcd_pred.points)
    pcd_gt = preprocess_pcd(o3d.io.read_point_cloud(gt_path))
    points_gt = np.array(pcd_gt.points)

    jsd_pred = compute_hist_metrics(points_pred, points_gt)

    cd_pred = compute_chamfer(pcd_pred, pcd_gt)

    pr_pred, re_pred, f1_pred = compute_precision_recall(pcd_pred, pcd_gt)

    return cd_pred, pr_pred, re_pred, f1_pred
