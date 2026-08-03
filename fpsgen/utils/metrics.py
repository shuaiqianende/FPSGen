import open3d as o3d
import numpy as np
import scipy
import torch
import torchist
import itertools
import torch.nn.functional as F
import scipy.integrate
from typing import Tuple
from torch import Tensor
from tqdm import tqdm
from chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()


def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
    dist1, dist2, idx1, idx2 = chamfer_dist(gt, output)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = dist1.mean(1) + dist2.mean(1)

    if separate:
        res = [
            torch.cat([
                torch.sqrt(dist1).mean(1).unsqueeze(0),
                torch.sqrt(dist2).mean(1).unsqueeze(0)
            ]),
            torch.cat([
                dist1.mean(1).unsqueeze(0),
                dist2.mean(1).unsqueeze(0)
            ])
        ]
    else:
        res = [cd_p, cd_t]

    if calc_f1:
        raise NotImplementedError("calc_f1 is not supported in this calc_cd helper.")
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])
    return res


def calc_dcd(x, gt, alpha=10, n_lambda=1, return_raw=False, non_reg=False):
    x = x.float()
    gt = gt.float()
    batch_size, n_x, _ = x.shape
    _, n_gt, _ = gt.shape
    assert batch_size == gt.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
    exp_dist1 = torch.exp(-dist1 * alpha)
    exp_dist2 = torch.exp(-dist2 * alpha)

    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(1, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=1)

    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(1, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=1)

    loss = (loss1 + loss2) / 2

    res = [loss, cd_p, cd_t]
    if return_raw:
        res.extend([dist1, dist2, idx1, idx2])

    return res

def _chamfer_sqrt(p1, p2):
    p1 = p1.unsqueeze(0)
    p2 = p2.unsqueeze(0)
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def _chamfer(p1, p2):
    p1 = p1.unsqueeze(0)
    p2 = p2.unsqueeze(0)
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)

MESHTYPE = 6
TETRATYPE = 10
PCDTYPE = 1


@torch.no_grad()
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    import pykeops
    pykeops.set_verbose(False)
    xi = pykeops.torch.LazyTensor(q_points.unsqueeze(-2))
    xj = pykeops.torch.LazyTensor(s_points.unsqueeze(-3))
    dij = (xi - xj).sqnorm2()
    knn_d2, knn_indices = dij.Kmin_argKmin(k, dim=q_points.dim() - 1)
    return knn_d2, knn_indices


class EMD_MultiRange:
    def __init__(self, voxel_size=0.5, dist_bins=[(0, 20), (20, 35), (35, 50)], epsilon=0.01, max_iter=500, tol=1e-3):
        self.voxel_size = voxel_size
        self.dist_bins = dist_bins if dist_bins is not None else []
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.reset()

    def reset(self):
        self.emd_records = {f"{d[0]}-{d[1]}m": [] for d in self.dist_bins}

    def voxelize_tensor(self, points):
        if len(points) == 0:
            return torch.empty(0, 3, device=self.device), torch.empty(0, device=self.device)
        voxel_indices = torch.floor(points / self.voxel_size).long()
        unique_voxels, voxel_counts = torch.unique(voxel_indices, dim=0, return_counts=True)
        weights = voxel_counts.float() / voxel_counts.sum()
        voxel_centers = (unique_voxels.float() + 0.5) * self.voxel_size
        return voxel_centers, weights

    def sinkhorn_knopp_pt(self, src_pts, tgt_pts, src_w, tgt_w):
        import pykeops
        from pykeops.torch import LazyTensor
        pykeops.set_verbose(False)

        if len(src_pts) == 0 or len(tgt_pts) == 0:
            return float('nan')

        src_pts, tgt_pts = src_pts.contiguous(), tgt_pts.contiguous()
        N, M = len(src_pts), len(tgt_pts)

        x_i = LazyTensor(src_pts.view(N, 1, 3))
        y_j = LazyTensor(tgt_pts.view(1, M, 3))

        D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()
        max_cost = D_ij.max(dim=1).max().item()
        if max_cost == 0: return 0.0

        C_ij = D_ij / max_cost
        K_ij = (-C_ij / self.epsilon).exp()

        u = torch.ones((N, 1), dtype=src_pts.dtype, device=src_pts.device)
        v = torch.ones((M, 1), dtype=tgt_pts.dtype, device=tgt_pts.device)
        src_w_col, tgt_w_col = src_w.view(N, 1), tgt_w.view(M, 1)

        for m in range(self.max_iter):
            u_prev = u.clone()
            v_j = LazyTensor(v.view(1, M, 1))
            u = src_w_col / ((K_ij * v_j).sum(dim=1) + 1e-9)
            u_i = LazyTensor(u.view(N, 1, 1))
            v = tgt_w_col / ((K_ij * u_i).sum(dim=0) + 1e-9)

            if torch.norm(u - u_prev, p=1) < self.tol:
                break

        u_i, v_j = LazyTensor(u.view(N, 1, 1)), LazyTensor(v.view(1, M, 1))
        emd_approx = (u_i * v_j * K_ij * C_ij).sum(dim=1).sum() * max_cost
        return emd_approx.item()

    def update(self, gt_pcd, pt_pcd):
        if not self.dist_bins:
            return

        with torch.no_grad():
            gt_pts = torch.from_numpy(np.asarray(gt_pcd.points)).float().to(self.device)
            pt_pts = torch.from_numpy(np.asarray(pt_pcd.points)).float().to(self.device)

            rad_gt = torch.norm(gt_pts[:, :2], dim=1) if len(gt_pts) > 0 else torch.empty(0, device=self.device)
            rad_pt = torch.norm(pt_pts[:, :2], dim=1) if len(pt_pts) > 0 else torch.empty(0, device=self.device)

            for d_min, d_max in self.dist_bins:
                bin_name = f"{d_min}-{d_max}m"
                sub_gt = gt_pts[(rad_gt >= d_min) & (rad_gt < d_max)]
                sub_pt = pt_pts[(rad_pt >= d_min) & (rad_pt < d_max)]

                v_gt_c, v_gt_w = self.voxelize_tensor(sub_gt)
                v_pt_c, v_pt_w = self.voxelize_tensor(sub_pt)

                emd_val = self.sinkhorn_knopp_pt(v_pt_c, v_gt_c, v_pt_w, v_gt_w)
                if not np.isnan(emd_val):
                    self.emd_records[bin_name].append(emd_val)

    def compute(self):
        if not self.dist_bins:
            return {}, float('nan')

        res = {}
        all_values = []

        for bin_name, emd_list in self.emd_records.items():
            if emd_list:
                avg = sum(emd_list) / len(emd_list)
                res[bin_name] = avg
                all_values.extend(emd_list)
            else:
                res[bin_name] = float('nan')

        overall_mean = sum(all_values) / len(all_values) if all_values else float('nan')
        return res, overall_mean

class PrecisionRecall_MultiRange:
    def __init__(self, thresholds=[0.1, 0.2, 0.5], dist_bins=[(0, 20), (20, 35), (35, 50)]):
        self.thresholds = np.sort(np.array(thresholds))
        self.dist_bins = dist_bins
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.reset()

    def reset(self):
        self.stats = {}
        for d in self.dist_bins:
            bin_name = f"{d[0]}-{d[1]}m"
            self.stats[bin_name] = {
                'pr': {t: [] for t in self.thresholds},
                're': {t: [] for t in self.thresholds},
                'f1': {t: [] for t in self.thresholds}
            }

    def update(self, gt_pcd, pt_pcd):
        gt_pts = torch.from_numpy(np.asarray(gt_pcd.points)).float().contiguous().to(self.device)
        pt_pts = torch.from_numpy(np.asarray(pt_pcd.points)).float().contiguous().to(self.device)

        rad_gt = torch.norm(gt_pts[:, :2], dim=1) if len(gt_pts) > 0 else torch.empty(0, device=self.device)
        rad_pt = torch.norm(pt_pts[:, :2], dim=1) if len(pt_pts) > 0 else torch.empty(0, device=self.device)

        if len(pt_pts) == 0 or len(gt_pts) == 0:
            dist_pt_2_gt = torch.empty(0, device=self.device)
            dist_gt_2_pt = torch.empty(0, device=self.device)
        else:
            pt2gt_d2, _ = keops_knn(pt_pts, gt_pts, k=1)
            gt2pt_d2, _ = keops_knn(gt_pts, pt_pts, k=1)
            dist_pt_2_gt = torch.sqrt(pt2gt_d2.squeeze(-1))
            dist_gt_2_pt = torch.sqrt(gt2pt_d2.squeeze(-1))

        for d_min, d_max in self.dist_bins:
            bin_name = f"{d_min}-{d_max}m"

            mask_pt = (rad_pt >= d_min) & (rad_pt < d_max)
            mask_gt = (rad_gt >= d_min) & (rad_gt < d_max)

            sub_dist_pt_2_gt = dist_pt_2_gt[mask_pt] if len(dist_pt_2_gt) > 0 else torch.empty(0)
            sub_dist_gt_2_pt = dist_gt_2_pt[mask_gt] if len(dist_gt_2_pt) > 0 else torch.empty(0)

            len_pt = len(sub_dist_pt_2_gt)
            len_gt = len(sub_dist_gt_2_pt)

            for t in self.thresholds:
                if len_pt == 0:
                    p = 0.0
                else:
                    p_count = (sub_dist_pt_2_gt < t).sum().item()
                    p = 100.0 * p_count / len_pt

                if len_gt == 0:
                    r = 0.0
                else:
                    r_count = (sub_dist_gt_2_pt < t).sum().item()
                    r = 100.0 * r_count / len_gt

                f = 0.0 if (p == 0 or r == 0) else 2 * p * r / (p + r)

                self.stats[bin_name]['pr'][t].append(p)
                self.stats[bin_name]['re'][t].append(r)
                self.stats[bin_name]['f1'][t].append(f)

    def compute_at_threshold(self, threshold):
        t = self.find_nearest_threshold(threshold)
        res = {}
        for bin_name, metrics in self.stats.items():
            pr = sum(metrics['pr'][t]) / len(metrics['pr'][t]) if metrics['pr'][t] else 0.0
            re = sum(metrics['re'][t]) / len(metrics['re'][t]) if metrics['re'][t] else 0.0
            f1 = sum(metrics['f1'][t]) / len(metrics['f1'][t]) if metrics['f1'][t] else 0.0
            res[bin_name] = (pr, re, f1)
        return res, t

    def compute_at_all_thresholds(self):
        res = {}
        for bin_name, metrics in self.stats.items():
            pr = [sum(metrics['pr'][t]) / len(metrics['pr'][t]) if metrics['pr'][t] else 0.0 for t in self.thresholds]
            re = [sum(metrics['re'][t]) / len(metrics['re'][t]) if metrics['re'][t] else 0.0 for t in self.thresholds]
            f1 = [sum(metrics['f1'][t]) / len(metrics['f1'][t]) if metrics['f1'][t] else 0.0 for t in self.thresholds]
            res[bin_name] = (pr, re, f1)
        return res

    def compute_auc(self):
        perfect_predictor = scipy.integrate.simpson(y=np.ones_like(self.thresholds), x=self.thresholds)

        all_threshold_res = self.compute_at_all_thresholds()
        auc_res = {}

        for bin_name, (pr, re, f1) in all_threshold_res.items():
            pr_area = scipy.integrate.simpson(y=pr, x=self.thresholds)
            norm_pr_area = pr_area / perfect_predictor

            re_area = scipy.integrate.simpson(y=re, x=self.thresholds)
            norm_re_area = re_area / perfect_predictor

            f1_area = scipy.integrate.simpson(y=f1, x=self.thresholds)
            norm_f1_area = f1_area / perfect_predictor

            auc_res[bin_name] = (norm_pr_area, norm_re_area, norm_f1_area)

        return auc_res

    def find_nearest_threshold(self, value):
        idx = (np.abs(self.thresholds - value)).argmin()
        return self.thresholds[idx]

class Metrics3D():
    def prediction_is_empty(self, geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                empty_v = self.is_empty(len(geom.vertices))
                empty_t = self.is_empty(len(geom.triangles))
                empty = empty_t or empty_v
            elif geom_type == PCDTYPE:
                empty = self.is_empty(len(geom.points))
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            empty = self.is_empty(len(geom[:, :3]))
        elif isinstance(geom, torch.Tensor):
            empty = self.is_empty(len(geom[:, :3]))
        else:
            assert False, '{} type not supported'.format(type(geom))

        return empty

    @staticmethod
    def convert_to_pcd(geom):

        if isinstance(geom, o3d.geometry.Geometry):
            geom_type = geom.get_geometry_type().value
            if geom_type == MESHTYPE or geom_type == TETRATYPE:
                geom_pcd = geom.sample_points_uniformly(1000000)
            elif geom_type == PCDTYPE:
                geom_pcd = geom
            else:
                assert False, '{} geometry not supported'.format(geom.get_geometry_type())
        elif isinstance(geom, np.ndarray):
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        elif isinstance(geom, torch.Tensor):
            geom = geom.detach().cpu().numpy()
            geom_pcd = o3d.geometry.PointCloud()
            geom_pcd.points = o3d.utility.Vector3dVector(geom[:, :3])
        else:
            assert False, '{} type not supported'.format(type(geom))

        return geom_pcd

    @staticmethod
    def is_empty(length):
        empty = True
        if length:
            empty = False
        return empty

        input()

class RMSE():
    def __init__(self):
        self.dists = []

        return

    def update(self, gt_pcd, pt_pcd):
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))

        self.dists.append(np.mean(dist_pt_2_gt))

    def reset(self):
        self.dists = []

    def compute(self):
        dist = np.array(self.dists)
        return dist.mean(), dist.std()

class RMSETorch():
    def __init__(self):
        self.dists = []

    def update(self, gt_pcd, pt_pcd):
        print('pt_pcd', pt_pcd.shape)
        print('gt_pcd', gt_pcd.shape)
        mean_dist = _chamfer_sqrt(gt_pcd, pt_pcd)
        self.dists.append(mean_dist)

    def reset(self):
        self.dists = []

    def compute(self):
        dists_tensor = torch.stack(self.dists)
        return dists_tensor.mean().item(), dists_tensor.std().item()

class CompletionIoUTorch:
    def __init__(self, voxel_sizes=[0.5, 0.2, 0.1], device=None):
        """
        voxel_sizes:
        device: ， CPU
        """
        self.voxel_sizes = voxel_sizes
        self.device = device if device is not None else torch.device('cpu')
        self.conf_matrix = torch.zeros((len(self.voxel_sizes), 3), dtype=torch.int64, device=self.device)

    def _compute_histogram(self, vox_coords, bins, max_range, vsize):
        mask = (
                (vox_coords[:, 0] >= -max_range / vsize) & (vox_coords[:, 0] < max_range / vsize) &
                (vox_coords[:, 1] >= -max_range / vsize) & (vox_coords[:, 1] < max_range / vsize) &
                (vox_coords[:, 2] >= -max_range / vsize) & (vox_coords[:, 2] < max_range / vsize)
        )
        valid = vox_coords[mask]
        if valid.shape[0] == 0:
            return torch.zeros((bins, bins, bins), dtype=torch.int32, device=self.device)
        valid = valid + int(max_range / vsize)
        lin_indices = valid[:, 0] * (bins * bins) + valid[:, 1] * bins + valid[:, 2]
        hist = torch.zeros(bins * bins * bins, dtype=torch.int32, device=self.device)
        ones = torch.ones_like(lin_indices, dtype=torch.int32, device=self.device)
        hist.scatter_add_(0, lin_indices, ones)
        hist = hist.reshape(bins, bins, bins)
        occupancy = (hist > 0).int()
        print('occupancy', occupancy[occupancy!=0])
        return occupancy

    def update(self, gt, pred):
        """
        （tp, fn, fp）
        :param gt:  ground truth ， points
        :param pred: ， points
        """
        max_range = 50.0

        for i, vsize in enumerate(self.voxel_sizes):
            bins = int(2 * max_range / vsize)
            gt_h = torch.round(gt / vsize).to(torch.int32)
            pred_h = torch.round(pred / vsize).to(torch.int32)

            hist_gt = torchist.histogramdd(gt_h, bins=bins, low=-max_range, upp=max_range)
            hist_pred = torchist.histogramdd(pred_h, bins=bins, low=-max_range, upp=max_range)

            hist_gt = (hist_gt > 0).int()
            hist_pred = (hist_pred > 0).int()

            tp = torch.sum((hist_gt == 1) & (hist_pred == 1))
            fn = torch.sum((hist_gt == 1) & (hist_pred == 0))
            fp = torch.sum((hist_gt == 0) & (hist_pred == 1))
            tn = torch.sum((hist_gt == 0) & (hist_pred == 0))

            gt_sum = torch.sum((hist_gt == 1))
            pre_sum = torch.sum((hist_pred == 1))

            self.conf_matrix[i, 0] += tp.to(torch.int64)
            self.conf_matrix[i, 1] += fn.to(torch.int64)
            self.conf_matrix[i, 2] += fp.to(torch.int64)

    def compute(self):
        """
         IoU ()
        :return: ，， IoU
        """
        res_vsizes = {}
        for i, vsize in enumerate(self.voxel_sizes):
            tp = self.conf_matrix[i, 0].item()
            fn = self.conf_matrix[i, 1].item()
            fp = self.conf_matrix[i, 2].item()
            union = tp + fn + fp + 1e-15
            res_vsizes[vsize] = tp / union
        return res_vsizes

    def reset(self):
        """"""
        self.conf_matrix = torch.zeros((len(self.voxel_sizes), 3), dtype=torch.int64, device=self.device)


class CompletionIoU_MultiRange_Morphology():
    def __init__(self,
                 voxel_sizes=[0.5, 0.2, 0.1],
                 dist_bins=[(0, 20), (20, 35), (35, 50)],

                 kernel_radii=[1, 4, 8],
                 close_gt=False):

        assert len(voxel_sizes) == len(kernel_radii), "voxel_sizes  kernel_radii ！"

        self.voxel_sizes = voxel_sizes
        self.dist_bins = dist_bins
        self.kernel_radii = kernel_radii
        self.close_gt = close_gt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conf_matrix = torch.zeros((len(voxel_sizes), len(dist_bins), 3),
                                       dtype=torch.int64, device=self.device)

    def encode(self, v):
        """ 3D  1D ID """
        return ((v[:, 0] + 2000) << 40) | ((v[:, 1] + 2000) << 20) | (v[:, 2] + 2000)

    def morphology_close(self, v_coords, radius):
        if len(v_coords) == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        if radius == 0:
            return torch.unique(self.encode(v_coords))

        vmin = v_coords.min(dim=0)[0]
        vmax = v_coords.max(dim=0)[0]
        shape = (vmax - vmin + 1).tolist()
        local_coords = v_coords - vmin

        grid = torch.zeros((1, 1, *shape), dtype=torch.float32, device=self.device)
        grid[0, 0, local_coords[:, 0], local_coords[:, 1], local_coords[:, 2]] = 1.0

        k = 2 * radius + 1
        pad = radius
        dilated_grid = F.max_pool3d(grid, kernel_size=k, stride=1, padding=pad)

        eroded_grid = -F.max_pool3d(-dilated_grid, kernel_size=k, stride=1, padding=pad)

        original_mask = (grid > 0.5)
        eroded_grid[original_mask] = grid[original_mask]

        closed_local_coords = torch.nonzero(eroded_grid[0, 0] > 0.5)

        closed_global_coords = closed_local_coords + vmin

        return torch.unique(self.encode(closed_global_coords))

    def update(self, gt, pred):
        p_gt = torch.from_numpy(np.array(gt.points)).to(self.device)
        p_pred = torch.from_numpy(np.array(pred.points)).to(self.device)

        dist_gt = torch.norm(p_gt[:, :2], dim=1)
        dist_pred = torch.norm(p_pred[:, :2], dim=1)

        for j, (d_min, d_max) in enumerate(self.dist_bins):
            sub_gt = p_gt[(dist_gt >= d_min) & (dist_gt < d_max)]
            sub_pred = p_pred[(dist_pred >= d_min) & (dist_pred < d_max)]

            if len(sub_gt) == 0 and len(sub_pred) == 0:
                continue

            for i, (vsize, radius) in enumerate(zip(self.voxel_sizes, self.kernel_radii)):
                v_gt = torch.floor(sub_gt / vsize).long()
                v_pred = torch.floor(sub_pred / vsize).long()

                id_pred = self.morphology_close(v_pred, radius)

                if self.close_gt:
                    id_gt = self.morphology_close(v_gt, radius)
                else:
                    if len(v_gt) > 0:
                        id_gt = torch.unique(self.encode(v_gt))
                    else:
                        id_gt = torch.empty(0, dtype=torch.int64, device=self.device)

                tp_mask = torch.isin(id_gt, id_pred)
                tp = torch.sum(tp_mask)
                fn = len(id_gt) - tp
                fp = len(id_pred) - tp

                self.conf_matrix[i, j, 0] += tp
                self.conf_matrix[i, j, 1] += fn
                self.conf_matrix[i, j, 2] += fp

    def compute(self):
        conf_np = self.conf_matrix.cpu().numpy()
        res = {}
        for i, vsize in enumerate(self.voxel_sizes):
            res[vsize] = {}
            for j, d in enumerate(self.dist_bins):
                tp, fn, fp = conf_np[i, j]
                iou = tp / (tp + fn + fp + 1e-15)
                res[vsize][f"{d[0]}-{d[1]}m"] = float(iou)
        return res

    def reset(self):
        self.conf_matrix.zero_()

class CompletionIoU_MultiRange_Morphology_():
    def __init__(self,
                 voxel_sizes=[0.5, 0.2, 0.1],
                 dist_bins=[(0, 20), (20, 35), (35, 50)],
                 kernel_radius=3,
                 close_gt=False):
        self.voxel_sizes = voxel_sizes
        self.dist_bins = dist_bins
        self.close_gt = close_gt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        r_range = list(range(-kernel_radius, kernel_radius + 1))
        self.offsets = torch.tensor(
            list(itertools.product(r_range, repeat=3)),
            device=self.device
        )
        self.kernel_size = len(self.offsets)

        self.conf_matrix = torch.zeros((len(voxel_sizes), len(dist_bins), 3),
                                       dtype=torch.int64, device=self.device)

    def encode(self, v):
        """ 3D  1D ID"""
        return ((v[:, 0] + 2000) << 40) | ((v[:, 1] + 2000) << 20) | (v[:, 2] + 2000)

    def decode(self, ids):
        """ 1D ID  3D """
        mask = (1 << 20) - 1
        x = (ids >> 40) - 2000
        y = ((ids >> 20) & mask) - 2000
        z = (ids & mask) - 2000
        return torch.stack([x, y, z], dim=1)

    def dilate_ids(self, ids):
        coords = self.decode(ids)
        dilated_id_list = []
        for offset in self.offsets:
            shifted_coords = coords + offset
            dilated_id_list.append(self.encode(shifted_coords))

        return torch.unique(torch.cat(dilated_id_list))

    def erode_ids(self, ids):
        coords = self.decode(ids)
        counts = torch.zeros(len(ids), dtype=torch.int32, device=self.device)

        for offset in self.offsets:
            shifted_coords = coords + offset
            shifted_ids = self.encode(shifted_coords)
            counts += torch.isin(shifted_ids, ids).int()

        mask = (counts == self.kernel_size)
        return ids[mask]

    def morphology_close(self, v_coords):
        if len(v_coords) == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)

        base_ids = torch.unique(self.encode(v_coords))
        dilated_ids = self.dilate_ids(base_ids)
        closed_ids = self.erode_ids(dilated_ids)

        return closed_ids

    def update(self, gt, pred):
        p_gt = torch.from_numpy(np.array(gt.points)).to(self.device)
        p_pred = torch.from_numpy(np.array(pred.points)).to(self.device)

        dist_gt = torch.norm(p_gt[:, :2], dim=1)
        dist_pred = torch.norm(p_pred[:, :2], dim=1)

        for j, (d_min, d_max) in enumerate(self.dist_bins):
            sub_gt = p_gt[(dist_gt >= d_min) & (dist_gt < d_max)]
            sub_pred = p_pred[(dist_pred >= d_min) & (dist_pred < d_max)]

            if len(sub_gt) == 0 and len(sub_pred) == 0:
                continue

            for i, vsize in enumerate(self.voxel_sizes):
                v_gt = torch.floor(sub_gt / vsize).long()
                v_pred = torch.floor(sub_pred / vsize).long()

                id_pred = self.morphology_close(v_pred)

                if self.close_gt:
                    id_gt = self.morphology_close(v_gt)
                else:
                    if len(v_gt) > 0:
                        id_gt = torch.unique(self.encode(v_gt))
                    else:
                        id_gt = torch.empty(0, dtype=torch.int64, device=self.device)

                tp_mask = torch.isin(id_gt, id_pred)
                tp = torch.sum(tp_mask)
                fn = len(id_gt) - tp
                fp = len(id_pred) - tp

                self.conf_matrix[i, j, 0] += tp
                self.conf_matrix[i, j, 1] += fn
                self.conf_matrix[i, j, 2] += fp

    def compute(self):
        conf_np = self.conf_matrix.cpu().numpy()
        res = {}
        for i, vsize in enumerate(self.voxel_sizes):
            res[vsize] = {}
            for j, d in enumerate(self.dist_bins):
                tp, fn, fp = conf_np[i, j]
                iou = tp / (tp + fn + fp + 1e-15)
                res[vsize][f"{d[0]}-{d[1]}m"] = float(iou)
        return res

    def reset(self):
        self.conf_matrix.zero_()

class CompletionIoU_MultiRange():
    def __init__(self, voxel_sizes=[0.5, 0.2, 0.1], dist_bins=[(0, 20), (20, 35), (35, 50)]):
        self.voxel_sizes = voxel_sizes
        self.dist_bins = dist_bins
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_matrix = torch.zeros((len(voxel_sizes), len(dist_bins), 3),
                                       dtype=torch.int64, device=self.device)

    def update(self, gt, pred):
        p_gt = torch.from_numpy(np.array(gt.points)).to(self.device)
        p_pred = torch.from_numpy(np.array(pred.points)).to(self.device)

        dist_gt = torch.norm(p_gt[:, :2], dim=1)
        dist_pred = torch.norm(p_pred[:, :2], dim=1)

        for j, (d_min, d_max) in enumerate(self.dist_bins):
            sub_gt = p_gt[(dist_gt >= d_min) & (dist_gt < d_max)]
            sub_pred = p_pred[(dist_pred >= d_min) & (dist_pred < d_max)]

            if len(sub_gt) == 0 and len(sub_pred) == 0:
                continue

            for i, vsize in enumerate(self.voxel_sizes):
                v_gt = torch.floor(sub_gt / vsize).long()
                v_pred = torch.floor(sub_pred / vsize).long()

                def encode(v):
                    return ((v[:, 0] + 2000) << 40) | \
                        ((v[:, 1] + 2000) << 20) | \
                        (v[:, 2] + 2000)

                id_gt = torch.unique(encode(v_gt))
                id_pred = torch.unique(encode(v_pred))

                tp_mask = torch.isin(id_gt, id_pred)
                tp = torch.sum(tp_mask)
                fn = len(id_gt) - tp
                fp = len(id_pred) - tp

                self.conf_matrix[i, j, 0] += tp
                self.conf_matrix[i, j, 1] += fn
                self.conf_matrix[i, j, 2] += fp

    def compute(self):
        conf_np = self.conf_matrix.cpu().numpy()
        res = {}
        for i, vsize in enumerate(self.voxel_sizes):
            res[vsize] = {}
            for j, d in enumerate(self.dist_bins):
                tp, fn, fp = conf_np[i, j]
                iou = tp / (tp + fn + fp + 1e-15)
                res[vsize][f"{d[0]}-{d[1]}m"] = float(iou)
        return res

    def reset(self):
        self.conf_matrix.zero_()

class CompletionIoU():
    def __init__(self, voxel_sizes=[0.5, 0.2, 0.1]):
        self.voxel_sizes = voxel_sizes
        self.conf_matrix = np.zeros((len(self.voxel_sizes), 3)).astype(np.uint64)

    def update(self, gt, pred):
        max_range = 50.
        vox_coords_gt = np.array(gt.points)
        vox_coords_pred = np.array(pred.points)
        vox_coords_all = np.concatenate([vox_coords_gt, vox_coords_pred], axis=0)
        z_min = np.floor(vox_coords_all[:, 2].min())
        z_max = np.ceil(vox_coords_all[:, 2].max())
        for i, vsize in enumerate(self.voxel_sizes):
            bins = [int(2 * max_range / vsize), int(2 * max_range / vsize), int((z_max-z_min) / vsize)]


            hist_gt = np.histogramdd(
                vox_coords_gt, bins=bins,
                range=([-max_range, max_range], [-max_range, max_range], [z_min, z_max])
            )[0].astype(bool).astype(int)

            hist_pred = np.histogramdd(
                vox_coords_pred, bins=bins,
                range=([-max_range, max_range], [-max_range, max_range], [z_min, z_max])
            )[0].astype(bool).astype(int)

            self.conf_matrix[i][0] += ((hist_gt == 1) & (hist_pred == 1)).sum()
            self.conf_matrix[i][1] += ((hist_gt == 1) & (hist_pred == 0)).sum()
            self.conf_matrix[i][2] += ((hist_gt == 0) & (hist_pred == 1)).sum()


    def compute(self):
        res_vsizes = {}
        for i, vsize in enumerate(self.voxel_sizes):
            tp = self.conf_matrix[i][0]
            fn = self.conf_matrix[i][1]
            fp = self.conf_matrix[i][2]

            intersection = tp
            union = tp + fn + fp + 1e-15

            res_vsizes[vsize] = intersection / union

        return res_vsizes

    def reset(self):
        self.conf_matrix = np.zeros((len(self.voxel_sizes), 3)).astype(np.uint)


class _CompletionIoU():
    def __init__(self, voxel_sizes=[0.5, 0.2, 0.1]):
        self.voxel_sizes = voxel_sizes
        self.conf_matrix = np.zeros((len(self.voxel_sizes), 3)).astype(np.uint64)

    def update(self, gt, pred):
        max_range = 50.
        for i, vsize in enumerate(self.voxel_sizes):
            bins = int(2 * max_range / vsize)
            vox_coords_gt = np.round(np.array(gt.points) / vsize).astype(int)
            hist_gt = np.histogramdd(
                    vox_coords_gt, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range])
            )[0].astype(bool).astype(int)

            vox_coords_pred = np.round(np.array(pred.points) / vsize).astype(int)
            hist_pred = np.histogramdd(
                    vox_coords_pred, bins=bins, range=([-max_range,max_range],[-max_range,max_range],[-max_range,max_range])
            )[0].astype(bool).astype(int)

            tp = ((hist_gt == 1) & (hist_pred == 1)).sum()
            fn = ((hist_gt == 1) & (hist_pred == 0)).sum()
            fp = ((hist_gt == 0) & (hist_pred == 1)).sum()
            tn = ((hist_gt == 0) & (hist_pred == 0)).sum()

            gt_sum = (hist_gt == 1).sum()
            pre_sum = (hist_pred == 1).sum()
            print('tp fn fp tn numpy', tp, fn, fp, tn, gt_sum, pre_sum)
            print('pred.points', pred.points.shape)
            print('gt.points', gt.points.shape)
            self.conf_matrix[i][0] += ((hist_gt == 1) & (hist_pred == 1)).sum()
            self.conf_matrix[i][1] += ((hist_gt == 1) & (hist_pred == 0)).sum()
            self.conf_matrix[i][2] += ((hist_gt == 0) & (hist_pred == 1)).sum()

    def compute(self):
        res_vsizes = {}
        for i, vsize in enumerate(self.voxel_sizes):
            tp = self.conf_matrix[i][0]
            fn = self.conf_matrix[i][1]
            fp = self.conf_matrix[i][2]

            intersection = tp
            union = tp + fn + fp + 1e-15

            res_vsizes[vsize] = intersection / union

        return res_vsizes

    def reset(self):
        self.conf_matrix = np.zeros((len(self.voxel_sizes), 3)).astype(np.uint)


class ChamferDistance_MultiRange():
    def __init__(self, dist_bins=[(0, 20), (20, 35), (35, 50)]):
        self.dist_bins = dist_bins
        self.dists_dict = {f"{d[0]}_{d[1]}m": [] for d in self.dist_bins}

    def update(self, gt_pcd, pt_pcd):
        pts_gt = np.asarray(gt_pcd.points)
        pts_pred = np.asarray(pt_pcd.points)

        dist_gt_origin = np.linalg.norm(pts_gt[:, :2], axis=1)
        dist_pred_origin = np.linalg.norm(pts_pred[:, :2], axis=1)

        for d_min, d_max in self.dist_bins:
            idx_gt = np.where((dist_gt_origin >= d_min) & (dist_gt_origin < d_max))[0]
            idx_pred = np.where((dist_pred_origin >= d_min) & (dist_pred_origin < d_max))[0]

            if len(idx_gt) == 0 or len(idx_pred) == 0:
                continue

            sub_gt_pcd = gt_pcd.select_by_index(idx_gt)
            sub_pred_pcd = pt_pcd.select_by_index(idx_pred)

            dist_pt_2_gt = np.asarray(sub_pred_pcd.compute_point_cloud_distance(sub_gt_pcd))
            dist_gt_2_pt = np.asarray(sub_gt_pcd.compute_point_cloud_distance(sub_pred_pcd))

            cd_val = (np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2
            self.dists_dict[f"{d_min}_{d_max}m"].append(cd_val)

    def reset(self):
        self.dists_dict = {f"{d[0]}_{d[1]}m": [] for d in self.dist_bins}

    def compute(self):
        results = {}
        all_means = []

        for key, dist_list in self.dists_dict.items():
            if len(dist_list) > 0:
                arr = np.array(dist_list)
                m, s = arr.mean(), arr.std()
                results[key] = {"mean": m, "std": s}
                all_means.append(m)
            else:
                results[key] = {"mean": 0.0, "std": 0.0}

        overall_mean = np.mean(all_means) if all_means else 0.0
        return results, overall_mean


class DCD_MultiRange():
    def __init__(self, dist_bins=[(0, 20), (20, 35), (35, 50)], alpha=5, n_lambda=1, non_reg=False):
        self.dist_bins = dist_bins
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.non_reg = non_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def update(self, gt_pcd, pt_pcd):
        pts_gt = np.asarray(gt_pcd.points)
        pts_pred = np.asarray(pt_pcd.points)

        dist_gt_origin = np.linalg.norm(pts_gt[:, :2], axis=1)
        dist_pred_origin = np.linalg.norm(pts_pred[:, :2], axis=1)

        for d_min, d_max in self.dist_bins:
            idx_gt = np.where((dist_gt_origin >= d_min) & (dist_gt_origin < d_max))[0]
            idx_pred = np.where((dist_pred_origin >= d_min) & (dist_pred_origin < d_max))[0]

            if len(idx_gt) == 0 or len(idx_pred) == 0:
                continue

            sub_gt = torch.from_numpy(pts_gt[idx_gt]).to(self.device, dtype=torch.float32).unsqueeze(0).contiguous()
            sub_pred = torch.from_numpy(pts_pred[idx_pred]).to(self.device, dtype=torch.float32).unsqueeze(0).contiguous()

            dcd_val = calc_dcd(
                sub_pred,
                sub_gt,
                alpha=self.alpha,
                n_lambda=self.n_lambda,
                non_reg=self.non_reg
            )[0].item()
            self.dists_dict[f"{d_min}_{d_max}m"].append(dcd_val)

    def reset(self):
        self.dists_dict = {f"{d[0]}_{d[1]}m": [] for d in self.dist_bins}

    def compute(self):
        results = {}
        all_means = []

        for key, dist_list in self.dists_dict.items():
            if len(dist_list) > 0:
                arr = np.array(dist_list)
                m, s = arr.mean(), arr.std()
                results[key] = {"mean": m, "std": s}
                all_means.append(m)
            else:
                results[key] = {"mean": 0.0, "std": 0.0}

        overall_mean = np.mean(all_means) if all_means else 0.0
        return results, overall_mean


class DCD():
    def __init__(self, alpha=10, n_lambda=1, non_reg=False):
        self.alpha = alpha
        self.n_lambda = n_lambda
        self.non_reg = non_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dists = []

    def update(self, gt_pcd, pt_pcd):
        pts_gt = np.asarray(gt_pcd.points, dtype=np.float32)
        pts_pred = np.asarray(pt_pcd.points, dtype=np.float32)

        if len(pts_gt) == 0 or len(pts_pred) == 0:
            return

        gt_tensor = torch.from_numpy(pts_gt).to(self.device).unsqueeze(0).contiguous()
        pred_tensor = torch.from_numpy(pts_pred).to(self.device).unsqueeze(0).contiguous()
        dcd_val = calc_dcd(
            pred_tensor,
            gt_tensor,
            alpha=self.alpha,
            n_lambda=self.n_lambda,
            non_reg=self.non_reg,
        )[0].item()
        self.dists.append(dcd_val)

    def reset(self):
        self.dists = []

    def compute(self):
        if len(self.dists) == 0:
            return 0.0, 0.0
        dists = np.array(self.dists)
        return float(dists.mean()), float(dists.std())

class ChamferDistance():
    def __init__(self):
        self.dists = []

        return

    def update(self, gt_pcd, pt_pcd):
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))
        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        self.dists.append((np.mean(dist_gt_2_pt) + np.mean(dist_pt_2_gt)) / 2)

    def reset(self):
        self.dists = []

    def compute(self):
        cdist = np.array(self.dists)
        return cdist.mean(), cdist.std()


class ChamferDistanceTorch():
    def __init__(self):
        self.dists = []

    def update(self, gt_pcd, pt_pcd):

        mean_dist = _chamfer_sqrt(gt_pcd, pt_pcd)
        self.dists.append(mean_dist)

    def reset(self):
        self.dists = []

    def compute(self):
        dists_tensor = torch.stack(self.dists)
        return dists_tensor.mean().item(), dists_tensor.std().item()

class PrecisionRecall(Metrics3D):

    def __init__(self, min_t, max_t, num):
        self.thresholds = np.linspace(min_t, max_t, num)
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def update(self, gt_pcd, pt_pcd):
        dist_pt_2_gt = np.asarray(pt_pcd.compute_point_cloud_distance(gt_pcd))

        dist_gt_2_pt = np.asarray(gt_pcd.compute_point_cloud_distance(pt_pcd))

        for t in self.thresholds:
            p = np.where(dist_pt_2_gt < t)[0]
            p = 100 / len(dist_pt_2_gt) * len(p)
            self.pr_dict[t].append(p)

            r = np.where(dist_gt_2_pt < t)[0]
            r = 100 / len(dist_gt_2_pt) * len(r)
            self.re_dict[t].append(r)

            if p == 0 or r == 0:
                f = 0
            else:
                f = 2 * p * r / (p + r)
            self.f1_dict[t].append(f)

    def reset(self):
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def compute_at_threshold(self, threshold):
        t = self.find_nearest_threshold(threshold)
        pr = sum(self.pr_dict[t]) / len(self.pr_dict[t])
        re = sum(self.re_dict[t]) / len(self.re_dict[t])
        f1 = sum(self.f1_dict[t]) / len(self.f1_dict[t])
        return pr, re, f1, t

    def compute_auc(self):
        dx = self.thresholds[1] - self.thresholds[0]
        perfect_predictor = scipy.integrate.simpson(np.ones_like(self.thresholds), dx=dx)

        pr, re, f1 = self.compute_at_all_thresholds()

        pr_area = scipy.integrate.simpson(pr, dx=dx)
        norm_pr_area = pr_area / perfect_predictor

        re_area = scipy.integrate.simpson(re, dx=dx)
        norm_re_area = re_area / perfect_predictor

        f1_area = scipy.integrate.simpson(f1, dx=dx)
        norm_f1_area = f1_area / perfect_predictor


        return norm_pr_area, norm_re_area, norm_f1_area

    def compute_at_all_thresholds(self):
        pr = [sum(self.pr_dict[t]) / len(self.pr_dict[t]) for t in self.thresholds]
        re = [sum(self.re_dict[t]) / len(self.re_dict[t]) for t in self.thresholds]
        f1 = [sum(self.f1_dict[t]) / len(self.f1_dict[t]) for t in self.thresholds]
        return pr, re, f1

    def find_nearest_threshold(self, value):
        idx = (np.abs(self.thresholds - value)).argmin()
        return self.thresholds[idx]


class PrecisionRecallTorch(Metrics3D):
    def __init__(self, min_t, max_t, num):
        self.thresholds = torch.linspace(min_t, max_t, num).tolist()
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def update(self, gt_pcd, pt_pcd):
        """
         ground truth  (gt_pcd)  (pt_pcd) ，
         precisionrecall  f1 score（）
         torch
        """

        dist_pt_2_gt = torch.tensor(pt_pcd.compute_point_cloud_distance(gt_pcd), dtype=torch.float32)
        dist_gt_2_pt = torch.tensor(gt_pcd.compute_point_cloud_distance(pt_pcd), dtype=torch.float32)

        num_pt = dist_pt_2_gt.numel()
        num_gt = dist_gt_2_pt.numel()

        for t in self.thresholds:
            p_count = torch.sum(dist_pt_2_gt < t).item()
            p = 100.0 * p_count / num_pt
            self.pr_dict[t].append(p)

            r_count = torch.sum(dist_gt_2_pt < t).item()
            r = 100.0 * r_count / num_gt
            self.re_dict[t].append(r)

            if p == 0 or r == 0:
                f = 0
            else:
                f = 2 * p * r / (p + r)
            self.f1_dict[t].append(f)

    def reset(self):
        """"""
        self.pr_dict = {t: [] for t in self.thresholds}
        self.re_dict = {t: [] for t in self.thresholds}
        self.f1_dict = {t: [] for t in self.thresholds}

    def compute_at_threshold(self, threshold):
        """
         threshold  precisionrecallf1 score
        ：(pr, re, f1, nearest_threshold)
        """
        t_nearest = self.find_nearest_threshold_torch(threshold)
        pr = sum(self.pr_dict[t_nearest]) / len(self.pr_dict[t_nearest])
        re = sum(self.re_dict[t_nearest]) / len(self.re_dict[t_nearest])
        f1 = sum(self.f1_dict[t_nearest]) / len(self.f1_dict[t_nearest])
        return pr, re, f1, t_nearest

    def compute_auc(self):
        """
         torch.trapz  precisionrecall  f1 score  AUC
        """
        thresholds_tensor = torch.tensor(self.thresholds, dtype=torch.float32)
        perfect_predictor = torch.trapz(torch.ones_like(thresholds_tensor), thresholds_tensor)

        pr_list, re_list, f1_list = self.compute_at_all_thresholds()

        pr_tensor = torch.tensor(pr_list, dtype=torch.float32)
        re_tensor = torch.tensor(re_list, dtype=torch.float32)
        f1_tensor = torch.tensor(f1_list, dtype=torch.float32)

        pr_area = torch.trapz(pr_tensor, thresholds_tensor)
        norm_pr_area = pr_area / perfect_predictor

        re_area = torch.trapz(re_tensor, thresholds_tensor)
        norm_re_area = re_area / perfect_predictor

        f1_area = torch.trapz(f1_tensor, thresholds_tensor)
        norm_f1_area = f1_area / perfect_predictor

        return norm_pr_area.item(), norm_re_area.item(), norm_f1_area.item()

    def compute_at_all_thresholds(self):
        """
         precisionrecall  f1 score，

        """
        pr = [sum(self.pr_dict[t]) / len(self.pr_dict[t]) for t in self.thresholds]
        re = [sum(self.re_dict[t]) / len(self.re_dict[t]) for t in self.thresholds]
        f1 = [sum(self.f1_dict[t]) / len(self.f1_dict[t]) for t in self.thresholds]
        return pr, re, f1

    def find_nearest_threshold_torch(self, value):
        """
         value
        """
        return min(self.thresholds, key=lambda x: abs(x - value))
