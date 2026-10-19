"""Selective reuse of FPSGen GPU CD and point-voxel IoU implementations.

No distribution metrics, EMD, spacing, F-score or semantic evaluator is created.
The optional hierarchy only supplies the previously frozen physical ROI contract.
"""
import math
import torch


def select_metrics(names):
    names = tuple(names)
    if not names or len(set(names)) != len(names) or any(n not in ('cd', 'iou05') for n in names):
        raise ValueError('Choose distinct metrics from cd,iou05')
    return names


def physical_roi_points(points, geometry, *, target_points=False):
    """GPU counterpart of the frozen physical_grid_ids point/cell validity."""
    p = points.double()
    origin = p.new_tensor(geometry.origin)
    maximum = p.new_tensor(geometry.maximum)
    keep = ((p>origin)&(p<maximum)).all(-1)
    if target_points:
        keep &= p.square().sum(-1)<geometry.sphere_radius**2
    candidate = p[keep]
    coordinates = torch.floor((candidate-origin)/.5)
    lo = origin+coordinates*.5
    hi = torch.minimum(lo+.5, maximum)
    nearest = torch.maximum(lo, torch.zeros_like(lo))+torch.minimum(hi, torch.zeros_like(hi))
    valid_cell = nearest.square().sum(-1)<geometry.sphere_radius**2
    return points[keep][valid_cell]


def pad_existing_points(clouds):
    """Maskable CUDA batch padding, not resampling or a changed point cloud.

    Repeating an ALREADY EXISTING support point cannot change a nearest distance.
    Query padding is excluded from every reduction using the original lengths.
    Zero/far artificial points must never be inserted into a valid support set.
    """
    if not clouds or any(p.ndim != 2 or p.shape[1] != 3 or not len(p) for p in clouds):
        raise ValueError('Nonempty ragged XYZ clouds required')
    lengths = [len(p) for p in clouds]
    maximum = max(lengths)
    values = [torch.cat((p.float(), p[:1].float().expand(maximum-len(p), -1)))
        if len(p) < maximum else p.float() for p in clouds]
    return torch.stack(values).contiguous(), lengths


class SelectedPointMetrics:
    def __init__(self, metrics=('cd', 'iou05'), *, cd_backend='cuda', geometry=None):
        self.metrics = select_metrics(metrics)
        if cd_backend not in ('cuda', 'keops'):
            raise ValueError('Existing project CUDA/KeOps CD backends only')
        self.cd_backend = cd_backend
        self.geometry = geometry
        self.chamfer = None
        if 'cd' in self.metrics and cd_backend == 'cuda':
            from fpsgen.ops.chamfer import Chamfer3DDist
            self.chamfer = Chamfer3DDist()

    @torch.no_grad()
    def evaluate(self, predicted, target):
        if predicted.device != target.device:
            raise ValueError('Prediction and target must share a device')
        for points in (predicted, target):
            if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
                raise ValueError('Finite XYZ clouds required')
        result = {}
        if 'cd' in self.metrics:
            if predicted.device.type != 'cuda':
                raise ValueError('CD execution requires the explicitly admitted GPU')
            if not len(predicted) or not len(target):
                result.update(valid_cd=False, cd_m=None, pred_to_gt_m=None, gt_to_pred_m=None)
            else:
                with torch.autocast('cuda', enabled=False):
                    p, q = predicted.float().contiguous(), target.float().contiguous()
                    if self.cd_backend == 'cuda':
                        dp, dq, _, _ = self.chamfer(p[None], q[None])
                        forward = dp.clamp_min(0).sqrt().double().mean()
                        reverse = dq.clamp_min(0).sqrt().double().mean()
                    else:
                        from fpsgen.ops.endpoint_refinement.knn import knn
                        # Reuse project sparse KNN; recompute selected distances
                        # exactly as voxel_point.metrics.chamfer_sqrt does.
                        _, ip = knn(p, q, 1, 'keops')
                        _, iq = knn(q, p, 1, 'keops')
                        forward = (p-q[ip[:, 0]]).norm(dim=-1).double().mean()
                        reverse = (q-p[iq[:, 0]]).norm(dim=-1).double().mean()
                    a, b = float(forward), float(reverse)
                    if not math.isfinite(a+b):
                        raise ValueError('Nonfinite CD')
                    result.update(valid_cd=True, cd_m=.5*(a+b), pred_to_gt_m=a, gt_to_pred_m=b)
        if 'iou05' in self.metrics:
            from fpsgen.voxel_point.metrics import point_voxel_occupancy_metrics
            p, q = predicted, target
            if self.geometry is not None:
                # The configured origin is an integer multiple of0.5m; standard
                # project floor(XYZ/.5) cells match historical shifted IDs.
                if any(not math.isclose(x/.5, round(x/.5), abs_tol=1e-12) for x in self.geometry.origin):
                    raise ValueError('Historical origin must align with existing0.5m cells')
                p = physical_roi_points(p, self.geometry)
                q = physical_roi_points(q, self.geometry, target_points=True)
            value = point_voxel_occupancy_metrics(p, q, voxel_size=.5)
            result.update(iou05=value['iou'], tp=value['tp'], fp=value['fp'], fn=value['fn'])
        return result

    @torch.no_grad()
    def evaluate_batch(self, predictions, targets):
        if not predictions or len(predictions) != len(targets):
            raise ValueError('One raw target per generated scene required')
        if ('cd' not in self.metrics or self.cd_backend != 'cuda'
            or any(not len(p) or not len(q) for p,q in zip(predictions,targets))):
            return [self.evaluate(p,q) for p,q in zip(predictions,targets)]
        for points in (*predictions,*targets):
            if points.device != predictions[0].device or points.device.type != 'cuda':
                raise ValueError('Admitted GPU clouds only')
            if points.ndim != 2 or points.shape[1] != 3 or not bool(torch.isfinite(points).all()):
                raise ValueError('Finite XYZ required')
        p, np_real = pad_existing_points(predictions)
        q, nq_real = pad_existing_points(targets)
        with torch.autocast('cuda', enabled=False):
            dp, dq, _, _ = self.chamfer(p,q)
            rows = []
            for index,(n,m) in enumerate(zip(np_real,nq_real)):
                a = float(dp[index,:n].clamp_min(0).sqrt().double().mean())
                b = float(dq[index,:m].clamp_min(0).sqrt().double().mean())
                if not math.isfinite(a+b):
                    raise ValueError('Nonfinite batched CD')
                rows.append(dict(valid_cd=True,cd_m=.5*(a+b),pred_to_gt_m=a,gt_to_pred_m=b))
        if 'iou05' in self.metrics:
            iou_only = SelectedPointMetrics(('iou05',),geometry=self.geometry)
            for row, predicted, target in zip(rows,predictions,targets):
                row.update(iou_only.evaluate(predicted,target))
        return rows
