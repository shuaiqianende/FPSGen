"""Teacher transport mapping for source-indexed endpoint construction.

The teacher is used only during training. Given a BEV-supported source
``P_0`` and the complete scene ``P_gt`` as a sparse anchor, it predicts the
residual mapping that defines the source-indexed clean endpoint ``P_1_dagger``.
"""

import torch
import fpsgen.models.minkunet as minknet
import MinkowskiEngine as ME

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from fpsgen.utils.collations_gen import bev_resample, feats_to_coord
from chamfer3D.dist_chamfer_3D import chamfer_3DDist
from torch import Tensor
from typing import Tuple

chamfer_dist = chamfer_3DDist()

@torch.no_grad()
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Return squared distances and indices for batched k-nearest neighbors.

    Args:
        q_points: Query tensor with shape ``[..., N, C]``.
        s_points: Support tensor with shape ``[..., M, C]``.
        k: Number of neighbors.

    Returns:
        Squared distances and support indices with shape ``[..., N, k]``.
    """
    import pykeops
    pykeops.set_verbose(False)
    xi = pykeops.torch.LazyTensor(q_points.unsqueeze(-2))
    xj = pykeops.torch.LazyTensor(s_points.unsqueeze(-3))
    dij = (xi - xj).sqnorm2()
    knn_d2, knn_indices = dij.Kmin_argKmin(k, dim=q_points.dim() - 1)
    return knn_d2, knn_indices

def chamfer_sqrt(p1, p2):
    d1, d2, idx1, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2, idx1

class DiffusionPoints(LightningModule):
    r"""Learn the teacher transport mapping :math:`\Gamma_\eta`.

    The network receives the BEV-supported source :math:`\mathcal{P}_0` and
    the complete scene :math:`\mathcal{P}^{gt}`. Its residual output defines
    the source-indexed clean endpoint :math:`\mathcal{P}^{\dagger}_1`, which
    supplies the Approx-OT pair used by the student point flow.
    """
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.model = minknet.MinkUNet_NoTime(in_channels=3, out_channels=self.hparams['model']['out_dim'])

        self.cd_ls = chamfer_sqrt
        self.cnt = 0

    def forward(self, x_full, x_full_sparse, x_part, t):
        """Predict the network residual for each source-indexed point.

        The retained backbone follows the historical denoising sign convention,
        so the source-to-endpoint displacement is the negative network output.
        ``t`` is used only to recover the dense batch shape because the teacher
        deliberately omits the diffusion/flow time branch.
        """
        part_feat = self.partial_enc(x_part)
        out = self.model(x_full, x_full_sparse, part_feat)
        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def points_to_tensor(self, x_feats, mean, std):
        """Convert ``[B, N, 3]`` metric points to a sparse TensorField.

        MinkowskiEngine prepends a batch index, producing features
        ``[batch, x, y, z]``. The metric XYZ values are retained as features,
        while sparse coordinates are rounded to voxel indices at
        ``data.resolution``. Points colliding in one voxel are averaged.

        ``mean`` and ``std`` remain in the signature for compatibility with
        the older normalized LiDiff interface; FPSGen quantizes metric XYZ.
        """
        x_feats = ME.utils.batched_coordinates(list(x_feats[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord[:,1:] = feats_to_coord(x_feats[:,1:], self.hparams['data']['resolution'], mean, std)

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t

    def _shared_step(self, batch: dict, metric_prefix: str):
        r"""Optimize the set-level teacher objective.

        The endpoint is supervised by Chamfer distance and the local repulsion
        regularizer from :math:`\mathcal{L}_T`.
        """
        torch.cuda.empty_cache()
        t = torch.zeros(
            (batch['pcd_full'].shape[0],),
            dtype=torch.long,
            device=self.device,
        )

        # P_0 = R(B_bar; N, Sigma), with B_bar = Phi(P_gt). Unlike a global
        # Gaussian source, P_0 follows the scene-level support encoded by the
        # ground-truth BEV density.
        _, noisy_pcd, _ = bev_resample(
            batch['pcd_full'], target_pts=batch['pcd_full'].shape[1]
        )

        x_full = self.points_to_tensor(noisy_pcd, t, t)
        x_part = self.points_to_tensor(batch['pcd_full'], t, t)
        target_scene = batch['pcd_full']

        network_residual = self.forward(x_full, x_full.sparse(), x_part, t)
        source_to_endpoint = -network_residual
        teacher_endpoint = noisy_pcd + source_to_endpoint

        loss_cd, _ = self.cd_ls(teacher_endpoint, target_scene)

        # L_rep penalizes endpoint pairs closer than r_rep = 0.2 m. Neighbor
        # indices are selected without gradients; the selected distances remain
        # differentiable with respect to the endpoint coordinates.
        _, knn_indices = keops_knn(
            teacher_endpoint.detach(), teacher_endpoint.detach(), k=2
        )
        nn_idx = knn_indices[:, :, 1]
        B, N, _ = teacher_endpoint.shape
        batch_idx = torch.arange(B, device=self.device).view(-1, 1).expand(B, N)
        nn_points = teacher_endpoint[batch_idx, nn_idx, :]
        nn_dist = torch.norm(teacher_endpoint - nn_points, p=2, dim=-1)
        knn_threshold = 0.2
        loss_knn = torch.mean(torch.relu(knn_threshold - nn_dist))

        lambda_knn = 0.5
        loss = loss_cd + lambda_knn * loss_knn
        self.log(f'{metric_prefix}/loss', loss, prog_bar=True)
        self.log(f'{metric_prefix}/loss_cd', loss_cd, prog_bar=True)
        self.log(f'{metric_prefix}/loss_knn', loss_knn, prog_bar=True)

        torch.cuda.empty_cache()

        if metric_prefix == 'train':
            self.cnt += 1
        return loss

    def training_step(self, batch: dict, batch_idx):
        """Fit the teacher from a density-weighted BEV-supported source."""
        return self._shared_step(batch, metric_prefix='train')

    def validation_step(self, batch:dict, batch_idx):
        """Validation is disabled in the released three-stage schedule."""
        return None

    def test_step(self, batch:dict, batch_idx):
        with torch.no_grad():
            return self._shared_step(batch, metric_prefix='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['train']['lr'], betas=(0.9, 0.999))

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [scheduler_dict]
