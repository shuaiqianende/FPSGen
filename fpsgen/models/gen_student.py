"""Approximate-OT point flow for flexible-condition scene generation.

The student learns the time-dependent point velocity field ``v_psi`` between
the BEV-supported source ``P_0`` and the teacher-estimated, source-indexed clean
endpoint ``P_1_dagger``. The same network supports unconditional, layout-only,
LiDAR-only, and mixed-condition generation.
"""

import math
import os

import torch
import torch.nn.functional as F
import fpsgen.models.minkunet as minknet
import fpsgen.models.gen_img as genimg
import fpsgen.models.minkunet_refine as minknetin
import MinkowskiEngine as ME

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from fpsgen.utils.collations_gen import bev_resample


class DiffusionPoints(LightningModule):
    r"""Learn the Approximate-OT point velocity field :math:`v_\psi`.

    For each teacher pair :math:`(\mathcal{P}_0,\mathcal{P}^{\dagger}_1)`,
    the model regresses the constant conditional-flow-matching target
    :math:`u_P=\mathcal{P}^{\dagger}_1-\mathcal{P}_0` along the linear path.
    BEV context and the active condition tuple :math:`C_m` modulate every
    sparse U-Net stage.
    """
    def __init__(self, hparams:dict, data_module: LightningDataModule = None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        diff_path = self.hparams['train'].get('teacher_checkpoint', '')
        if not diff_path:
            raise ValueError(
                'train.teacher_checkpoint is required for student training. '
                'Set it to the stage-2 teacher checkpoint in the YAML config.'
            )
        if not os.path.isfile(diff_path):
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {diff_path}")
        # Load on CPU first so a checkpoint saved from any physical GPU remains
        # portable. ``load_state_dict`` moves tensors into the model afterwards.
        ckpt_diff = torch.load(diff_path, map_location='cpu')
        if 'state_dict' not in ckpt_diff:
            raise KeyError(f"Teacher checkpoint has no 'state_dict': {diff_path}")

        # The frozen transport teacher constructs P_1_dagger from P_0 and P_gt.
        self.partial_enc_t = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.model_t = minknet.MinkUNet_NoTime(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        state_dict = ckpt_diff['state_dict']
        partial_enc_state = {
            k.removeprefix('partial_enc.'): v
            for k, v in state_dict.items()
            if k.startswith('partial_enc.')
        }
        model_state = {
            k.removeprefix('model.'): v
            for k, v in state_dict.items()
            if k.startswith('model.')
        }
        # Strict loading is intentional: silently missing teacher parameters
        # would change the distillation target while training still appears to
        # run normally.
        self.partial_enc_t.load_state_dict(partial_enc_state, strict=True)
        self.model_t.load_state_dict(model_state, strict=True)

        for param in self.partial_enc_t.parameters():
            param.requires_grad = False
        for param in self.model_t.parameters():
            param.requires_grad = False
        self.partial_enc_t.eval()
        self.model_t.eval()

        self.cls = 0
        # Instance normalization reduces sensitivity to heterogeneous condition
        # combinations within a training batch.
        self.partial_enc = minknetin.MinkGlobalEncIN(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.model = minknetin.MinkUNetDiffIN(in_channels=3, out_channels=self.hparams['model']['out_dim'])

        self.processor = genimg.BEVDataProcessor(
            max_density=50.0,
            min_z=-4.0,
            max_z=5.4,
            grid_size=256,
            pc_range=50.0
        )

        self.cnt = 0

    def train(self, mode=True):
        """Set student mode while keeping the distillation teacher in eval mode.

        ``requires_grad=False`` freezes parameters but does not freeze batch
        normalization running statistics. Lightning calls ``train()`` on the
        complete module tree, so the teacher must explicitly be returned to
        evaluation mode every time.
        """
        super().train(mode)
        self.partial_enc_t.eval()
        self.model_t.eval()
        return self

    def fm_sample(self, x_0, t, x_1):
        r"""Evaluate :math:`\mathcal{P}_t=(1-t)\mathcal{P}_0+t\mathcal{P}_1^\dagger`."""
        t = t.view(-1, 1, 1)
        x_t = (1 - t) * x_0 + t * x_1
        return x_t

    def p_losses(self, y, noise):
        return F.mse_loss(y, noise)

    def teacher_forward(self, x_full, x_full_sparse, x_part, t):
        """Evaluate the frozen transport teacher without state updates.

        The retained teacher backbone returns the historical denoising-sign
        residual. Its negative is the paper's source-to-endpoint displacement
        :math:`\Gamma_\eta`.
        """
        with torch.no_grad():
            self.partial_enc_t.eval()
            self.model_t.eval()
            part_feat = self.partial_enc_t(x_part)
            out = self.model_t(x_full, x_full_sparse, part_feat)
            torch.cuda.empty_cache()
            return out.reshape(t.shape[0],-1,3).detach()
    def student_forward(self, img_main, layout_mask, x_full, x_full_sparse, x_part, t):
        r"""Predict :math:`v_\psi(\mathcal{P}_t,t,\bar{B},C_m)`."""
        part_feat = self.partial_enc(x_part)
        out = self.model(img_main, layout_mask, x_full, x_full_sparse, part_feat, t)
        torch.cuda.empty_cache()

        return out.reshape(t.shape[0], -1, 3+self.cls)

    def points_to_tensor(self, x_joint):
        """Build a sparse TensorField while retaining metric XYZ as features.

        MinkowskiEngine coordinates have layout ``[batch, voxel_x, voxel_y,
        voxel_z]``. Features remain the unquantized metric XYZ values; only the
        coordinates used for sparse hashing are divided by ``resolution``.
        Duplicate points in a voxel are averaged by ``UNWEIGHTED_AVERAGE``.
        """
        batched_joint = ME.utils.batched_coordinates(list(x_joint[:]), dtype=torch.float32, device=self.device)

        x_coord = batched_joint[:, :4].clone()

        x_coord[:, 1:] = torch.round(batched_joint[:, 1:4] / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=batched_joint[:, 1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t
    def _shared_step(self, batch: dict, metric_prefix: str):
        r"""Compute the point-flow objective :math:`\mathcal{L}_{Point}`."""
        torch.cuda.empty_cache()
        B = batch['pcd_full'].shape[0]
        # Sample P_0 = R(B_bar; N, Sigma) from the ground-truth prior and draw
        # an independent perturbation for the coupling diagnostic.
        source_anchors, point_source, independent_source = bev_resample(
            batch['pcd_full'], target_pts=batch['pcd_full'].shape[1]
        )

        x_full = self.points_to_tensor(point_source)
        x_part_teacher = self.points_to_tensor(batch['pcd_full'])

        t_min = 0.001
        # Avoid the exactly-zero endpoint where voxel collisions can make the
        # sparse representation especially degenerate; the sampled interval is
        # otherwise uniform and spans the complete transport path.
        t = t_min + (1.0 - t_min) * torch.rand(B, device=self.device)
        teacher_network_residual = self.teacher_forward(
            x_full, x_full.sparse(), x_part_teacher, t
        )
        source_to_endpoint = -teacher_network_residual
        teacher_endpoint = point_source + source_to_endpoint

        # Hybrid coupling preserves the source marginal while controlling its
        # correlation with the teacher pair. The reported model uses beta=0.
        beta = 0.0
        aligned_noise = point_source - source_anchors
        independent_noise = independent_source - source_anchors
        hybrid_noise = (
            math.sqrt(1.0 - beta) * aligned_noise
            + math.sqrt(beta) * independent_noise
        )
        coupled_source = source_anchors + hybrid_noise

        # u_P = P_1_dagger - P_0 is the velocity target along the linear path.
        target_velocity = teacher_endpoint - coupled_source
        point_state_t = self.fm_sample(coupled_source, t, teacher_endpoint)
        x_full = self.points_to_tensor(point_state_t)

        # Uniformly sample C_m from all eight LiDAR/vehicle/road mask states.
        probs = torch.full((8,), 0.125, device=self.device)
        state_indices = torch.multinomial(
            probs, num_samples=B, replacement=True
        )

        drop_lidar = (state_indices & 4) == 0
        drop_vehicle = (state_indices & 2) == 0
        drop_road = (state_indices & 1) == 0

        pcd_part = batch['pcd_part'].clone()
        if drop_lidar.any():
            pcd_part[drop_lidar] = 0.0
            pcd_part[drop_lidar, 0, :3] = 1.0
        x_part_student = self.points_to_tensor(pcd_part)

        layout_mask = self.processor.get_layout_bev(
            batch['pcd_full'], batch['full_label']
        )
        layout_mask = layout_mask * 2.0 - 1.0

        # C_m stores the vehicle and road/ground layout cues in channels 0 and 1.
        vehicle_channels = [0]
        road_channels = [1]

        drop_vehicle_mask = drop_vehicle.view(B, 1, 1, 1)
        drop_road_mask = drop_road.view(B, 1, 1, 1)

        layout_mask[:, vehicle_channels] = torch.where(
            drop_vehicle_mask,
            torch.zeros_like(layout_mask[:, vehicle_channels]),
            layout_mask[:, vehicle_channels]
        )
        layout_mask[:, road_channels] = torch.where(
            drop_road_mask,
            torch.zeros_like(layout_mask[:, road_channels]),
            layout_mask[:, road_channels]
        )

        # B_bar = Phi(P_gt) supplies density, maximum height, and occupancy-mask
        # context. All three normalized channels condition the point flow.
        img_main = self.processor.points_to_bev_target(batch['pcd_full'])

        denoise_t_student = self.student_forward(
            img_main,
            layout_mask,
            x_full,
            x_full.sparse(),
            x_part_student,
            t
        )

        pred_xyz_vel = denoise_t_student[:, :, :3]

        # Conditional flow matching regresses the source-to-endpoint velocity,
        # not absolute point coordinates.
        loss_mse_xyz = self.p_losses(target_velocity, pred_xyz_vel)
        loss = loss_mse_xyz

        self.log(f'{metric_prefix}/loss', loss, prog_bar=True)
        self.log(f'{metric_prefix}/loss_mse_xyz', loss_mse_xyz, prog_bar=True)

        if metric_prefix == 'train':
            self.cnt += 1
        torch.cuda.empty_cache()
        return loss

    def training_step(self, batch: dict, batch_idx):
        """Train point transport from ``P_0`` at t=0 to ``P_1_dagger`` at t=1."""
        return self._shared_step(batch, metric_prefix='train')

    def validation_step(self, batch:dict, batch_idx):
        return

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
