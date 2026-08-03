"""Stage-1 Flexible Condition BEV Flow Prior.

This module implements the paper's BEV projection :math:`\Phi`, conditional
Flow-Matching objective, and Euler sampler. A scene is represented by
:math:`B=[D,H,M]`: log-normalized density, maximum height, and occupancy.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import matplotlib
from PIL import Image

from pytorch_lightning.core.lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

import fpsgen.models.image_flow_net as IFN

def apply_colormap_to_tensor(tensor_grid, colormap_name='turbo'):
    gray_np = tensor_grid.detach().cpu().numpy()

    valid_mask = gray_np > 1e-6
    if valid_mask.any():
        v_min = 0.0
        v_max = np.percentile(gray_np[valid_mask], 99)
        gray_np = np.clip((gray_np - v_min) / (v_max - v_min + 1e-8), 0, 1)

    cmap = matplotlib.colormaps[colormap_name]

    colored_np = cmap(gray_np[0])[..., :3]
    colored_np = np.transpose(colored_np, (2, 0, 1))

    return colored_np


def save_bev_images_train(gt_raw_bev, pred_raw_bev, pred_mask, gt_mask, save_dir, step):
    """Save a qualitative ``[D, H, M]`` prediction/target comparison.

    The first row contains the predicted density, height, and occupancy; the
    second row contains the corresponding ground-truth channels.
    """
    os.makedirs(save_dir, exist_ok=True)
    idx = 0

    with torch.no_grad():
        def normalize_tensor(t):
            t = torch.clamp(t, min=0.0)
            return t / (t.max() + 1e-8)

        p_density = normalize_tensor(pred_raw_bev[idx, 0:1])
        p_depth = normalize_tensor(pred_raw_bev[idx, 1:2])
        p_mask = pred_mask[idx]

        g_density = normalize_tensor(gt_raw_bev[idx, 0:1])
        g_depth = normalize_tensor(gt_raw_bev[idx, 1:2])
        g_mask = gt_mask[idx]

    p_density_color = apply_colormap_to_tensor(p_density, colormap_name='turbo')
    g_density_color = apply_colormap_to_tensor(g_density, colormap_name='turbo')

    p_depth_color = apply_colormap_to_tensor(p_depth, colormap_name='terrain')
    g_depth_color = apply_colormap_to_tensor(g_depth, colormap_name='terrain')

    def to_binary_mask(t):
        t_binary = (t > 0.5).float()
        return t_binary.detach().cpu().numpy().repeat(3, axis=0)

    p_mask_color = to_binary_mask(p_mask)
    g_mask_color = to_binary_mask(g_mask)

    row1 = np.concatenate([p_density_color, p_depth_color, p_mask_color], axis=2)

    row2 = np.concatenate([g_density_color, g_depth_color, g_mask_color], axis=2)

    final_img_np = np.concatenate([row1, row2], axis=1)

    final_img_pil = final_img_np.transpose((1, 2, 0))
    final_img_pil = (final_img_pil * 255.0).astype(np.uint8)
    final_img_pil = Image.fromarray(final_img_pil)

    save_path = os.path.join(save_dir, f"colored_train_step_{int(step)}.png")
    final_img_pil.save(save_path)

class BEVDataProcessor:
    """Implement the scene-to-BEV projection :math:`B=\Phi(\mathcal P)`.

    ``D`` is log-normalized point density, ``H`` is maximum cell height, and
    ``M`` is binary occupancy; all channels are scaled to ``[-1, 1]``. FPSGen
    uses tensor axes ``[x_index, y_index]`` and flattens a cell as
    ``x_index * grid_size + y_index``. This differs from the common image
    convention ``[row=y, column=x]`` and is therefore documented explicitly.
    """
    def __init__(self, max_density=50.0, min_z=-4.0, max_z=5.4, grid_size=256, pc_range=50.0):
        self.max_density = max_density
        self.max_log_density = math.log1p(max_density)
        self.min_z = min_z
        self.max_z = max_z
        self.z_range = max_z - min_z
        self.grid_size = int(grid_size)
        self.pc_range = pc_range

    def points_to_bev_target(self, pcd):
        """Rasterize batched XYZ points into the normalized target ``[D,H,M]``.

        Density is accumulated with ``scatter_add`` and capped at
        ``max_density`` before logarithmic normalization. Height uses
        ``scatter_reduce(..., reduce="amax")`` as required by the maximum-
        height definition. Empty height cells and the empty state of all other
        channels are encoded as ``-1``.

        Args:
            pcd: Metric point clouds with shape ``[B, N, 3]``.

        Returns:
            Normalized BEV targets with shape ``[B, 3, G, G]``.
        """
        B = pcd.shape[0]
        device = pcd.device
        G = self.grid_size
        xy = pcd[:, :, :2].float()
        z = pcd[:, :, 2].float()
        xy_norm = (xy + self.pc_range) / (self.pc_range * 2.0)
        xy_pixels = (xy_norm * G).long()
        xy_pixels = xy_pixels.clamp(0, G - 1)
        flat_indices = xy_pixels[:, :, 0] * G + xy_pixels[:, :, 1]
        density_flat = torch.zeros((B, G * G), dtype=torch.float32, device=device)
        ones = torch.ones_like(flat_indices, dtype=torch.float32)
        density_flat.scatter_add_(1, flat_indices, ones)
        z_max_flat = torch.full((B, G * G), -1000.0, dtype=torch.float32, device=device)
        z_max_flat.scatter_reduce_(1, flat_indices, z, reduce="amax", include_self=False)
        density = density_flat.view(B, 1, G, G)
        z_max = z_max_flat.view(B, 1, G, G)
        density_clamped = density.clamp(0, self.max_density)
        norm_density = (torch.log1p(density_clamped) / self.max_log_density) * 2.0 - 1.0
        mask_bool = (density > 0).float()
        norm_z = (z_max - self.min_z) / self.z_range * 2.0 - 1.0
        norm_z = norm_z * mask_bool + (-1.0) * (1.0 - mask_bool)
        norm_mask = mask_bool * 2.0 - 1.0
        bev_target = torch.cat([norm_density, norm_z, norm_mask], dim=1)
        return bev_target

    def denormalize_predictions(self, generated_bev):
        """Map normalized density/height to visualization-space values.

        The returned height is shifted by ``-min_z`` and is therefore
        non-negative. This helper is used for image visualization, not for
        reconstructing metric XYZ coordinates.
        """
        norm_density = generated_bev[:, 0:1]
        norm_z = generated_bev[:, 1:2]

        log_density = (norm_density + 1.0) / 2.0 * self.max_log_density
        raw_density = torch.expm1(log_density).clamp(min=0, max=self.max_density)

        raw_z = ((norm_z + 1.0) / 2.0) * self.z_range + self.min_z
        raw_z = raw_z.clamp(self.min_z, self.max_z)
        raw_z = raw_z - self.min_z

        return torch.cat([raw_density, raw_z], dim=1)

    def get_layout_bev_kitti360(self, pcd, labels, target_classes=["vehicle", "ground"]):
        """Rasterize KITTI-360 semantics into binary layout conditions.

        Args:
            pcd: Metric XYZ points with shape ``[B, N, 3]``.
            labels: Raw semantic IDs with shape ``[B, N]`` or ``[B, N, 1]``.
            target_classes: Ordered semantic groups to emit. The default
                represents the paper's vehicle and road/ground conditions.

        Returns:
            Binary layout maps with shape ``[B, len(target_classes), G, G]``.
        """
        B = pcd.shape[0]
        G = self.grid_size
        device = pcd.device

        class_label_dict = {
            "vehicle": [26, 27, 28, 29, 30, 31],
            "vehicle_with_motorcycle": [26, 27, 28, 29, 30, 31, 32],
            "ground": [7, 8, 9],
            "road": [7, 8, 9],
            "road_with_terrain": [7, 8, 9, 22],
            "building": [11, 12, 13],
            "vegetation": [21, 22],
            "person": [24, 25],
            "two_wheel": [32, 33],
            "traffic": [19, 20],
        }

        xy = pcd[:, :, :2].float()
        xy_norm = (xy + self.pc_range) / (self.pc_range * 2.0)
        xy_pixels = (xy_norm * G).long().clamp(0, G - 1)
        flat_indices = xy_pixels[:, :, 0] * G + xy_pixels[:, :, 1]

        if labels.dim() == 3:
            labels_sq = labels.squeeze(-1)
        else:
            labels_sq = labels
        labels_sq = labels_sq.to(device)

        ones = torch.ones_like(flat_indices, dtype=torch.float32, device=device)
        channel_layouts = []

        for cls_name in target_classes:
            if cls_name not in class_label_dict:
                raise ValueError(
                    f"Unknown KITTI-360 class group {cls_name!r}; "
                    f"available groups: {list(class_label_dict)}"
                )
            target_tensor = torch.tensor(class_label_dict[cls_name], dtype=labels_sq.dtype, device=device)
            class_mask = torch.isin(labels_sq, target_tensor)
            valid_points = ones * class_mask.float()
            layout_flat = torch.zeros((B, G * G), dtype=torch.float32, device=device)
            layout_flat.scatter_add_(1, flat_indices, valid_points)
            layout_flat = (layout_flat > 0).float()
            channel_layouts.append(layout_flat.view(B, 1, G, G))

        return torch.cat(channel_layouts, dim=1)

    def get_layout_bev(self, pcd, labels, target_classes=['vehicle', 'ground']):
        """Rasterize SemanticKITTI semantics into binary layout conditions.

        Channel order follows ``target_classes`` and defaults to the vehicle
        and road/ground cues used in :math:`C_m`.
        """
        B = pcd.shape[0]
        G = self.grid_size
        device = pcd.device

        class_label_dict = {
            'vehicle': [10, 13, 18, 20, 252, 256, 258, 259],
            'ground': [40, 44, 48, 49],
            'building': [50],
            'vegetation': [70, 71, 72]
        }

        xy = pcd[:, :, :2].float()
        xy_norm = (xy + self.pc_range) / (self.pc_range * 2.0)
        xy_pixels = (xy_norm * G).long()
        xy_pixels = xy_pixels.clamp(0, G - 1)
        flat_indices = xy_pixels[:, :, 0] * G + xy_pixels[:, :, 1]

        labels_sq = labels.squeeze(-1)
        ones = torch.ones_like(flat_indices, dtype=torch.float32)

        channel_layouts = []

        for cls_name in target_classes:
            if cls_name not in class_label_dict:
                raise ValueError(
                    f"Unknown SemanticKITTI class group {cls_name!r}; "
                    f"available groups: {list(class_label_dict)}"
                )

            target_labels = class_label_dict[cls_name]
            target_tensor = torch.tensor(target_labels, dtype=labels.dtype, device=device)

            class_mask = torch.isin(labels_sq, target_tensor)
            valid_points = ones * class_mask.float()

            layout_flat = torch.zeros((B, G * G), dtype=torch.float32, device=device)
            layout_flat.scatter_add_(1, flat_indices, valid_points)
            layout_flat = (layout_flat > 0).float()

            channel_layouts.append(layout_flat.view(B, 1, G, G))

        multi_channel_layout = torch.cat(channel_layouts, dim=1)

        return multi_channel_layout

    def get_bev_semantic_label(self, pcd, labels):
        """Build an optional 20-class BEV semantic visualization target.

        Points are sorted by height before scatter so the highest point in a
        cell determines its SemanticKITTI class index.
        """
        B = pcd.shape[0]
        G = self.grid_size
        device = pcd.device

        class_mapping = {
            1: [10, 252],
            2: [11],
            3: [15],
            4: [18, 258],
            5: [13, 16, 20, 256, 257, 259],
            6: [30, 254],
            7: [31, 253],
            8: [32, 255],
            9: [40, 60],
            10: [44],
            11: [48],
            12: [49],
            13: [50],
            14: [51],
            15: [70],
            16: [71],
            17: [72],
            18: [80],
            19: [81],
        }

        labels_sq = labels.squeeze(-1).long()

        lut = torch.zeros(300, dtype=torch.long, device=device)
        for target_idx, raw_labels in class_mapping.items():
            lut[torch.tensor(raw_labels, dtype=torch.long, device=device)] = target_idx

        mapped_labels = lut[labels_sq]

        xy = pcd[:, :, :2].float()
        z = pcd[:, :, 2].float()

        xy_norm = (xy + self.pc_range) / (self.pc_range * 2.0)
        xy_pixels = (xy_norm * G).long()
        xy_pixels = xy_pixels.clamp(0, G - 1)

        flat_indices = xy_pixels[:, :, 0] * G + xy_pixels[:, :, 1]

        sorted_indices = torch.argsort(z, dim=1)

        flat_indices_sorted = torch.gather(flat_indices, 1, sorted_indices)
        mapped_labels_sorted = torch.gather(mapped_labels, 1, sorted_indices).float()

        bev_semantic_flat = torch.zeros((B, G * G), dtype=torch.float32, device=device)
        bev_semantic_flat.scatter_(1, flat_indices_sorted, mapped_labels_sorted)

        bev_semantic = bev_semantic_flat.view(B, G, G).long()

        return bev_semantic

    def map_point_labels(self, labels):
        """Map raw SemanticKITTI IDs to the compact ``0..19`` class space.

        Args:
            labels: Raw IDs with shape ``[B, N]`` or ``[B, N, 1]``.

        Returns:
            Mapped class indices with shape ``[B, N]``.
        """
        device = labels.device
        class_mapping = {
            1: [10, 252],
            2: [11],
            3: [15],
            4: [18, 258],
            5: [13, 16, 20, 256, 257, 259],
            6: [30, 254],
            7: [31, 253],
            8: [32, 255],
            9: [40, 60],
            10: [44],
            11: [48],
            12: [49],
            13: [50],
            14: [51],
            15: [70],
            16: [71],
            17: [72],
            18: [80],
            19: [81],
        }

        labels_sq = labels.squeeze(-1).long()

        lut = torch.zeros(300, dtype=torch.long, device=device)
        for target_idx, raw_labels in class_mapping.items():
            lut[torch.tensor(raw_labels, dtype=torch.long, device=device)] = target_idx

        mapped_labels = lut[labels_sq]

        return mapped_labels


class FlowIMG(LightningModule):
    """Learn the Flexible Condition BEV Flow velocity :math:`v_\phi`.

    For :math:`B_0\sim\mathcal N(0,I)` and target
    :math:`B_1=\Phi(\mathcal P^{gt})`, training samples the linear path
    :math:`B_\tau=(1-\tau)B_0+\tau B_1` and regresses the constant velocity
    :math:`u_B=B_1-B_0`. The historical class name is kept for checkpoint
    compatibility.
    """
    def __init__(self, hparams: dict, data_module=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module

        self.processor = BEVDataProcessor(
            max_density=50.0,
            min_z=-4.0,
            max_z=5.4,
            grid_size=256,
            pc_range=50.0
        )

        self.model = IFN.BEVFlowTransNet(
            base_ch=32,
            time_dim=256,
            cls=0,
            layout_ch=2
        )

        self.cnt = 0

    def _shared_step(self, batch: dict, metric_prefix: str):
        """Compute :math:`\mathcal L_{BEV}` for training or smoke testing.

        Each sample independently draws one of the eight active condition
        tuples :math:`C_m=(m_lc_l,m_vc_v,m_rc_r)` with equal probability.
        An inactive condition is represented by an all-zero tensor.
        """
        gt_points = batch['pcd_full']
        B = gt_points.shape[0]

        probs = torch.tensor([
            0.125,
            0.125,
            0.125,
            0.125,
            0.125,
            0.125,
            0.125,
            0.125
        ], device=self.device)

        # Sample all eight LiDAR/vehicle/road conditioning combinations uniformly.
        state_indices = torch.multinomial(probs, num_samples=B, replacement=True)

        drop_lidar = ~((state_indices & 4) > 0)
        drop_vehicle = ~((state_indices & 2) > 0)
        drop_road = ~((state_indices & 1) > 0)

        drop_lidar_mask = drop_lidar.view(B, 1, 1, 1)
        drop_vehicle_mask = drop_vehicle.view(B, 1, 1, 1)
        drop_road_mask = drop_road.view(B, 1, 1, 1)

        layout_mask = self.processor.get_layout_bev(batch['pcd_full'], batch['full_label'])
        layout_mask = layout_mask * 2.0 - 1.0

        vehicle_channels = [0]
        road_channels = [1]

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

        cond_points = batch['pcd_part']
        raw_pc_cond = self.model.get_raw_pc_bev(cond_points)


        if self.training:
            raw_pc_cond = torch.where(drop_lidar_mask, torch.zeros_like(raw_pc_cond), raw_pc_cond)

        x1 = self.processor.points_to_bev_target(gt_points)

        t = torch.rand((B,), device=self.device)
        t_expand = t.view(B, 1, 1, 1)

        # B_tau = (1-tau)B_0 + tau B_1; u_B = B_1 - B_0.
        x0 = torch.randn_like(x1)
        xt = (1 - t_expand) * x0 + t_expand * x1
        ut = x1 - x0

        out = self.model(xt, t, raw_pc_cond, layout_mask)

        w = torch.tensor([1.0, 2.0, 1.0], device=out.device).view(1, -1, 1, 1)
        loss_mse = (F.mse_loss(out, ut, reduction='none') * w).mean()

        loss = loss_mse

        self.log(f'{metric_prefix}/loss_mse', loss_mse, prog_bar=True)
        self.log(f'{metric_prefix}/loss', loss, prog_bar=True)

        if (
            metric_prefix == 'train'
            and self.global_step % 100 == 0
            and (self.global_rank == 0 or self.trainer.is_global_zero)
        ):
            with torch.no_grad():
                pred_x1_norm = xt + (1.0 - t_expand) * out

                pred_density_depth = pred_x1_norm[:, 0:2, :, :]
                pred_mask = pred_x1_norm[:, 2:3, :, :]

                pred_raw_bev = self.processor.denormalize_predictions(pred_density_depth)
                gt_raw_bev = self.processor.denormalize_predictions(x1[:, 0:2, :, :])
                gt_mask = x1[:, 2:3, :, :]

                save_dir = os.environ.get("FPSGEN_OUTPUT_DIR", "outputs")
                print(f'\n[Vis] Step: {self.global_step}, t: {t[0].item():.4f}')

                save_bev_images_train(
                    gt_raw_bev=gt_raw_bev,
                    pred_raw_bev=pred_raw_bev,
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    save_dir=save_dir,
                    step=(self.global_step // 100) % 10
                )

        if metric_prefix == 'train':
            self.cnt += 1
        return loss

    def training_step(self, batch: dict, batch_idx):
        """Train a velocity field from Gaussian BEV noise (t=0) to target BEV (t=1)."""
        return self._shared_step(batch, metric_prefix='train')

    @torch.no_grad()
    def p_sample_loop(self, cond_points, layout_mask, steps=25):
        """Integrate :math:`v_\phi` from :math:`B_0` with forward Euler."""
        if steps < 1:
            raise ValueError("steps must be positive")
        B = cond_points.shape[0]
        # Density, height, and occupancy are all generated; using two channels
        # here is incompatible with BEVFlowTransNet's three-channel input.
        x = torch.randn((B, 3, self.processor.grid_size, self.processor.grid_size), device=self.device)
        dt = 1.0 / steps

        for i in tqdm(range(steps), desc="ODE Sampling", leave=False):
            t = torch.ones(B, device=self.device) * (i / steps)
            v_pred = self.model(x, t, cond_points, layout_mask)


            x = x + v_pred * dt

        return x

    def validation_step(self, batch: dict, batch_idx):
        return

    def test_step(self, batch: dict, batch_idx):
        with torch.no_grad():
            return self._shared_step(batch, metric_prefix='test')

    def configure_optimizers(self):
        """Use AdamW with linear warmup followed by cosine learning-rate decay."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['train']['lr'],
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = 1000

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler_dict = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [scheduler_dict]
