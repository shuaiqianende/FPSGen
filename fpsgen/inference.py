"""Unified FPSGen inference for flexible-condition LiDAR scene generation.

Inference first integrates the BEV velocity field :math:`v_\phi` to obtain
the structural prior :math:`\hat B`, samples the PointFlow source
:math:`\mathcal P_0=\mathcal R(\hat B;N,\Sigma)`, and then integrates the
point velocity field :math:`v_\psi` to produce the completed scene. Both
stages use classifier-free guidance (CFG) and forward Euler integration.
"""

import numpy as np
import MinkowskiEngine as ME
import torch
import fpsgen.models.minkunet as minknet
import fpsgen.models.minkunet_refine as minknetin
import fpsgen.models.gen_img as genimg
import open3d as o3d
from pytorch_lightning.core.lightning import LightningModule
import os
import tqdm
from natsort import natsorted
import click
import time
import math

def read_ply_xyz_label(ply_path):
    from plyfile import PlyData
    import numpy as np
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    label = np.asarray(v["label"]).astype(np.float32).reshape(-1, 1)
    return np.concatenate([xyz, label], axis=1)

def load_points_any(path_or_array):
    if isinstance(path_or_array, np.ndarray):
        return path_or_array

    if isinstance(path_or_array, torch.Tensor):
        return path_or_array.detach().cpu().numpy()

    if isinstance(path_or_array, str):
        suffix = os.path.splitext(path_or_array)[-1].lower()

        if suffix == ".npy":
            return np.load(path_or_array)

        elif suffix == ".npz":
            data = np.load(path_or_array)
            return data[data.files[0]]

        elif suffix == ".pcd":
            pcd = o3d.io.read_point_cloud(path_or_array)
            return np.asarray(pcd.points)

        else:
            raise ValueError(f"Unsupported point cloud file type: {suffix}")

    raise TypeError(f"Unsupported input type: {type(path_or_array)}")

class DiffCompletion(LightningModule):
    """Run the paper's Unified Inference pipeline.

    The public class name is retained for checkpoint and script compatibility.
    ``point_ckpt`` is the Stage-3 Point Flow checkpoint and ``bev_ckpt`` is
    the Stage-1 BEV Flow checkpoint.
    """
    def __init__(self, point_ckpt, bev_ckpt, refine_path, denoising_steps, cond_weight):
        super().__init__()
        if not os.path.isfile(point_ckpt):
            raise FileNotFoundError(f"Point Flow checkpoint does not exist: {point_ckpt}")
        if not os.path.isfile(bev_ckpt):
            raise FileNotFoundError(f"BEV Flow checkpoint does not exist: {bev_ckpt}")
        ckpt_diff = torch.load(point_ckpt, map_location='cpu')
        if 'hyper_parameters' not in ckpt_diff or 'state_dict' not in ckpt_diff:
            raise KeyError(
                "Student checkpoint must contain 'hyper_parameters' and 'state_dict'"
            )
        self.save_hyperparameters(ckpt_diff['hyper_parameters'])
        if denoising_steps < 1:
            raise ValueError('denoising_steps must be positive.')
        if cond_weight < 0:
            raise ValueError('cond_weight must be non-negative.')
        # Keep the CLI/test setting available to both BEV and PointFlow CFG.
        # The reference FPSGen inference uses scale 2; evaluation entry points
        # therefore default to 2 while still allowing controlled ablations.
        self.guidance_scale = float(cond_weight)

        self.partial_enc = minknetin.MinkGlobalEncIN(in_channels=3,
                                                     out_channels=self.hparams['model']['out_dim']).cuda()
        self.model = minknetin.MinkUNetDiffIN(in_channels=3, out_channels=self.hparams['model']['out_dim']).cuda()

        partial_enc_state = {}
        model_state = {}

        for k, v in ckpt_diff['state_dict'].items():
            if k.startswith('partial_enc.'):
                partial_enc_state[k.replace('partial_enc.', '')] = v
            elif k.startswith('model.'):
                model_state[k.replace('model.', '')] = v

        self.partial_enc.load_state_dict(partial_enc_state, strict=True)
        self.model.load_state_dict(model_state, strict=True)

        from fpsgen.models.image_flow_net import BEVFlowTransNet
        self.bev_gen_net = BEVFlowTransNet(base_ch=32, time_dim=256, cls=0).cuda()
        ckpt_bev = torch.load(bev_ckpt, map_location='cpu')
        bev_state_dict = {}
        for k, v in ckpt_bev['state_dict'].items():
            # FlowIMG checkpoints store BEVFlowTransNet below ``model.``.
            # ``net.`` is accepted for compatibility with early checkpoints.
            new_key = k.removeprefix('model.').removeprefix('net.')
            bev_state_dict[new_key] = v
        self.bev_gen_net.load_state_dict(bev_state_dict, strict=True)
        self.bev_gen_net.eval()
        for param in self.bev_gen_net.parameters():
            param.requires_grad = False
        print("Successfully loaded BEVFlowTransNet from checkpoint.")

        self.processor = genimg.BEVDataProcessor(
            max_density=50.0,
            min_z=-4.0,
            max_z=5.4,
            grid_size=256,
            pc_range=50.0
        )

        if refine_path and str(refine_path) != "None" and os.path.exists(refine_path):
            self.model_refine = minknet.MinkUNet(in_channels=3, out_channels=3 * 6)
            ckpt_refine = torch.load(refine_path, map_location='cpu')
            refine_state = {}
            for key, value in ckpt_refine['state_dict'].items():
                if key.startswith('model_refine.'):
                    refine_state[key.removeprefix('model_refine.')] = value
                elif key.startswith('model.'):
                    refine_state[key.removeprefix('model.')] = value
            if not refine_state:
                raise KeyError(
                    "Refinement checkpoint contains neither 'model_refine.' "
                    "nor 'model.' parameters"
                )
            # Loading into ``self`` would silently leave ``model_refine``
            # untouched; the extracted weights belong to this submodule.
            self.model_refine.load_state_dict(refine_state, strict=True)
            self.model_refine.eval()
            self.is_refine = True
        else:
            self.is_refine = False

        self.partial_enc.eval()
        self.model.eval()
        self.cuda()

        self.hparams['data']['max_range'] = 50.

        self.min_z = -4.0
        self.max_z = 5.4
        self.z_range = self.max_z - self.min_z
        self.max_density = 50.0
        self.max_log_density = math.log1p(self.max_density)
        self.cnt = 0

    @torch.no_grad()
    def predict_full_bev(self, cond_pc, layout_mask, steps=25, guidance_scale=2.0):
        """Generate :math:`\hat B` by integrating the CFG BEV velocity field.

        Conditional and unconditional batches are concatenated so one network
        evaluation implements ``v_cfg = v_u + s * (v_c - v_u)`` at each step.
        """
        if steps < 1:
            raise ValueError("BEV integration steps must be positive")
        cond_pc = cond_pc.float()
        B = cond_pc.shape[0]
        device = cond_pc.device
        xt = torch.randn((B, 3, 256, 256), device=device)
        uncond_pc = torch.zeros_like(cond_pc)
        uncond_mask = torch.zeros_like(layout_mask)
        max_t = 1.0
        dt = max_t / steps

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        for i in range(steps):
            t = torch.full((B,), i / steps * max_t, device=device, dtype=torch.float32)
            if guidance_scale > 1.0:
                xt_double = torch.cat([xt, xt], dim=0)
                t_double = torch.cat([t, t], dim=0)
                pc_double = torch.cat([uncond_pc, cond_pc], dim=0)
                mask_double = torch.cat([uncond_mask, layout_mask], dim=0)
                v_double = self.bev_gen_net(xt_double, t_double, pc_double, mask_double)
                v_uncond, v_cond = v_double.chunk(2, dim=0)
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_pred = self.bev_gen_net(xt, t, cond_pc, layout_mask)
            xt = xt + v_pred * dt
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(f"[BEV Flow] Euler integration: {curr_time:.2f} ms")
        return xt

    def reconstruct_from_model_output(
        self,
        pred_bev,
        target_pts=180000,
        grid_size=256,
        noise_std=1.0,
        density_to_one=False,
    ):
        """Evaluate the BEV source sampler :math:`\mathcal R(\hat B;N,\Sigma)`.

        Density supplies multinomial weights and occupancy suppresses invalid
        cells. A selected cell is represented by its metric center, assigned
        zero height, and perturbed by independent Gaussian XYZ noise. Samples
        beyond the 50 m evaluation radius are excluded before drawing.

        FPSGen uses the non-image axis order ``[x_index, y_index]``. With a
        row-major flattened BEV, X is therefore ``index // width`` and Y is
        ``index % width``.

        Args:
            pred_bev: Normalized ``[B, 3, H, W]`` tensor ordered as density,
                maximum height, and occupancy.
            target_pts: Number :math:`N` of source points per scene.
            grid_size: Spatial resolution of each BEV axis.
            noise_std: Isotropic standard deviation of :math:`\Sigma`.
            density_to_one: Use uniform weights over occupied cells; intended
                only for controlled ablations.

        Returns:
            A metric XYZ tensor with shape ``[B, target_pts, 3]``.
        """
        if pred_bev.ndim != 4 or pred_bev.shape[1] < 3:
            raise ValueError(
                f"pred_bev must have shape [B, >=3, H, W], got {tuple(pred_bev.shape)}"
            )
        if target_pts < 1 or grid_size < 1:
            raise ValueError("target_pts and grid_size must be positive")
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative")

        B, _, H, W = pred_bev.shape
        if H != grid_size or W != grid_size:
            raise ValueError(
                f"BEV shape {(H, W)} does not match grid_size={grid_size}"
            )
        device = pred_bev.device
        MAX_LOG_DENSITY = self.max_log_density

        pred_norm_density = torch.clamp(pred_bev[:, 0, :, :].reshape(B, -1), -1.0, 1.0)
        log1p_density = (pred_norm_density + 1.0) / 2.0 * MAX_LOG_DENSITY
        raw_density = torch.expm1(log1p_density)

        bev_mask = pred_bev[:, 2, :, :].reshape(B, -1) > 0.0

        if density_to_one:
            raw_density = torch.ones_like(raw_density)

        raw_density[~bev_mask] = 0.0

        x_idx, y_idx = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        x_phys = ((x_idx.float() + 0.5) / H) * 100.0 - 50.0
        y_phys = ((y_idx.float() + 0.5) / W) * 100.0 - 50.0

        dist_map = torch.sqrt(x_phys ** 2 + y_phys ** 2).view(-1).unsqueeze(0).expand(B, -1)

        raw_density[dist_map > 50.0] = 0.0

        weights = raw_density
        empty_batches = torch.nonzero(weights.sum(dim=1) <= 0, as_tuple=False).flatten()
        if empty_batches.numel() > 0:
            raise RuntimeError(
                "Predicted BEV has no valid occupied cells for batch indices "
                f"{empty_batches.tolist()}"
            )

        sampled_indices = torch.multinomial(weights, num_samples=target_pts, replacement=True)

        sampled_x_idx = (sampled_indices // W).float()
        sampled_y_idx = (sampled_indices % W).float()

        # Map integer BEV indices to cell centers, matching the source sampler
        # used to construct teacher and student training pairs.
        sampled_x = ((sampled_x_idx + 0.5) / H) * 100.0 - 50.0
        sampled_y = ((sampled_y_idx + 0.5) / W) * 100.0 - 50.0

        sampled_z = torch.zeros_like(sampled_x)

        noise_x = torch.randn_like(sampled_x) * noise_std
        noise_y = torch.randn_like(sampled_y) * noise_std
        noise_z = torch.randn_like(sampled_z) * noise_std

        new_pcd = torch.stack([sampled_x, sampled_y, sampled_z], dim=-1)
        noise = torch.stack([noise_x, noise_y, noise_z], dim=-1)

        noisy_pcd = new_pcd + noise

        return noisy_pcd

    def points_to_tensor(self, x_joint):
        """Quantize metric XYZ for sparse convolution while preserving features."""
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

    def preprocess_scan(self, scan):
        """Range-filter and deterministically size one partial LiDAR scan.

        The sparse condition contains ``num_points / 10`` unique/downsampled
        points. Repeating it ten times creates the 180k-point PointFlow source
        cardinality without changing the observed geometry.
        """
        scan = np.asarray(scan)
        if scan.ndim != 2 or scan.shape[1] < 3:
            raise ValueError(f"scan must have shape [N, >=3], got {scan.shape}")
        # Ignore intensity/label columns when computing geometric range.
        xyz = scan[:, :3]
        dist = np.sqrt(np.sum(xyz ** 2, -1))
        scan = xyz[(dist < self.hparams['data']['max_range']) & (dist > 3.5)]

        target_num = int(self.hparams['data']['num_points'] / 10)
        current_num = scan.shape[0]
        if current_num == 0:
            raise ValueError(
                "scan has no points in the valid radial range (3.5 m, max_range)"
            )
        print(
            f"[Preprocess] valid input points={current_num}, "
            f"condition points={target_num}"
        )
        if current_num < target_num:
            idx = np.random.choice(current_num, target_num - current_num, replace=True)
            scan = np.concatenate([scan, scan[idx]], axis=0)

        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        pcd_scan = pcd_scan.farthest_point_down_sample(int(self.hparams['data']['num_points'] / 10))
        scan = torch.tensor(np.array(pcd_scan.points)).cuda()

        scan_rpt = scan.repeat(10, 1)
        scan_rpt = scan_rpt[None, :, :]

        return scan_rpt, scan[None, :, :]

    def postprocess_scan(self, completed_scan, input_scan):
        post_scan = completed_scan

        return post_scan

    def complete_scan(self, scan_gt, cond_mode="100", dataset="SemanticKITTI", steps=(10, 4)):
        """Complete one scan using LiDAR/vehicle/road conditioning.

        ``cond_mode`` is a three-bit string ordered as LiDAR, vehicle, road.
        ``steps`` contains ``(bev_euler_steps, point_euler_steps)``.
        """
        if len(cond_mode) != 3 or any(bit not in '01' for bit in cond_mode):
            raise ValueError("cond_mode must be a three-bit string such as '100'")
        if len(steps) != 2 or any(step < 1 for step in steps):
            raise ValueError("steps must contain two positive integers")
        scan, _, bin_path = scan_gt
        img_steps, point_steps = steps
        _, scan = self.preprocess_scan(scan)

        if dataset == "KITTI360":
            gt_path = bin_path.replace("/input/", "/gt/").replace(".ply", ".ply")
            gt_np = read_ply_xyz_label(gt_path)
        elif dataset == "SemanticKITTI":
            if bin_path.endswith(".npy"):
                # Native FPSGen preprocessed layout: sequence/input_/frame.npy
                gt_path = bin_path.replace("/input_/", "/gt_/")
            else:
                # Raw SemanticKITTI layout: sequence/velodyne/frame.bin
                gt_path = bin_path.replace("velodyne", "gt_").replace(".bin", ".npy")
            gt_np = np.load(gt_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        gt_full = torch.tensor(gt_np[:, :4]).unsqueeze(0).float().cuda()
        gt_label = gt_full[:, :, 3:].long()

        scan = scan.float()

        keep_lidar = (cond_mode[0] == '1')
        keep_vehicle = (cond_mode[1] == '1')
        keep_road = (cond_mode[2] == '1')
        print(
            f"[Condition {cond_mode}] "
            f"LiDAR={keep_lidar}, vehicle={keep_vehicle}, road={keep_road}"
        )

        scan_2d = self.bev_gen_net.get_raw_pc_bev(scan)
        if not keep_lidar:
            scan_2d = torch.zeros_like(scan_2d)

        if dataset == "KITTI360":
            layout_mask = self.processor.get_layout_bev_kitti360(gt_full[:, :, :3], gt_label)
        else:
            layout_mask = self.processor.get_layout_bev(gt_full[:, :, :3], gt_label)
        layout_mask = layout_mask * 2.0 - 1.0

        vehicle_channels = [0]
        road_channels = [1]
        if not keep_vehicle:
            layout_mask[:, vehicle_channels] = torch.zeros_like(layout_mask[:, vehicle_channels])
        if not keep_road:
            layout_mask[:, road_channels] = torch.zeros_like(layout_mask[:, road_channels])

        img_gd = 1 if cond_mode == "000" else self.guidance_scale
        pred_bev = self.predict_full_bev(scan_2d, layout_mask, steps=img_steps, guidance_scale=img_gd)
        pred_bev = process_pred_bev(pred_bev)
        recon_pcd = self.reconstruct_from_model_output(
            pred_bev[:, :3, :, :], target_pts=180000, density_to_one=False
        )

        x_full = recon_pcd[:, :, :3]

        x_full = self.points_to_tensor(x_full)

        scan_cond = scan if keep_lidar else torch.zeros_like(scan)
        x_cond = self.points_to_tensor(scan_cond)
        x_uncond = self.points_to_tensor(torch.zeros_like(scan))

        # PointFlow receives the complete structural prior B=[D,H,M]. The
        # occupancy channel remains part of stage-wise BEV-to-sparse fusion
        # after it has been used to sample the PointFlow source.
        img_main = pred_bev

        torch.cuda.empty_cache()

        point_gd = 1 if cond_mode == "000" else self.guidance_scale
        completed_scanwlabel = self.completion_loop_vpred(img_main, layout_mask, x_full, x_cond, x_uncond, step=point_steps,
                                                          guidance_scale=point_gd)

        completed_scan = completed_scanwlabel[:, :3]
        post_scan = self.postprocess_scan(completed_scan, scan)

        self.cnt = self.cnt + 1

        if self.is_refine:
            refine_in = self.points_to_tensor(post_scan[None, :, :])
            offset = self.refine_forward(refine_in).reshape(-1, 6, 3)
            refine_complete_scan = post_scan[:, None, :] + offset.cpu().numpy()
            return post_scan, refine_complete_scan.reshape(-1, 3)

        return post_scan, None

    def refine_forward(self, x_in):
        with torch.no_grad():
            offset = self.model_refine(x_in)

        return offset

    def forward(self, img_main, img_mask, x_full, x_full_sparse, x_part, t):
        """Evaluate the PointFlow velocity network for one conditioning state."""
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(img_main, img_mask, x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out

    def classfree_forward(self, img_main, img_mask, x_t, x_cond, x_uncond, t, guidance_scale=2.0):
        """Evaluate the classifier-free guided PointFlow velocity."""
        x_t_sparse = x_t.sparse()
        if guidance_scale == 1.0:
            return self.forward(img_main, img_mask, x_t, x_t_sparse, x_cond, t)
        img_mask_uncond = torch.zeros_like(img_mask)
        if guidance_scale == 0.0:
            return self.forward(img_main, img_mask_uncond, x_t, x_t_sparse, x_uncond, t)
        noise_uncond = self.forward(img_main, img_mask_uncond, x_t, x_t_sparse, x_uncond, t)
        noise_cond = self.forward(img_main, img_mask, x_t, x_t_sparse, x_cond, t)
        noise_cfg = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return noise_cfg

    @torch.no_grad()
    def completion_loop_vpred(self, img_main, img_mask, x_t, x_cond, x_uncond,
                              step=25, guidance_scale=2.0):
        """Integrate PointFlow from :math:`\mathcal P_0` to the clean endpoint."""
        if step < 1:
            raise ValueError("PointFlow integration steps must be positive")
        B = img_main.shape[0]
        device = x_t.F.device

        # Integrate the PointFlow ODE from the noise endpoint (t=0) to the
        # target endpoint (t=1). This direction matches student training.
        time_edges = torch.linspace(0.0, 1.0, step + 1, device=device)
        my_time = time_edges[:-1]
        dt_list = time_edges[1:] - time_edges[:-1]

        for t_val, current_dt in zip(my_time, dt_list):
            t = t_val.repeat(B)

            pred_out = self.classfree_forward(
                img_main, img_mask, x_t, x_cond, x_uncond, t,
                guidance_scale=guidance_scale
            )

            pred_v = pred_out[:, :3]

            # Forward Euler step for dx/dt = v_theta(x_t, t, condition).
            x_t_coords = x_t.F + pred_v * current_dt
            x_t = self.points_to_tensor(x_t_coords.reshape(B, -1, 3))

        return x_t.F.cpu().detach().numpy()

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1, 4))[:, :3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(
            f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

@click.command()
@click.option('--point-ckpt', required=True, type=click.Path(exists=True),
              help='Stage-3 Point Flow checkpoint')
@click.option('--bev-ckpt', required=True, type=click.Path(exists=True),
              help='Stage-1 BEV Flow checkpoint')
@click.option('--refine', '-r', type=str, default='checkpoints/refine_net.point_cloud',
              help='path to the scan sequence')
@click.option('--denoising_steps', '-T', type=int, default=50, help='number of denoising steps (default: 50)')
@click.option('--cond_weight', '-s', type=float, default=2.0, help='conditioning weight (default: 2.0)')
def main(point_ckpt, bev_ckpt, refine, denoising_steps, cond_weight):
    exp_dir = point_ckpt.split('/')[-1].split('.')[0].replace('=', '') + f'_T{denoising_steps}_s{cond_weight}'

    diff_completion = DiffCompletion(
        point_ckpt, bev_ckpt, refine, denoising_steps, cond_weight
    )

    path = './Datasets/test/'

    os.makedirs(f'./results/{exp_dir}/refine', exist_ok=True)
    os.makedirs(f'./results/{exp_dir}/diff', exist_ok=True)

    for pcd_path in tqdm.tqdm(natsorted(os.listdir(path))):
        pcd_file = os.path.join(path, pcd_path)
        points = load_pcd(pcd_file)

        start = time.time()
        refine_scan, diff_scan = diff_completion.complete_scan(points)
        end = time.time()
        print(f'took: {end - start}s')
        pcd_refine = o3d.geometry.PointCloud()
        pcd_refine.points = o3d.utility.Vector3dVector(refine_scan)
        pcd_refine.estimate_normals()
        o3d.io.write_point_cloud(f'./results/{exp_dir}/refine/{pcd_path.split(".")[0]}.ply', pcd_refine)

        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(diff_scan)
        pcd_diff.estimate_normals()
        o3d.io.write_point_cloud(f'./results/{exp_dir}/diff/{pcd_path.split(".")[0]}.ply', pcd_diff)

def process_pred_bev(pred_bev):
    """Project a generated BEV onto the valid ``[D, H, M]`` representation.

    The occupancy logit is binarized at zero. Empty cells receive normalized
    background value ``-1`` in density and height, whereas occupied cells
    retain the generated values.
    """
    out_bev = pred_bev.clone()

    mask_bool = (out_bev[:, 2:3, :, :] > 0.0).float()

    out_bev[:, 0:1, :, :] = out_bev[:, 0:1, :, :] * mask_bool + (-1.0) * (1.0 - mask_bool)
    out_bev[:, 1:2, :, :] = out_bev[:, 1:2, :, :] * mask_bool + (-1.0) * (1.0 - mask_bool)

    out_bev[:, 2:3, :, :] = mask_bool * 2.0 - 1.0

    return out_bev

if __name__ == '__main__':
    main()
