"""Deployment graph: XYZ → density → 180k anchors → PointFlow. No GT imports."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .anchor_sampler import sample_gaussian_anchors
from .count_allocator import allocate_counts
from .point_flow import euler_rollout


def _synchronize(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


@torch.no_grad()
def density_to_source(density, spec, *, seed):
    generator = torch.Generator(device=density.device).manual_seed(seed)
    counts = allocate_counts(density, spec, generator=generator)
    anchors, ids = sample_gaussian_anchors(spec, counts, generator=generator)
    if anchors.shape != (180000, 3) or ids.shape != (180000,):
        raise ValueError("Deployment source must contain 180000 points")
    return counts, anchors, ids


class VoxelPointPipeline:
    def __init__(self, density_model, pointflow, spec, *, steps=8):
        self.density_model = density_model.eval()
        self.pointflow = pointflow.eval()
        self.spec, self.steps = spec, steps

    @torch.no_grad()
    def __call__(self, input_xyz, *, seed):
        if input_xyz.ndim != 2 or input_xyz.shape[1] != 3:
            raise ValueError("Inference accepts XYZ only")
        # Seed the upstream tokenizer/serialization RNG as well as the explicit
        # allocation/anchor generator; inference is reproducible per manifest.
        torch.manual_seed(seed)
        _synchronize(input_xyz)
        start = time.perf_counter()
        density = self.density_model(input_xyz)["pred_density"][0, 0]
        _synchronize(density)
        network_end = time.perf_counter()
        counts, anchors, ids = density_to_source(density, self.spec, seed=seed)
        _synchronize(anchors)
        anchor_end = time.perf_counter()
        points = euler_rollout(self.pointflow, anchors, ids, steps=self.steps)
        _synchronize(points)
        if points.shape != (180000, 3):
            raise ValueError("Final output count changed")
        end = time.perf_counter()
        return {"density": density, "counts": counts, "anchors": anchors,
                "voxel_ids": ids, "final_points": points,
                "timing_ms": {"stage1": 1000 * (network_end-start),
                    "allocation_and_anchors": 1000 * (anchor_end-network_end),
                    "pointflow": 1000 * (end-anchor_end), "total": 1000 * (end-start)}}


def save_inference(directory, result, *, seed, spec, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    array = lambda name: result[name].detach().cpu().numpy()
    np.savez_compressed(directory / "voxel_density.npz", density=array("density"))
    np.savez_compressed(directory / "allocated_counts.npz", counts=array("counts"))
    np.save(directory / "source_anchors.npy", array("anchors").astype(np.float32))
    np.save(directory / "final_points.npy", array("final_points").astype(np.float32))
    manifest = {"seed": seed, "geometry": spec.to_dict(), "point_count": 180000,
                "flat_order": "Z-major X-fastest", "source_order": "voxel_id ascending",
                "stage1_network_evaluations": 1, "timing_ms": result["timing_ms"],
                "provenance": provenance}
    anchors = array("anchors")
    manifest["generated_support_distribution"] = {
        "allocated_nonempty_voxels":int((array("counts") > 0).sum()),
        "allocated_voxel_fraction":float((array("counts") > 0).sum() / spec.num_voxels),
        "density_sum":float(array("density").sum(dtype=np.float64)),
        "anchor_xyz_mean":anchors.mean(axis=0).tolist(),
        "anchor_xyz_std":anchors.std(axis=0).tolist()}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
