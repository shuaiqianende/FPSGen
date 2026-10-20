"""GT-free PatchScene Ring_Spread_V2 inference on FPSGen's frozen seq08 frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FPSGEN = HERE.parents[2]
sys.path.insert(0, str(FPSGEN))

from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha


SCENE_BOUNDS = np.asarray([[-50., 50.], [-50., 50.], [-3.75, 3.75]], dtype=np.float32)
NATIVE_VOXEL = 0.15625


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha_array(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(memoryview(arr).cast("B")).hexdigest()


def _make_inference_grid():
    """Reuse official KITTI crop-bound methods without constructing map/GT data."""
    from network.dataloader_ultimate import KITTI_Ultimate_Dataset
    obj = object.__new__(KITTI_Ultimate_Dataset)
    obj.split = "val"
    obj.vol_bnds_scene = SCENE_BOUNDS.copy()
    obj.vol_bnds = np.asarray([[-10., 10.], [-10., 10.], [-3.75, 3.75]], dtype=np.float32)
    obj.vol_origin = obj.vol_bnds_scene[:, 0]
    obj.vol_crop_size = obj.vol_bnds[:, 1] - obj.vol_bnds[:, 0]
    obj.voxel_size = NATIVE_VOXEL
    obj.bnds_list = obj.generate_crop_bnds(obj.vol_bnds_scene, obj.vol_crop_size, [10., 10.])
    return obj


def _infer_frame(generator, grid, xyz, *, steps, device, profile=None):
    from test_semantickitti import PRE_DEFINE, convert_ring_dict, convert_bev_dict_to_world_bounds_centered
    from tools.helpers import insert_local_occu_to_global, points2occu_torch

    # Match the official inference scan filtering and occupancy voxelization.
    phase_started = time.perf_counter()
    points = torch.from_numpy(np.asarray(xyz, dtype=np.float32))
    keep = (points[:, :2].square().sum(dim=1) < 50.0 ** 2)
    points = points[keep]
    x_cond = points2occu_torch(points, grid.vol_bnds_scene, grid.voxel_size).unsqueeze(0)
    if profile is not None:
        profile["input_voxelization_ms"] = (time.perf_counter() - phase_started) * 1000.0

    phase_started = time.perf_counter()
    bev_patch_coords = convert_ring_dict(PRE_DEFINE)
    patch_shape = (128, 128, 48)
    bounds = convert_bev_dict_to_world_bounds_centered(
        bev_patch_coords, patch_shape, grid.voxel_size)

    ring_masks = {}
    mask_occ_dump = torch.ones((1, 128, 128, 48), dtype=x_cond.dtype)
    for ring, ring_bounds in bounds.items():
        ring_mask = torch.zeros_like(x_cond)
        for local_bounds in ring_bounds:
            ring_mask += insert_local_occu_to_global(
                grid.vol_bnds_scene, local_bnds=local_bounds,
                local_occ=mask_occ_dump, voxel_size=grid.voxel_size)
        ring_masks[ring] = ring_mask[..., :1]

    cfg_data = {
        "pc_curr": points, "pc_curr_in_last_occu": None,
        "pred_last": None, "guidance_weight": None, "xs_last": None,
        "pose_last": None, "pose_curr": None,
        "voxel_size": grid.voxel_size, "vol_bnds_scene": grid.vol_bnds_scene,
        "frame": None, "which_ring": None,
    }
    last_ring_xs = {}
    entire_xs = []
    if profile is not None:
        profile["ring_preparation_ms"] = (time.perf_counter() - phase_started) * 1000.0
        profile["number_of_rings"] = len(bounds)
        profile["patch_batch_size_per_ring"] = {k: len(v) for k, v in bounds.items()}
        profile["ring_generation_wall_ms"] = 0.0
    phase_started = time.perf_counter()
    for ring, ring_bounds in bounds.items():
        starts = [torch.tensor(x) for x in bev_patch_coords[ring]]
        cfg_data["which_ring"] = ring_masks[ring]
        cond = cfg_data["pc_curr_in_last_occu"] if cfg_data["pc_curr_in_last_occu"] is not None else x_cond
        xs, _ = generator.genralized_Ring_Spread_CFG(
            x_cond=cond, vol_bnds_list=ring_bounds,
            vol_bnds_start_indices_list=starts, pos_idx_list=None,
            vol_bnds_scene=grid.vol_bnds_scene, voxel_size=grid.voxel_size,
            CFG_Data=cfg_data, last_ring_xs=last_ring_xs,
            manual_batching_size=16, steps=steps, verbose=False)
        if profile is not None:
            profile["ring_generation_wall_ms"] += (time.perf_counter() - phase_started) * 1000.0
            phase_started = time.perf_counter()
        last_ring_xs["xs"] = xs
        if not entire_xs:
            entire_xs = xs
        else:
            for i in range(len(xs)):
                entire_xs[i] = torch.where(entire_xs[i] == 0, xs[i], entire_xs[i])
        last_ring_xs["vol_bnds_list"] = ring_bounds
        last_ring_xs["ring_mask"] = ring_masks[ring]

    phase_started = time.perf_counter()
    prediction = entire_xs[-1][0]
    if not torch.isfinite(prediction).all():
        raise RuntimeError("PatchScene output contains NaN/Inf")
    coords = torch.nonzero(prediction > 0, as_tuple=False).cpu().numpy().astype(np.int32)
    if not len(coords):
        raise RuntimeError("PatchScene produced empty occupancy")
    if profile is not None:
        profile["output_occupancy_ms"] = (time.perf_counter() - phase_started) * 1000.0
    return coords


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patchscene-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="Explicit seq08 frame IDs; default is 0,10,...,4070")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--seed-base", type=int, default=20261019)
    p.add_argument("--physical-gpu", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    if a.physical_gpu not in (0, 1, 2, 3):
        raise ValueError("physical GPU ID must be one of 0,1,2,3")
    if not torch.cuda.is_available():
        raise RuntimeError("PatchScene inference requires CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(a.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must map only the selected physical GPU")
    if a.steps not in (10, 15):
        raise ValueError("Supported PatchScene denoising steps are 10 and 15")

    official_root = a.patchscene_root.resolve()
    sys.path.insert(0, str(official_root))
    # The public inference entry disables these interactive breakpoints. Stub
    # only the debugger module if the isolated runtime does not install ipdb.
    try:
        import ipdb  # noqa: F401
    except ImportError:
        import types
        ipdb = types.ModuleType("ipdb")
        ipdb.set_trace = lambda *args, **kwargs: None
        sys.modules["ipdb"] = ipdb

    from network.model_trainer import DiffusionModel
    from tools.helpers import points2occu_torch  # dependency preflight
    del points2occu_torch

    patch_commit = __import__("subprocess").check_output(
        ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True).strip()
    frame_files = sorted((a.data_root / "08" / "input_").glob("*.npy"), key=lambda q: int(q.stem))
    if len(frame_files) != 4071 or [int(x.stem) for x in frame_files] != list(range(4071)):
        raise RuntimeError("Expected all audited seq08 input_ frames")
    frames = [int(x.stem) for x in frame_files[::10]]
    if a.frames is not None:
        if any(f not in frames for f in a.frames):
            raise ValueError("requested frame is not in frozen seq08/skip10 manifest")
        frames = list(a.frames)
    if a.max_frames is not None:
        frames = frames[:a.max_frames]

    protocol = {
        "status": "RUNNING", "sequence": "08", "stride": 10,
        "frames": frames, "frame_count": len(frames),
        "patchscene_repository": "https://github.com/AFARI-Research/PatchScene",
        "patchscene_commit": patch_commit,
        "checkpoint_sha256": file_sha(a.checkpoint),
        "ema_weights": True, "generate_method": "Ring_Spread_V2",
        "sampling_steps": a.steps, "seed_base": a.seed_base,
        "input_root": str(a.data_root / "08" / "input_"),
        "input_contract": "shared FPSGen 18K XYZ subset from matching SemanticKITTI velodyne scan; uses XYZ only",
        "gt_loaded_during_inference": False,
        "native_grid": [640, 640, 48], "native_voxel_size_xyz": [NATIVE_VOXEL] * 3,
        "native_origin_xyz": SCENE_BOUNDS[:, 0].tolist(),
        "native_maximum_xyz": SCENE_BOUNDS[:, 1].tolist(),
        "native_output": "occupied voxel indices XYZ, before point sampling",
        "temporal_context": "none; official Ring_Spread_V2 uses ring-to-ring state within each frame only",
    }
    a.output.mkdir(parents=True, exist_ok=True)
    protocol_path = a.output / "protocol.json"
    journal = a.output / "per_frame.jsonl"
    if a.resume:
        saved = json.loads(protocol_path.read_text())
        saved["status"] = "RUNNING"
        if saved != protocol:
            raise RuntimeError("Resume protocol identity mismatch")
        rows = [json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else []
    else:
        if any(a.output.iterdir()):
            raise FileExistsError("fresh PatchScene output must be empty")
        atomic_json(protocol_path, protocol)
        rows = []
    done = {int(r["frame"]) for r in rows}

    # load_from_checkpoint uses strict state loading; inference explicitly uses
    # the EMA branch selected by the official README entry point.
    model = DiffusionModel.load_from_checkpoint(
        str(a.checkpoint), map_location="cpu", strict=True).eval()
    generator = model.ema_model.eval().to("cuda:0")
    grid = _make_inference_grid()
    start = time.perf_counter()
    for index, frame in enumerate(frames):
        if frame in done:
            continue
        input_path = a.data_root / "08" / "input_" / f"{frame:06d}.npy"
        xyz = input_xyz(input_path)
        seed = int(a.seed_base + frame)
        _seed(seed)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        coords = _infer_frame(generator, grid, xyz, steps=a.steps, device=torch.device("cuda:0"))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        output = a.output / "predictions" / f"08_{frame:06d}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, occupied_coords_xyz=coords)
        tmp.replace(output)
        row = {
            "sequence": "08", "frame": frame,
            "input_sha256": file_sha(input_path), "input_point_count": int(len(xyz)),
            "input_xyz_sha256": _sha_array(xyz), "prediction_path": str(output),
            "prediction_sha256": file_sha(output), "occupied_voxels": int(len(coords)),
            "seed": seed, "steps": a.steps,
            "inference_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        with journal.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        rows.append(row); done.add(frame)
        atomic_json(a.output / "progress.json", {
            "status": "RUNNING", "completed_frames": len(rows),
            "requested_frames": len(frames), "last_frame": frame,
            "elapsed_seconds": time.perf_counter() - start,
        })
        print(f"{len(rows)}/{len(frames)} frame={frame:04d} occupied={len(coords)} "
              f"seconds={elapsed:.2f} peak={row['peak_allocated_gib']:.2f}GiB", flush=True)
    if len(done) != len(frames):
        raise RuntimeError(f"incomplete inference: {len(done)}/{len(frames)}")
    protocol["status"] = "COMPLETE"
    atomic_json(protocol_path, protocol)
    atomic_json(a.output / "progress.json", {
        "status": "COMPLETE", "completed_frames": len(rows),
        "requested_frames": len(frames), "elapsed_seconds": time.perf_counter() - start,
    })


if __name__ == "__main__":
    main()
