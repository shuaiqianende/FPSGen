import numpy as np
import pytest
import torch

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.inference_fast import (
    gpu_occupied_fine_coords,
    gpu_observed_children,
    gpu_select_occupancy_candidates,
)
from scripts.research_v2.hier_occ_inference.native_voxel_point_sampling import (
    sample_native_occupied_voxels,
    sample_native_occupied_voxels_cuda,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _geom():
    return SparseHierarchyGeometry(
        parent_resolution=(8, 8, 4), child_split=(2, 2, 2),
        origin=(-4., -4., -2.), maximum=(4., 4., 2.), sphere_radius=5.0,
    )


def test_gpu_candidate_and_observed_child_parity():
    g = _geom()
    rng = np.random.default_rng(11)
    score = rng.random((1, 1, 4, 8, 8), dtype=np.float32)
    cpu_valid = np.zeros(g.num_parents, dtype=bool)
    # CPU reference selector is the registered public implementation.
    from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
    cpu_ids, cpu_scores, _ = select_occupancy_candidates(score[0, 0], g, strategy="threshold", threshold=.5)
    gpu_score = torch.from_numpy(score).cuda()
    ids, scores, _ = gpu_select_occupancy_candidates(gpu_score, g, strategy="threshold", threshold=.5)
    assert np.array_equal(ids.cpu().numpy(), cpu_ids)
    np.testing.assert_array_equal(scores.cpu().numpy(), cpu_scores)

    points = np.array([
        [-3.75, -3.75, -1.75], [-3.74, -3.75, -1.75],
        [0., 0., 0.], [4., 0., 0.], [0., 0., 2.],
    ], dtype=np.float32)
    expected = __import__("fpsgen.hier_occ_flow.condition", fromlist=["observed_children"]).observed_children(
        torch.from_numpy(points).cuda(), cpu_ids, g)
    actual = gpu_observed_children(torch.from_numpy(points).cuda(), ids, g).cpu().numpy()
    np.testing.assert_array_equal(actual, expected)


def test_gpu_occupancy_decode_matches_cpu_coords():
    g = _geom()
    ids = torch.tensor([0, 1, 5, 10, g.num_parents - 1], device="cuda")
    valid = torch.as_tensor(g.valid_children(ids), device="cuda")
    occ = torch.tensor([[1,0,1,0,1,0,1,0]] * len(ids), device="cuda", dtype=torch.bool)
    actual = gpu_occupied_fine_coords(ids, occ, valid, g).cpu().numpy()
    pairs = torch.nonzero(occ & valid).cpu().numpy()
    expected = []
    for row, child in pairs:
        p = g.unflatten(np.array([int(ids[row])]))[0]
        expected.append(g.compose(p[None], np.array([child]))[0])
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.int64).reshape(-1, 3))


def test_cuda_native_jitter_exact_target_and_source_voxel_membership():
    g = _geom()
    coords = np.array([[3, 6, 2], [4, 6, 2], [5, 6, 2], [6, 6, 2]], dtype=np.int32)
    cpu, cpu_meta = sample_native_occupied_voxels(
        coords, g, point_mode="native_fine_jitter", target_points=257, seed=1901)
    out, meta = sample_native_occupied_voxels_cuda(
        coords, g, target_points=257, seed=1901, device="cuda")
    out = out.cpu().numpy()
    assert out.shape == (257, 3)
    assert np.isfinite(out).all()
    assert meta["actual_output_points"] == 257
    assert meta["sampled_occupied_voxels"] == cpu_meta["sampled_occupied_voxels"] == 4
    vox = np.floor((out.astype(np.float64) - np.asarray(g.origin)) / np.asarray(g.fine_size)).astype(np.int64)
    assert set(map(tuple, vox)).issubset(set(map(tuple, coords)))
    per_voxel = np.array([np.all(vox == c, axis=1).sum() for c in coords])
    assert per_voxel.min() > 0 and per_voxel.max() - per_voxel.min() <= 1
    assert ((out > np.asarray(g.origin)) & (out < np.asarray(g.maximum))).all()
    assert (np.square(out.astype(np.float64)).sum(-1) < g.sphere_radius ** 2).all()
    replay, _ = sample_native_occupied_voxels_cuda(coords, g, target_points=257, seed=1901, device="cuda")
    np.testing.assert_array_equal(out, replay.cpu().numpy())
    # A different GPU seed changes locations but retains the same native cells.
    other, _ = sample_native_occupied_voxels_cuda(coords, g, target_points=257, seed=1902, device="cuda")
    assert not np.array_equal(out, other.cpu().numpy())

    # A target smaller than the occupied support must report the exact dropped
    # voxel count and still select a reproducible voxel subset.
    small, small_meta = sample_native_occupied_voxels_cuda(
        coords, g, target_points=2, seed=1903, device="cuda")
    small_np = small.cpu().numpy()
    small_vox = np.floor((small_np.astype(np.float64) - np.asarray(g.origin)) / np.asarray(g.fine_size)).astype(np.int64)
    assert small.shape == (2, 3)
    assert small_meta["sampled_occupied_voxels"] == 2
    assert small_meta["dropped_occupied_voxels"] == 2
    assert len(np.unique(small_vox, axis=0)) == 2
