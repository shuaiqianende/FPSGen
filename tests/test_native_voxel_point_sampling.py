import numpy as np
import pytest

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from scripts.research_v2.hier_occ_inference.native_voxel_point_sampling import (
    sample_native_occupied_voxels,
)


@pytest.fixture
def geometry():
    # Fine cells are 0.1m on all axes for this compact geometry.
    return SparseHierarchyGeometry(
        parent_resolution=(20, 20, 20), child_split=(2, 2, 2),
        origin=(-2.0, -2.0, -2.0), maximum=(2.0, 2.0, 2.0),
        sphere_radius=2.0,
    )


def test_voxel_center_matches_native_geometry_centers(geometry):
    coords = np.asarray([[20, 20, 20], [39, 24, 20]], dtype=np.int64)
    points, meta = sample_native_occupied_voxels(
        coords, geometry, point_mode="voxel_center")
    assert np.array_equal(points, geometry.centers(coords, fine=True).astype(np.float32))
    assert meta["actual_output_points"] == len(coords)
    assert meta["dropped_occupied_voxels"] == 0


def test_native_jitter_exact_count_seed_and_anisotropic_cell_membership(geometry):
    coords = np.asarray([[39, 24, 20]], dtype=np.int64)
    a, meta = sample_native_occupied_voxels(
        coords, geometry, point_mode="native_fine_jitter",
        target_points=8192, seed=773)
    b, _ = sample_native_occupied_voxels(
        coords, geometry, point_mode="native_fine_jitter",
        target_points=8192, seed=773)
    c, _ = sample_native_occupied_voxels(
        coords, geometry, point_mode="native_fine_jitter",
        target_points=8192, seed=774)
    assert a.shape == (8192, 3) and a.dtype == np.float32
    assert np.isfinite(a).all() and np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.array_equal(geometry.voxelize_points(a)[0], np.repeat(coords, len(a), axis=0))
    assert np.linalg.norm(a, axis=1).max() < geometry.sphere_radius
    assert meta["actual_output_points"] == 8192


def test_native_jitter_is_uniform_inside_an_interior_voxel(geometry):
    coords = np.asarray([[20, 20, 20]], dtype=np.int64)  # [0,.1]^3, away from ROI boundary
    points, _ = sample_native_occupied_voxels(
        coords, geometry, point_mode="native_fine_jitter",
        target_points=30000, seed=20261019)
    expected_mean = np.full(3, 0.05)
    expected_variance = np.full(3, 0.1**2 / 12.0)
    assert np.allclose(points.mean(axis=0), expected_mean, atol=8e-4)
    assert np.allclose(points.var(axis=0), expected_variance, rtol=0.04, atol=2e-5)


def test_exact_target_drops_native_voxels_reproducibly(geometry):
    coords = np.asarray([[20, 20, 20], [21, 20, 20], [22, 20, 20], [23, 20, 20]])
    points, meta = sample_native_occupied_voxels(
        coords, geometry, point_mode="native_fine_jitter",
        target_points=2, seed=91)
    assert len(points) == 2
    assert meta["sampled_occupied_voxels"] == 2
    assert meta["dropped_by_point_budget_voxels"] == 2
    assert meta["voxel_coverage_ratio"] == 0.5


def test_near_tangent_roi_voxel_uses_finite_fallback():
    g = SparseHierarchyGeometry(
        parent_resolution=(20, 20, 20), child_split=(2, 2, 2),
        origin=(-2.0, -2.0, -2.0), maximum=(2.0, 2.0, 2.0),
        sphere_radius=float(np.linalg.norm([1.9, 0.6, 0.0]) + 1e-5),
    )
    coords = np.asarray([[39, 26, 20]], dtype=np.int64)
    points, meta = sample_native_occupied_voxels(
        coords, g, point_mode="native_fine_jitter", target_points=1, seed=18)
    assert points.shape == (1, 3) and np.isfinite(points).all()
    assert np.linalg.norm(points[0]) < g.sphere_radius
    assert np.array_equal(g.voxelize_points(points)[0], coords)
    assert meta["roi_hit_and_run_fallback_samples"] == 1


def test_bad_coordinates_and_empty_set_rejected(geometry):
    with pytest.raises(ValueError, match="empty"):
        sample_native_occupied_voxels(np.empty((0, 3), dtype=np.int64), geometry)
    with pytest.raises(TypeError, match="integer"):
        sample_native_occupied_voxels(np.zeros((1, 3), dtype=np.float32), geometry)
    with pytest.raises(ValueError, match="outside"):
        sample_native_occupied_voxels(np.asarray([[80, 0, 0]]), geometry)
