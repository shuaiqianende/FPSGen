import numpy as np
import pytest

from scripts.research_v2.hier_occ_inference.voxel_guided_resampling import (
    voxel_guided_uniform_resample,
)


BOUNDS = {"minimum": (-2.0, -2.0, -2.0),
          "maximum": (2.0, 2.0, 2.0), "sphere_radius": 2.0}


def test_exact_target_default_count_and_reproducibility():
    points = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
    a, meta = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        target_points=1_080_000, seed=123)
    b, _ = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        target_points=1_080_000, seed=123)
    assert a.shape == (1_080_000, 3)
    assert a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(a, b)
    assert meta["actual_output_points"] == 1_080_000
    assert meta["min_points_per_voxel"] == meta["max_points_per_voxel"] == 1_080_000


def test_different_seed_changes_points_but_keeps_occupied_voxel():
    points = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
    a, _ = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        voxel_size=0.5, target_points=64, seed=1)
    b, _ = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        voxel_size=0.5, target_points=64, seed=2)
    assert not np.array_equal(a, b)
    assert np.array_equal(np.floor((a + 2.0) / 0.5),
                          np.floor((points + 2.0) / 0.5).repeat(len(a), axis=0))


def test_uniform_samples_are_in_cell_and_roi():
    points = np.array([[1.8, 0.0, 0.0], [1.5, 0.2, 0.0]], dtype=np.float32)
    out, meta = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        voxel_size=0.5, target_points=4096, seed=77)
    source_cells = np.floor((points - np.array(BOUNDS["minimum"])) / 0.5).astype(int)
    output_cells = np.floor((out - np.array(BOUNDS["minimum"])) / 0.5).astype(int)
    assert set(map(tuple, output_cells)) <= set(map(tuple, source_cells))
    assert np.linalg.norm(out, axis=1).max() < 2.0
    assert np.all(out >= np.asarray(BOUNDS["minimum"]))
    assert np.all(out < np.asarray(BOUNDS["maximum"]))
    assert meta["voxel_coverage_ratio"] == 1.0


def test_uniform_distribution_inside_unclipped_voxel():
    points = np.array([[0.2, 0.2, 0.2]], dtype=np.float32)
    bounds = {"minimum": (-2.0, -2.0, -2.0), "maximum": (2.0, 2.0, 2.0)}
    out, _ = voxel_guided_uniform_resample(points, bounds=bounds,
        voxel_size=0.5, target_points=30_000, seed=1234)
    # The occupied cell is [0, .5)^3, whose uniform mean is (.25,.25,.25).
    assert np.allclose(out.mean(axis=0), 0.25, atol=0.004)
    assert np.all((out >= 0.0) & (out < 0.5))


def test_patchscene_ceil_count_policy():
    points = np.array([[-0.8, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=np.float32)
    out, meta = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        voxel_size=0.5, target_points=5, count_policy="patchscene_ceil", seed=4)
    assert meta["occupied_resample_voxels"] == 2
    assert out.shape == (6, 3)
    assert meta["actual_output_points"] == 6
    assert meta["min_points_per_voxel"] == meta["max_points_per_voxel"] == 3


def test_exact_target_below_voxel_count_reports_drops():
    points = np.array([[-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0],
                       [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    out, meta = voxel_guided_uniform_resample(points, bounds=BOUNDS,
        voxel_size=0.25, target_points=2, seed=99)
    assert out.shape == (2, 3)
    assert meta["sampled_voxels"] == 2
    assert meta["dropped_by_point_budget_voxels"] == 2
    assert meta["dropped_occupied_voxels"] == 2
    assert meta["voxel_coverage_ratio"] == 0.5


def test_empty_and_invalid_inputs_fail_loudly():
    with pytest.raises(ValueError, match="empty prediction"):
        voxel_guided_uniform_resample(np.empty((0, 3), np.float32), bounds=BOUNDS)
    with pytest.raises(ValueError, match="NaN or Inf"):
        voxel_guided_uniform_resample(np.array([[np.nan, 0, 0]]), bounds=BOUNDS)
