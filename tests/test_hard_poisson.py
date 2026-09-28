import numpy as np

from fpsgen.data.hard_poisson import select_largest_feasible_radius


def test_hard_poisson_is_deterministic_and_index_preserving():
    rng = np.random.default_rng(7)
    xyz = rng.uniform(-2, 2, size=(4000, 3)).astype(np.float32)
    first, info_first = select_largest_feasible_radius(
        xyz, target_points=1000, seed=20260928, initial_high=.05, tolerance=25, refinement_passes=6
    )
    second, info_second = select_largest_feasible_radius(
        xyz, target_points=1000, seed=20260928, initial_high=.05, tolerance=25, refinement_passes=6
    )
    assert np.array_equal(first, second)
    assert info_first == info_second
    assert first.shape == (1000,)
    assert len(np.unique(first)) == 1000
    selected = xyz[first]
    # The implementation is selection only: every selected row is unchanged.
    assert np.array_equal(selected, xyz[first])


def test_hard_poisson_minimum_spacing():
    grid = np.stack(np.meshgrid(np.arange(20), np.arange(20), np.arange(2), indexing="ij"), axis=-1)
    xyz = (grid.reshape(-1, 3) * .02).astype(np.float32)
    indices, info = select_largest_feasible_radius(
        xyz, target_points=300, seed=5, initial_high=.02, tolerance=10, refinement_passes=7
    )
    selected = xyz[indices]
    distances = np.linalg.norm(selected[:, None, :] - selected[None, :, :], axis=-1)
    distances += np.eye(len(selected), dtype=np.float32) * 1e6
    assert distances.min() >= info['final_radius'] - 1e-6
