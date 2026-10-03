import numpy as np

from fpsgen.datasets.dataloader.semantic_kitti import apply_shared_y_flip
from fpsgen.utils.pcd_transforms import random_flip_point_cloud


def test_fast_y_flip_matches_legacy_for_both_rng_outcomes():
    for seed in range(128):
        rng = np.random.default_rng(seed)
        full = rng.normal(size=(101, 4)).astype(np.float32)
        part = rng.normal(size=(29, 4)).astype(np.float32)
        full[:, 3] = rng.integers(0, 20, size=len(full))
        part[:, 3] = rng.integers(0, 20, size=len(part))

        expected = np.concatenate((full[:, :3], part[:, :3]), axis=0)
        np.random.seed(seed)
        expected = random_flip_point_cloud(expected[None, ...])[0]
        actual_full, actual_part = full[:, :3].copy(), part[:, :3].copy()
        np.random.seed(seed)
        flip_y = np.random.random() > 0.5
        apply_shared_y_flip(actual_full, actual_part, flip_y)

        np.testing.assert_array_equal(actual_full, expected[:len(full)])
        np.testing.assert_array_equal(actual_part, expected[len(full):])
        np.testing.assert_array_equal(full[:, 3], full[:, 3].copy())
        np.testing.assert_array_equal(part[:, 3], part[:, 3].copy())
