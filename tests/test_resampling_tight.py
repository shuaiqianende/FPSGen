import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/research_v2/hier_occ_inference'))
from voxel_guided_resampling_tight import voxel_guided_uniform_resample


def test_sphere_boundary_and_seed():
    bounds = {'minimum': [-50, -50, -5], 'maximum': [50, 50, 5], 'sphere_radius': 50}
    points = np.array([[49.999, .02, .02], [.02, .02, .02]])
    a, meta = voxel_guided_uniform_resample(points, bounds=bounds, target_points=10000, seed=4)
    b, _ = voxel_guided_uniform_resample(points, bounds=bounds, target_points=10000, seed=4)
    assert np.array_equal(a, b)
    assert meta['sampled_voxels'] == 2
    assert len(a) == 10000
    assert np.all(np.sum(a.astype(np.float64) ** 2, axis=1) < 2500)
    assert np.all(a >= bounds['minimum']) and np.all(a < bounds['maximum'])
    original = np.floor((points - bounds['minimum']) / .15625).astype(int)
    output = np.floor((a.astype(np.float64) - bounds['minimum']) / .15625).astype(int)
    _, counts = np.unique(output, axis=0, return_counts=True)
    assert sorted(counts) == [5000, 5000]
    assert {tuple(x) for x in original} == {tuple(x) for x in output}


def test_interior_uniform_moments():
    points = np.array([[.02, .02, .02]])
    bounds = {'minimum': [0, 0, 0], 'maximum': [1, 1, 1]}
    a, _ = voxel_guided_uniform_resample(points, bounds=bounds, target_points=100000, seed=5)
    u = a / .15625
    assert np.all(np.abs(u.mean(axis=0) - .5) < .005)
    assert np.all(np.abs(u.var(axis=0) - 1 / 12) < .003)
