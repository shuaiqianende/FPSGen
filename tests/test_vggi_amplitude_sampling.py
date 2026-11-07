"""Independent integral checks for signed continuous-field sampling."""
import numpy as np
import pytest
from fpsgen.vggi.field_generation.codec import AmplitudeCodec
from fpsgen.vggi.field_generation.sample import FieldSampler, density


def cell_integrals(a, origin, spacing):
    # B3 knots are at half-integer voxel coordinates. Four-point Gauss on
    # each half-cell integrates the squared cubic (degree six) exactly.
    g, w = np.polynomial.legendre.leggauss(4)
    p = np.concatenate([.25*(g+1), .5+.25*(g+1)])
    w = np.tile(w*.25, 2)
    grid = np.stack(np.meshgrid(p, p, p, indexing='ij'), axis=-1).reshape(-1, 3)
    weights = np.einsum('i,j,k->ijk', w, w, w).ravel()*np.prod(spacing)
    masses = np.zeros(a.shape)
    for cell in np.ndindex(a.shape):
        xyz = origin+(np.asarray(cell)+grid)*spacing
        masses[cell] = density(a, xyz, origin, spacing) @ weights
    return masses


def test_signed_density_integral_and_sampling():
    a = np.random.default_rng(15).normal(size=(3, 4, 2)).astype(np.float32)
    origin = np.array([-2., 1., -.5]); spacing = np.array([.5, .8, .6])
    sampler = FieldSampler(a, origin, spacing)
    codec = AmplitudeCodec(a.shape, origin, spacing, 1.)
    mass = cell_integrals(a, origin, spacing)
    assert mass.sum() == pytest.approx(codec.mass(codec.encode(a)), rel=2e-6)
    xyz, stats = sampler.sample(50000, seed=91, batch_size=16384)
    assert xyz.shape == (50000, 3)
    assert np.all(xyz >= origin) and np.all(xyz < origin+spacing*a.shape)
    assert density(a, xyz[:1000], origin, spacing) == pytest.approx(
        codec.density(a, xyz[:1000]), rel=3e-4, abs=1e-6)
    ids = np.floor((xyz-origin)/spacing).astype(int)
    observed = np.bincount(np.ravel_multi_index(ids.T, a.shape), minlength=a.size)/len(xyz)
    expected = mass.ravel()/mass.sum()
    assert np.max(np.abs(observed-expected)) < .006
    assert stats['acceptance'] == pytest.approx(mass.sum()/stats['envelope_mass'], rel=.04)


def test_sign_invariance_repeatability_and_empty():
    a = np.ones((2, 2, 2), dtype=np.float32)
    x, _ = FieldSampler(a, [0]*3, [1]*3).sample(100, seed=6)
    y, _ = FieldSampler(-a, [0]*3, [1]*3).sample(100, seed=6)
    np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError):
        FieldSampler(a*0, [0]*3, [1]*3)
