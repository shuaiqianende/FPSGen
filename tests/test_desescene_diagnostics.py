import numpy as np
from fpsgen.hier_occ_flow.desescene_diagnostics import parent_noise


def test_shared_parents_have_identical_noise_across_different_supports():
    a=parent_noise(np.array([1,4,7]),8,'00/500',19)
    b=parent_noise(np.array([9,7,1]),8,'00/500',19)
    assert np.array_equal(a[0],b[2]) and np.array_equal(a[2],b[1])
    assert not np.array_equal(a,parent_noise(np.array([1,4,7]),8,'00/500',20))


def test_stateless_gaussian_noise_moments_and_finiteness():
    x=parent_noise(np.arange(100000),8,'diagnostic-test',3)
    assert x.dtype==np.float32 and np.isfinite(x).all()
    assert abs(x.mean())<.005 and abs(x.var()-1)<.01
