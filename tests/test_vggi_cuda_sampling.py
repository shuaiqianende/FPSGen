import numpy as np
import pytest
import torch
from fpsgen.vggi.field_generation.sample_cuda import CUDAFieldSampler
from test_vggi_amplitude_sampling import cell_integrals
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

def test_signed_distribution_matches_exact_integrals():
    a=np.random.default_rng(15).normal(size=(3,4,2)).astype(np.float32)
    origin=np.array([-2.,1.,-.5]);spacing=np.array([.5,.8,.6])
    sampler=CUDAFieldSampler(a,origin,spacing);x,stats=sampler.sample(50000,seed=91);x=x.cpu().numpy()
    mass=cell_integrals(a,origin,spacing)
    ids=np.floor((x-origin)/spacing).astype(int)
    observed=np.bincount(np.ravel_multi_index(ids.T,a.shape),minlength=a.size)/len(x)
    assert np.max(np.abs(observed-mass.ravel()/mass.sum()))<.006
    assert stats['acceptance']==pytest.approx(mass.sum()/stats['envelope_mass'],rel=.04)
    again,_=sampler.sample(50000,seed=91)
    np.testing.assert_array_equal(x,again.cpu().numpy())
    negative,_=CUDAFieldSampler(-a,origin,spacing).sample(50000,seed=91)
    np.testing.assert_array_equal(x,negative.cpu().numpy())

def test_hierarchy_sparse_last_block_and_failure():
    a=np.zeros((7,11,13),np.float32);a[-1,-1,-1]=1
    s=CUDAFieldSampler(a,[0,0,0],[1,1,1]);x,_=s.sample(3000)
    assert ((x>=torch.tensor([4.5,8.5,10.5],device='cuda')) & (x<torch.tensor(a.shape,device='cuda'))).all()
    with pytest.raises(ValueError):CUDAFieldSampler(a*0,[0]*3,[1]*3)
    with pytest.raises(RuntimeError):s.sample(1000,max_attempts_per_point=1)


def test_cuda_gram_decode_matches_cpu():
    from fpsgen.vggi.field_generation.codec import AmplitudeCodec
    for shape in [(3,4,2),(11,7,5)]:
        codec=AmplitudeCodec(shape,[0]*3,[.3,.4,.5],2.7)
        a=np.random.default_rng(7).normal(size=shape).astype(np.float32)
        q=codec.encode(a);actual=codec.decode_cuda(torch.tensor(q,device='cuda')).cpu().numpy()
        np.testing.assert_allclose(actual,codec.decode(q),atol=8e-6,rtol=1e-4)
