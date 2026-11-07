import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from fpsgen.vggi.point_transport.bvh_neighbors import BVHIndex

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('kind',['random','plane','duplicates','outside'])
def test_exact_topk(kind):
    rng=np.random.default_rng(147)
    x=rng.normal(size=(193,3)).astype('float32')
    y=rng.normal(size=(387,3)).astype('float32')
    if kind=='plane':x[:,2]=0;y[:,2]=0
    if kind=='duplicates':x[:]=0;y[:]=0
    if kind=='outside':x+=100
    xt,yt=[torch.tensor(v,device='cuda') for v in (x,y)]
    xi,yi=BVHIndex(xt,16),BVHIndex(yt,16)
    for k in (1,32,64):
        ids,d=yi.query_index(xi,k)
        ids=ids.cpu().numpy();d=d.cpu().numpy()
        expected=cKDTree(y).query(x,k=k)[0].reshape(len(x),k)**2
        np.testing.assert_allclose(d,expected,atol=2e-5,rtol=2e-5)
        np.testing.assert_allclose(d,np.sum((x[:,None,:]-y[ids])**2,axis=2),atol=2e-5,rtol=2e-5)
        assert np.all(np.diff(np.sort(ids,axis=1),axis=1)>0)
        if kind=='duplicates':np.testing.assert_array_equal(ids,np.tile(np.arange(k),(len(x),1)))


def test_tiny_and_contract():
    x=torch.zeros((1,3),device='cuda')
    index=BVHIndex(x)
    ids,d=index.query_index(index,1)
    assert ids.item()==0 and d.item()==0
    with pytest.raises(ValueError):index.query_index(index,32)
    with pytest.raises(ValueError):BVHIndex(x,0)
    with pytest.raises(ValueError):BVHIndex(x+float('nan'))


def test_tree_layout_does_not_limit_neighbors():
    gen=torch.Generator(device='cuda').manual_seed(73)
    x=torch.randn((131,3),generator=gen,device='cuda')
    y=torch.randn((513,3),generator=gen,device='cuda')
    qi=BVHIndex(x,16)
    reference=BVHIndex(y,16).query_index(qi,64)
    permutation=torch.randperm(len(y),generator=gen,device='cuda')
    for leaf in (1,512):
        ids,d=BVHIndex(y[permutation],leaf).query_index(qi,64)
        torch.testing.assert_close(d,reference[1],rtol=0,atol=0)
        torch.testing.assert_close(permutation[ids],reference[0],rtol=0,atol=0)
