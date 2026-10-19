"""Real CUDA regression tests; run only with explicitly admitted GPU1."""
import os
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from fpsgen.ops.gpu_kdtree import load_extension,ExactGPUKDTree,chamfer_kdtree

pytestmark=pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES')!='1',
    reason='GPU test requires explicit physical GPU1 authorization and idle admission')


@pytest.fixture(scope='module')
def extension():
    return load_extension('/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/gpu_kdtree_v1')


@pytest.mark.parametrize('n,m',[(1,1),(3,9),(9,65),(31,71),(33,65),(1031,517)])
def test_every_nearest_distance_and_index(extension,n,m):
    rng=np.random.default_rng(n+m)
    s=rng.normal(size=(n,3)).astype('f4');q=rng.normal(size=(m,3)).astype('f4')
    tree=ExactGPUKDTree(torch.from_numpy(s).cuda(),extension=extension,leaf_size=8)
    d2,idx=tree.query(torch.from_numpy(q).cuda())
    ref=cKDTree(s).query(q,eps=0.)[0]
    np.testing.assert_allclose(d2.sqrt().cpu().numpy(),ref,atol=1e-10,rtol=1e-12)
    matched=np.linalg.norm(q.astype('f8')-s[idx.cpu().numpy()].astype('f8'),axis=1)
    np.testing.assert_allclose(matched,ref,atol=1e-10,rtol=1e-12)


def test_duplicates_and_far_queries(extension):
    support=torch.zeros((1031,3),device='cuda');support[::2]=1
    query=torch.tensor([[0,0,0],[.5,.5,.5],[1000,1000,1000]],device='cuda')
    tree=ExactGPUKDTree(support,extension=extension,leaf_size=8)
    d2,_=tree.query(query)
    torch.testing.assert_close(d2,torch.tensor([0,.75,3*999**2],device='cuda',dtype=torch.float64))


def test_original_order_invariance(extension):
    gen=torch.Generator(device='cuda').manual_seed(931)
    s=torch.randn(3017,3,device='cuda',generator=gen)
    q=torch.randn(1111,3,device='cuda',generator=gen)
    permutation=torch.randperm(len(s),device='cuda',generator=gen)
    a,_=ExactGPUKDTree(s,extension=extension).query(q)
    b,_=ExactGPUKDTree(s[permutation],extension=extension).query(q)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_chamfer_euclidean_half_factor_and_ragged_sizes(extension):
    p=torch.tensor([[0.,0.,0.]],device='cuda')
    q=torch.tensor([[3.,0.,0.],[5.,0.,0.]],device='cuda')
    value=chamfer_kdtree(p,q,extension=extension)
    assert value['pred_to_gt_m']==3
    assert value['gt_to_pred_m']==4
    assert value['cd_m']==3.5


def test_empty_support_and_nonfinite_rejected(extension):
    for s in [torch.empty(0,3,device='cuda'),torch.tensor([[float('nan'),0,0]],device='cuda')]:
        with pytest.raises(ValueError):ExactGPUKDTree(s,extension=extension)


def test_metric_adapter_preserves_per_scene_ragged_means(extension):
    from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
    metric=GPUKDTreeMetrics(('cd',))
    p=torch.tensor([[0.,0.,0.]],device='cuda')
    q=torch.tensor([[3.,0.,0.],[5.,0.,0.]],device='cuda')
    rows=metric.evaluate_batch([p,q],[q,q])
    assert rows[0]['cd_m']==3.5
    assert rows[1]['cd_m']==0
