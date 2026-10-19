"""Actual CUDA gradient tests for the optional Teacher CD adapter."""
import os
import pytest
import torch
from fpsgen.ops.gpu_kdtree import load_extension
from fpsgen.utils.gpu_kdtree_training import chamfer_sqrt_gpu_kdtree

pytestmark=pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES')!='1',reason='Explicit GPU1 testing only')


@pytest.mark.parametrize('same_points',[False,True])
def test_cd_loss_and_both_coordinate_gradients(same_points):
    extension=load_extension('/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/gpu_kdtree_v1')
    gen=torch.Generator(device='cuda').manual_seed(340)
    p=torch.randn(123,3,device='cuda',generator=gen).requires_grad_()
    g=(p.detach().clone() if same_points else torch.randn(201,3,device='cuda',generator=gen)).requires_grad_()
    actual=chamfer_sqrt_gpu_kdtree(p,g,extension=extension)
    # Small oracle only: squared dense distances select the same discrete NN.
    pair=(p[:,None]-g[None]).square().sum(-1)
    pg=pair.detach().argmin(1);gp=pair.detach().argmin(0)
    expected=.5*((p-g[pg]).norm(dim=-1).mean()+(g-p[gp]).norm(dim=-1).mean())
    ag=torch.autograd.grad(actual,(p,g),retain_graph=True)
    eg=torch.autograd.grad(expected,(p,g))
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    for a,e in zip(ag,eg):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,e,atol=1e-7,rtol=1e-6)
