"""Sphere geometry, GT-free sources, real shortcut math and raw-field losses."""

import inspect
import torch
from torch import nn
import pytest
from fpsgen.voxel_shortcut.geometry import ShortcutGeometry
from fpsgen.voxel_shortcut.count_mapping import CountMapping,normalize_density,InvalidDensityError
from fpsgen.voxel_shortcut.source_builder import build_source
from fpsgen.voxel_shortcut.shortcut_flow import EMA,linear_pair,bootstrap_target,rollout,path_diagnostics
from fpsgen.voxel_shortcut.shortcut_losses import masked_mse,surface_loss,sum_pool
from fpsgen.voxel_shortcut.shortcut_unet import ShortcutUNet
from fpsgen.voxel_shortcut.count_allocator import allocate_anchors
from fpsgen.voxel_shortcut.dataset import validate_gt_points,validate_inference_frames
from fpsgen.voxel_point.dataset import load_xyz
from fpsgen.voxel_shortcut.checkpointing import checkpoint_once_batchnorm


def test_geometry_axes_roundtrip_and_sphere_boundaries():
    g=ShortcutGeometry(nx=16,ny=8,nz=4)
    x=torch.arange(g.num_voxels).reshape(g.shape)
    assert g.to_field(x).shape==(4,8,16)
    assert torch.equal(g.to_xyz(g.to_field(x)),x)
    xyz=torch.tensor([[0.,0.,-4.],[0.,0.,4.4],[49.99,0.,4.],[49.9,0.,0.]])
    assert g.valid_points(xyz).tolist()==[True,False,False,True]
    ids,valid=g.voxelize_points(xyz,strict=False)
    assert torch.equal(g.flatten_voxel_ids(g.unflatten_voxel_ids(ids)),ids)


def test_boundary_intersection_is_not_center_mask():
    g=ShortcutGeometry(nx=256,ny=256,nz=32)
    ids=torch.arange(g.num_voxels)
    centers=g.voxel_centers(ids)
    valid=g.dense_to_flat(g.valid_circle_mask())
    assert bool((valid & (centers.double().square().sum(-1)>=2500)).any())
    assert int(valid.sum())<=int(g.representation_mask().sum())


def test_mapping_zero_high_counts_no_silent_saturation():
    m=CountMapping(20)
    n=torch.tensor([0.,1.,20.,50.,1000.])
    x=m.encode(n)
    assert x[0]==-1 and x[3]>1
    assert torch.allclose(m.decode(x),n,rtol=2e-6,atol=2e-5)
    count,diag=m.decode(torch.tensor([10000.]),return_diagnostics=True)
    assert torch.isfinite(count).all() and diag['decode_exponent_capped_fraction']==1


@pytest.mark.parametrize('bad',[0.,float('nan'),float('inf'),-1.])
def test_invalid_density_rejected(bad):
    with pytest.raises(InvalidDensityError):
        normalize_density(torch.full((1,1,8,8,8),bad),torch.ones((1,1,8,8,8),dtype=torch.bool))


def test_source_seed_noise_and_input_only_signature():
    shape=(1,1,8,8,8);n=torch.ones(shape);p=torch.ones(shape);known=torch.zeros(shape,dtype=torch.bool)
    known[:,:,:4]=True;valid=torch.ones_like(known);m=CountMapping(20)
    def source(seed):
        return build_source(n,p,known,valid,m,generator=torch.Generator().manual_seed(seed))[0]
    a,b,c=source(1),source(1),source(2)
    assert torch.equal(a,b) and not torch.equal(a,c)
    base=m.encode(normalize_density(n,valid));delta=a-base
    assert delta[~known].std()>3*delta[known].std()
    assert not any('gt' in name.lower() for name in inspect.signature(build_source).parameters)


class Constant(nn.Module):
    def __init__(self):
        super().__init__();self.velocity=nn.Parameter(torch.tensor(2.))
    def forward(self,x,t,h,condition,**kwargs):
        return torch.ones_like(x)*self.velocity


def test_constant_velocity_nfe_bootstrap_ema_detached():
    net=Constant();ema=EMA(net,decay=.5);x=torch.zeros(1,1,8,8,8)
    for nfe in (1,2,4,8,16):
        endpoint,detail=rollout(net,x,(),nfe=nfe)
        assert torch.equal(endpoint,torch.full_like(x,2.)) and detail['actual_nfe']==nfe
    target=bootstrap_target(ema.model,x,.5,.25,())
    assert torch.equal(target,torch.full_like(x,2.)) and not target.requires_grad
    assert not ema.model.training and all(not p.requires_grad for p in ema.parameters())
    with torch.no_grad():net.velocity.fill_(4.)
    ema.update(net)
    assert ema.model.velocity==3
    with pytest.raises(ValueError):bootstrap_target(ema.model,x,.75,.25,())


def test_flow_sign_endpoints_and_trajectory_diagnostic():
    source=torch.zeros(1,1,8,8,8);target=torch.ones_like(source)*2
    assert torch.equal(linear_pair(source,target,0.)[0],source)
    assert torch.equal(linear_pair(source,target,1.)[0],target)
    assert torch.equal(linear_pair(source,target,.5)[1],target-source)
    assert path_diagnostics([source,.5*target,target])['path_length_ratio']==pytest.approx(1)
    with pytest.raises(ValueError):path_diagnostics([source,target])


def test_sparse_surface_supervision_is_finite_and_differentiable():
    m=CountMapping(20);target=torch.zeros(1,1,8,8,8);target[:,:,3,3,3]=10
    state=torch.full_like(target,-.8,requires_grad=True);valid=torch.ones_like(target,dtype=torch.bool)
    loss,terms=surface_loss(state,target,valid,m)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(state.grad).all() and state.grad.abs().sum()>0
    assert sum_pool(target,2).sum()==target.sum()
    assert masked_mse(target,target,valid)==0


def test_real_unet_time_steps_full_field_shape_and_finite_backward():
    net=ShortcutUNet(widths=(8,16,32,64),condition_channels=4,embed_width=32)
    x=torch.randn(1,1,8,16,16)
    c=[torch.randn(1,4,4,8,8),torch.randn(1,4,2,4,4),torch.randn(1,4,1,2,2)]
    v=net(x,0.,1.,c)
    assert v.shape==x.shape
    masked_mse(v,torch.ones_like(v),torch.ones_like(v,dtype=torch.bool)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    with pytest.raises(ValueError):net(x,.5,1.,c)
    with pytest.raises(ValueError):net(x,0.,0.,c)
    assert net(x,.5,0.,c,instantaneous=True).shape==x.shape


def test_exact180k_spherical_anchor_bridge_and_ordering():
    g=ShortcutGeometry(nx=16,ny=16,nz=8)
    field=torch.zeros(1,1,*g.field_shape);field[:,:,4,8,8]=1
    out=allocate_anchors(field,g,generator=torch.Generator().manual_seed(5))
    assert out['anchors'].shape==(180000,3) and out['counts'].sum()==180000
    assert g.valid_points(out['anchors']).all()
    ids,_=g.voxelize_points(out['anchors'])
    assert torch.equal(ids,out['voxel_ids'])


def test_audited_filter_keeps_actual_target_mass_and_rejects_larger_errors():
    g=ShortcutGeometry(nx=16,ny=16,nz=8)
    gt=torch.zeros(180000,3);gt[0,0]=50.0005
    with pytest.raises(ValueError):validate_gt_points(gt,g)
    kept,report=validate_gt_points(gt,g,policy='filter_audited')
    assert len(kept)==179999 and report['filtered_points']==1
    assert report['gt_mass_repaired'] is False
    assert g.counts(kept,expected_points=len(kept)).sum()==179999
    gt[0,0]=50.01
    with pytest.raises(ValueError):validate_gt_points(gt,g,policy='filter_audited')
    gt[0,0]=float('nan')
    with pytest.raises(ValueError):validate_gt_points(gt,g,policy='filter_audited')
    gt=torch.zeros(180000,3);gt[:33,0]=50.0005
    with pytest.raises(ValueError):validate_gt_points(gt,g,policy='filter_audited')


def test_semantic_fourth_column_never_enters_xyz(tmp_path):
    import numpy as np
    p=tmp_path/'input.npy'
    raw=np.array([[1.,2.,3.,float('nan')],[4.,5.,6.,9999.]],dtype=np.float32)
    np.save(p,raw)
    xyz=load_xyz(p)
    assert xyz.shape==(2,3) and torch.equal(xyz,torch.from_numpy(raw[:,:3]))


def test_inference_schema_blocks_training_context_and_preserves_input_only():
    row={'id':'00_000001','input':'input_/000001.npy'}
    assert validate_inference_frames([row])==[row]
    for forbidden in ('gt','gt_counts','semantic_label','teacher','sinkhorn','condition_features'):
        with pytest.raises(ValueError):validate_inference_frames([{**row,forbidden:'anything'}])
    with pytest.raises(ValueError):validate_inference_frames([row,row])


def test_rollout_keeps_invalid_cells_unchanged():
    x=torch.full((1,1,8,8,8),-1.)
    mask=torch.zeros_like(x,dtype=torch.bool);mask[:,:,2:4]=True
    endpoint,_=rollout(Constant(),x,(),nfe=4,valid=mask)
    assert torch.equal(endpoint[~mask],x[~mask])
    assert torch.equal(endpoint[mask],torch.ones_like(endpoint[mask]))


def test_spherical_boundary_cell_anchor_truncation_is_repeatable():
    g=ShortcutGeometry()
    centers=g.voxel_centers(torch.arange(g.num_voxels))
    mask=g.dense_to_flat(g.valid_circle_mask())
    candidates=torch.nonzero(mask & (centers.double().square().sum(-1)>2500)).flatten()
    chosen=candidates[len(candidates)//2]
    flat=torch.zeros(g.num_voxels);flat[chosen]=1
    field=g.to_field(g.flat_to_dense(flat))[None,None]
    a=allocate_anchors(field,g,generator=torch.Generator().manual_seed(5))
    b=allocate_anchors(field,g,generator=torch.Generator().manual_seed(5))
    assert torch.equal(a['anchors'],b['anchors'])
    assert g.valid_points(a['anchors']).all()
    ids,_=g.voxelize_points(a['anchors'])
    assert torch.equal(ids,a['voxel_ids'])
    assert torch.unique(a['anchors'],dim=0).shape[0]>179000


def test_checkpoint_matches_uncheckpointed_gradients_and_batchnorm_updates():
    import copy
    torch.manual_seed(42)
    ordinary=nn.Sequential(nn.Conv3d(2,4,3,padding=1),nn.BatchNorm3d(4),nn.ReLU())
    checked=copy.deepcopy(ordinary)
    a=torch.randn(1,2,4,8,8,requires_grad=True);b=a.detach().clone().requires_grad_(True)
    normal=ordinary(a);safe=checkpoint_once_batchnorm(checked,b)
    normal.square().mean().backward();safe.square().mean().backward()
    assert torch.allclose(normal,safe) and torch.allclose(a.grad,b.grad)
    for p,q in zip(ordinary.parameters(),checked.parameters()):assert torch.allclose(p.grad,q.grad)
    for name,buffer in ordinary.named_buffers():assert torch.equal(buffer,dict(checked.named_buffers())[name])
    assert checked[1].num_batches_tracked==1
