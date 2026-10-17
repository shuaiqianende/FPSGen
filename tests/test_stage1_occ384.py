"""CPU math and field indexing contracts for 384-grid occupancy."""
import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_occupancy import (
    ids_to_dense_occupancy, occupancy_flow_pair, masked_occ_loss)


def test_384_8_geometry_and_flat_field():
    g = SparseHierarchyGeometry(parent_resolution=(384,384,32), child_split=(2,2,2))
    assert g.fine_resolution == (768,768,64)
    assert np.allclose(g.fine_size, (100/768,100/768,8.4/64))
    ids = np.array([0,1,383,384,384*384,384*384*32-1],dtype=np.uint32)
    f = ids_to_dense_occupancy(ids,g)
    assert f.shape == (1,1,32,384,384)
    assert torch.equal(f.view(-1)[torch.tensor(ids.astype(np.int64),dtype=torch.long)], torch.ones(len(ids)))
    assert f.sum().item() == len(ids)


def test_unbiased_gaussian_bridge_is_exact():
    gen = torch.Generator().manual_seed(9)
    target = torch.tensor([[[[[0.,1.,0.,1.]]]]])
    valid = torch.tensor([[[[[True,True,True,False]]]]])
    with pytest.raises(ValueError):
        occupancy_flow_pair(target, valid, generator=gen)
    target[...,3] = 0
    pair = occupancy_flow_pair(target,valid,generator=gen,time=torch.tensor([.37]))
    torch.testing.assert_close(pair["state"],pair["source"]*.63+pair["target"]*.37)
    torch.testing.assert_close(pair["velocity"],pair["target"]-pair["source"])
    assert pair["target"][0,0,0,0,0] == -1
    assert pair["target"][0,0,0,0,1] == 1
    assert pair["target"][0,0,0,0,3] == 0


def test_masked_losses_and_invalid_voxels():
    mask = torch.tensor([[[[[True,True,False]]]]])
    p = torch.tensor([[[[[0.,1.,10000.]]]]])
    t = torch.tensor([[[[[1.,2.,-3.]]]]])
    result = masked_occ_loss(p,t,mask,objective="fm")
    assert result.item() == pytest.approx(1.0)
    assert torch.isfinite(masked_occ_loss(p,torch.tensor([[[[[0.,1.,0.]]]]]),mask,objective="bce"))


def test_batch2_independent_scene_times_and_loss_normalization():
    occupied=torch.tensor([[[[[0.,1.,0.]]]],[[[[1.,0.,0.]]]]])
    valid=torch.tensor([[[[[True,True,False]]]]])
    pair=occupancy_flow_pair(occupied,valid,generator=torch.Generator().manual_seed(3),time=torch.tensor([.2,.8]))
    for i,t in enumerate((.2,.8)):
        torch.testing.assert_close(pair['state'][i],(1-t)*pair['source'][i]+t*pair['target'][i])
    assert pair['time'].shape==(2,) and torch.all(pair['state'][...,2]==0)
    loss=masked_occ_loss(torch.zeros_like(occupied),occupied,valid,objective='fm')
    assert loss.item()==pytest.approx(.5)  # broadcast mask denominator includes both scenes
    repeat=occupancy_flow_pair(occupied,valid,generator=torch.Generator().manual_seed(3),time=torch.tensor([.2,.8]))
    torch.testing.assert_close(pair['source'],repeat['source'])
