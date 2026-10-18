"""Exact requested CD/0.5m grid and fixed-source Euler evaluation contracts."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry

path=Path('scripts/research_v2/hier_occ_inference/eval_seq08_combinations.py')
spec=importlib.util.spec_from_file_location('stage2_combo_eval',path)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def test_cd_euclidean_meters_and_half_bidirectional_factor():
    p=np.array([[0.,0.,0.],[1.,0.,0.]])
    g=p+np.array([0.,.2,0.]);r=entry.raw_chamfer(p,g,workers=1)
    assert r['valid_cd'] and r['cd_m']==pytest.approx(.2)
    # Missing far target support must still participate in G->P distance.
    r=entry.raw_chamfer(p[:1],np.array([[0.,0.,0.],[10.,0.,0.]]),workers=1)
    assert r['cd_m']==pytest.approx(2.5)
    assert entry.raw_chamfer(p[:0],g,workers=1)['valid_cd'] is False


def test_half_meter_grid_origin_floor_and_clipped_top_cell():
    g=SparseHierarchyGeometry(parent_resolution=(384,384,32),child_split=(2,2,2))
    xyz=np.array([[0.,0.,0.],[.49,0.,0.],[.5,0.,0.],[-.01,0.,0.],[0.,0.,4.399],[0.,0.,4.4]])
    expected=np.array([100+200*(100+200*8),101+200*(100+200*8),99+200*(100+200*8),100+200*(100+200*16)])
    assert np.array_equal(entry.physical_grid_ids(xyz,g,target_points=True),np.sort(expected))
    partial=np.array([[49.99,0.,4.]])
    assert len(entry.physical_grid_ids(partial,g))==1
    assert len(entry.physical_grid_ids(partial,g,target_points=True))==0


@pytest.mark.parametrize('nfe',[1,2,4])
def test_euler_fixed_source_sign_and_exact_nfe(nfe):
    class Constant(torch.nn.Module):
        def __init__(self):super().__init__();self.times=[]
        def forward(self,ids,*,state,time,**c):self.times.append(float(time));return torch.ones_like(state)*.4
    model=Constant().eval();source=torch.full((3,8),-.3);original=source.clone()
    valid=torch.ones(3,8,dtype=torch.bool);valid[0,0]=False
    occ=entry.integrate(model,torch.arange(3),{'valid_child_mask':valid},source,nfe)
    assert torch.equal(source,original) and torch.equal(occ,valid)
    assert model.times==[i/nfe for i in range(nfe)]


def test_macro_frame_mean_is_not_micro_and_invalid_cd_is_not_silently_dropped():
    rows=[{'stage1_nfe':1,'stage2_nfe':1,'valid_cd':True,'cd_m':.1,'iou05':.5,'tp':1,'fp':0,'fn':1,'predicted_fine_voxels':1},
          {'stage1_nfe':1,'stage2_nfe':1,'valid_cd':True,'cd_m':.3,'iou05':.25,'tp':1,'fp':0,'fn':3,'predicted_fine_voxels':1}]
    r=entry.aggregate(rows)[0]
    assert r['cd_m']==pytest.approx(.2) and r['iou05_macro']==.375 and r['iou05_micro']==pytest.approx(1/3)
    rows[-1].update(valid_cd=False,cd_m=None)
    assert entry.aggregate(rows)[0]['cd_m'] is None


def test_single_four_plus_four_selection_preserves_registered_nfes():
    assert entry.requested_nfes([4])==(4,)
    assert entry.requested_nfes([1,2,4])==(1,2,4)
    for values in ([],[4,4],[8],[0]):
        with pytest.raises(ValueError):entry.requested_nfes(values)
    row={'stage1_nfe':4,'stage2_nfe':4,'valid_cd':True,'cd_m':.2,
         'iou05':.5,'tp':1,'fp':0,'fn':1,'predicted_fine_voxels':1}
    result=entry.aggregate([row])
    assert len(result)==1 and result[0]['stage1_nfe']==result[0]['stage2_nfe']==4
