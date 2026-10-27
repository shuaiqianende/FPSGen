import copy,inspect
import pytest
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.joint_fm_inference import POLICY,validate_policy,predicted_support,input_only_conditions


def test_policy_forbids_gt_neighbors_extra_parents_and_changed_nfe():
    validate_policy(POLICY)
    for key,value in [('gt_support',True),('expansion',1),('extra_parent_fraction',.25),('stage1_nfe',1),('observed_parent_union',True)]:
        p=copy.deepcopy(POLICY);p[key]=value
        with pytest.raises(ValueError):validate_policy(p)


def test_support_is_exact_strict_threshold_no_neighbor_fill():
    g=SparseHierarchyGeometry(parent_resolution=(2,2,2),child_split=(2,2,2));s=torch.tensor([.1,.5,.51,.9,.2,.3,.6,.7]);valid=torch.tensor([1,1,1,1,1,1,0,1],dtype=torch.bool)
    ids,values=predicted_support(s,valid,g)
    assert torch.equal(ids,torch.tensor([2,3,7]))
    assert torch.equal(values,s[ids]) and len(ids)==3


def test_gt_cannot_be_passed_and_empty_support_does_not_fallback():
    g=SparseHierarchyGeometry(parent_resolution=(2,2,2),child_split=(2,2,2));s=torch.zeros(8);v=torch.ones(8,dtype=torch.bool)
    assert 'gt' not in inspect.signature(predicted_support).parameters
    with pytest.raises(TypeError):predicted_support(s,v,g,oracle_ids=torch.tensor([1]))
    with pytest.raises(RuntimeError,match='no GT/neighbor fallback'):predicted_support(s,v,g)


def test_encoder_ablation_only_adds_features_without_changing_ids_scores_or_observed():
    g=SparseHierarchyGeometry(parent_resolution=(2,2,2),child_split=(2,2,2));ids=torch.tensor([0,1,3]);scores=torch.tensor([.51,.7,.8])
    xyz=torch.tensor([[1.,1.,0.],[2.,2.,.1]])
    no=input_only_conditions(ids,scores,xyz,g)
    yes=input_only_conditions(ids,scores,xyz,g,pyramid=tuple(torch.ones(1,128,2,2,2) for _ in range(3)))
    assert 'lidar_features' not in no and yes['lidar_features'].shape==(3,384)
    assert all(torch.equal(no[k],yes[k]) for k in no)
    assert torch.equal(no['parent_score'],scores)


def test_oracle_default_has_no_empty_parent_pool_and_explicit_25_percent_is_rejected(monkeypatch):
    from fpsgen.hier_occ_flow import stage2_candidates as candidates
    import numpy as np
    def forbidden(*a,**k):raise AssertionError('No neighbor enumeration is permitted')
    monkeypatch.setattr(candidates,'expand_parent_ids',forbidden)
    g=SparseHierarchyGeometry(parent_resolution=(2,2,2),child_split=(2,2,2))
    ids,score,proof=candidates.oracle_parent_candidates(np.array([1,3,1]),g,seed=1)
    assert ids.tolist()==[1,3] and proof['nearby_negatives']==0 and proof['nearby_pool_evaluated'] is False
    with pytest.raises(ValueError,match='retired'):
        candidates.oracle_parent_candidates(np.array([1,3]),g,seed=1,negative_fraction=.25)


def test_full_gt_fine_fn_includes_parents_that_stage1_missed():
    import numpy as np
    from fpsgen.hier_occ_flow.evaluation import fine_metrics
    g=SparseHierarchyGeometry(parent_resolution=(4,4,4),child_split=(2,2,2),origin=(-1.,-1.,-1.),maximum=(1.,1.,1.))
    prediction=np.zeros((1,8),dtype=bool);prediction[0,0]=True
    metric=fine_metrics(np.array([0]),prediction,np.array([0,1]),np.array([1,3],dtype=np.uint8),g,structural=False)
    assert metric['fine']['tp']==1 and metric['fine']['fn']==2
    assert metric['missed_parent_fine_false_negatives']==2
    assert metric['fine']['iou']==pytest.approx(1/3)


def test_joint_inference_function_has_no_gt_input_or_gt_loading():
    import ast
    from pathlib import Path
    tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts/research_v2/joint_fm/evaluate.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='infer')
    assert [a.arg for a in fn.args.args]==['xyz','s1','models','g','frame']
    assert not any(isinstance(n,ast.Name) and n.id in ('gt','gt_ids','gt_masks','record','raw') for n in ast.walk(fn))
