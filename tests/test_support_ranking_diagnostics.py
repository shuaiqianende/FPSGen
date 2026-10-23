import numpy as np
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.support_ranking_diagnostics import (
    stable_topk, six_neighbors, local_max6, is_calibration, ceiling, oracle_pool_upper_bound)


def test_topk_ties_use_parent_id_and_fixed_cardinality():
    score = np.array([0.,.9,.9,.9,.1])
    assert np.array_equal(stable_topk(score,np.array([4,3,2,1]),2),[1,2])


def test_neighbors_do_not_wrap_boundaries():
    g=SparseHierarchyGeometry(parent_resolution=(4,4,2),child_split=(2,2,2))
    valid=np.ones(g.num_parents,bool)
    result=six_neighbors(np.array([0]),g,valid)
    assert np.array_equal(result,[1,4,16])
    valid[4]=False
    assert np.array_equal(six_neighbors(np.array([0]),g,valid),[1,16])


def test_local_max_matches_six_neighbor_values():
    score=np.arange(32,dtype=float).reshape(2,4,4)
    result=local_max6(score)
    assert result[0,0,0]==16
    assert result[1,3,3]==30


def test_calibration_is_grouped_and_deterministic():
    assert is_calibration('00',200)==is_calibration('00',299)
    assert is_calibration('00',200)==is_calibration('00',200)


def test_oracle_is_only_an_upper_bound_and_counts_fine_mass():
    gt=np.array([1,2,3]);masks=np.array([255,1,3],dtype=np.uint8)
    assert ceiling(np.array([1]),gt,masks)==8/11
    assert oracle_pool_upper_bound(np.array([1,2,3,4]),gt,masks,1)==8/11
