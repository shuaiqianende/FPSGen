import numpy as np
import torch

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage2_condition import observed_child_geometry
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel


def test_child_geometry_is_input_only_and_maps_per_child():
    g = SparseHierarchyGeometry(parent_resolution=(384, 384, 32), child_split=(2, 2, 2))
    parent_coord = np.asarray([[120, 160, 10]], dtype=np.int64)
    parent_id = int(g.flatten(parent_coord)[0])
    fine_coords = g.compose(np.repeat(parent_coord, 2, axis=0), np.asarray([0, 7]))
    centers = g.centers(fine_coords, fine=True)
    points = np.stack((centers[0], centers[1], centers[1] + np.asarray([.01, 0., 0.])))
    evidence = observed_child_geometry(points, np.asarray([parent_id]), g)
    assert evidence.shape == (1, 8, 8)
    assert evidence[0, 0, 0] == 1
    assert evidence[0, 7, 0] == 1
    assert evidence[0, 7, 1] > evidence[0, 0, 1]
    assert np.isfinite(evidence).all()
    assert np.count_nonzero(evidence[:, :, 0]) == 2


def test_child_conditioned_decoder_keeps_parent_token_shape():
    g = SparseHierarchyGeometry(parent_resolution=(384, 384, 32), child_split=(2, 2, 2))
    model = Stage2OccupancyModel(g, objective='fm', backbone='local', width=16,
        lidar_channels=384, condition_mode='pyramid_observed', checkpointing=False,
        child_bridge='child_conditioned', child_input='geometry_stats', child_feature_width=8).eval()
    ids = torch.tensor([1000, 2000, 1000, 2000])
    batch = torch.tensor([0, 0, 1, 1])
    state = torch.randn(4, 8)
    child_geometry = torch.randn(4, 8, 8)
    valid = torch.ones((4, 8), dtype=torch.bool)
    with torch.no_grad():
        out = model(ids, batch=batch, state=state, time=torch.tensor([.25, .75]),
            lidar_features=torch.randn(4, 384), parent_score=torch.ones(4),
            observed_child=torch.zeros((4, 8), dtype=torch.bool),
            valid_child_mask=valid, child_geometry=child_geometry)
    assert out.shape == (4, 8)
    assert torch.isfinite(out).all()


def test_child_bridge_requires_explicit_geometry():
    g = SparseHierarchyGeometry(parent_resolution=(384, 384, 32), child_split=(2, 2, 2))
    model = Stage2OccupancyModel(g, objective='fm', backbone='local', width=16,
        lidar_channels=384, checkpointing=False, child_bridge='child_conditioned').eval()
    ids = torch.tensor([1000])
    try:
        model(ids, state=torch.zeros(1, 8), time=torch.tensor([.5]),
            lidar_features=torch.zeros(1, 384), parent_score=torch.ones(1),
            observed_child=torch.zeros(1, 8), valid_child_mask=torch.ones(1, 8, dtype=torch.bool))
    except ValueError as exc:
        assert 'Child-aware bridge requires' in str(exc)
    else:
        raise AssertionError('Child-aware inference silently accepted missing geometry evidence')


def test_position_only_control_ignores_raw_child_statistics():
    g = SparseHierarchyGeometry(parent_resolution=(384, 384, 32), child_split=(2, 2, 2))
    torch.manual_seed(4)
    model = Stage2OccupancyModel(g, objective='fm', backbone='local', width=16,
        lidar_channels=384, checkpointing=False, child_bridge='child_conditioned',
        child_input='position_only', child_feature_width=8).eval()
    ids=torch.tensor([1000]);state=torch.zeros(1,8);valid=torch.ones(1,8,dtype=torch.bool)
    shared={'lidar_features':torch.zeros(1,384),'parent_score':torch.ones(1),
        'observed_child':torch.zeros(1,8,dtype=torch.bool),'valid_child_mask':valid}
    with torch.no_grad():
        a=model(ids,state=state,time=torch.tensor([.5]),child_geometry=torch.zeros(1,8,8),**shared)
        b=model(ids,state=state,time=torch.tensor([.5]),child_geometry=torch.randn(1,8,8),**shared)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
