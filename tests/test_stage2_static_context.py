import torch
from torch import nn

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry


def _geometry():
    return SparseHierarchyGeometry(
        parent_resolution=(4, 4, 2), child_split=(2, 2, 2),
        origin=(-4.0, -4.0, -2.0), maximum=(4.0, 4.0, 2.0), sphere_radius=8.0,
    )


def test_static_fm_keeps_state_and_time_out_of_ptv3(monkeypatch):
    import fpsgen.hier_occ_flow.ptv3_backbone as backbone

    class RecordingPTv3(nn.Module):
        def __init__(self, width, **kwargs):
            super().__init__()
            self.time_conditioned = kwargs['time_conditioned']
            self.last_input = None

        def forward(self, features, *args):
            self.last_input = features.detach().clone()
            return features

    monkeypatch.setattr(backbone, 'ParentPTv3', RecordingPTv3)
    from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
    model = HierarchicalOccupancyModel(
        _geometry(), objective='fm', backbone='ptv3', width=16, lidar_channels=12,
        condition_mode='pyramid_observed', checkpointing=False,
        context_strategy='static',
    ).eval()
    assert model.trunk.time_conditioned is False
    ids = torch.tensor([1, 4, 9], dtype=torch.long)
    score = torch.tensor([0.2, 0.5, 0.8])
    lidar = torch.randn(3, 12)
    observed = torch.zeros(3, 8)
    valid = torch.ones(3, 8, dtype=torch.bool)
    state_a = torch.zeros(3, 8)
    state_b = torch.ones(3, 8)
    kwargs = dict(parent_score=score, lidar_features=lidar, observed_child=observed,
                  valid_child_mask=valid)
    out_a = model(ids, state=state_a, time=torch.tensor(0.0), **kwargs)
    trunk_input_a = model.trunk.last_input
    out_b = model(ids, state=state_b, time=torch.tensor(0.75), **kwargs)
    assert torch.equal(trunk_input_a, model.trunk.last_input)
    assert out_a.shape == out_b.shape == (3, 8)
    assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()
    assert not torch.equal(out_a, out_b)


def test_static_bce_is_direct_single_pass(monkeypatch):
    import fpsgen.hier_occ_flow.ptv3_backbone as backbone

    class IdentityPTv3(nn.Module):
        def __init__(self, width, **kwargs):
            super().__init__()
            self.time_conditioned = kwargs['time_conditioned']

        def forward(self, features, *args):
            return features

    monkeypatch.setattr(backbone, 'ParentPTv3', IdentityPTv3)
    from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
    model = HierarchicalOccupancyModel(
        _geometry(), objective='bce', backbone='ptv3', width=16, lidar_channels=12,
        condition_mode='pyramid_observed', checkpointing=False,
        context_strategy='static',
    ).eval()
    ids = torch.tensor([1, 4, 9], dtype=torch.long)
    out = model(ids, parent_score=torch.rand(3), lidar_features=torch.randn(3, 12),
                observed_child=torch.zeros(3, 8), valid_child_mask=torch.ones(3, 8, dtype=torch.bool))
    assert model.trunk.time_conditioned is False
    assert out.shape == (3, 8) and torch.isfinite(out).all()
