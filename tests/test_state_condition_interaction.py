import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.state_condition_interaction import StateConditionInteraction, load_shared_baseline


class PointStub(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class FakeBlock(nn.Module):
    def __init__(self, width, checkpointing):
        super().__init__()
        self.channels = width
        self.linear = nn.Linear(width, width)
        self.checkpointing = checkpointing

    def forward(self, point):
        if self.checkpointing and self.training and torch.is_grad_enabled():
            point.feat = checkpoint(self.linear, point.feat, use_reentrant=True)
        else:
            point.feat = self.linear(point.feat)
        return point


class FakePTv3(nn.Module):
    def __init__(self, width, checkpointing, **kwargs):
        super().__init__()
        self.net = nn.Module()
        self.net.enc = nn.ModuleDict({'enc0': nn.ModuleDict({'block0': FakeBlock(width, checkpointing)})})
        self.net.dec = nn.ModuleDict({'dec0': nn.ModuleDict({'block1': FakeBlock(width, checkpointing)})})

    def forward(self, features, coords, centers, time, resolution, *, interaction_residual=None):
        point = PointStub(feat=features)
        if interaction_residual is not None:
            point.state_condition_residual = interaction_residual
        point = self.net.enc['enc0']['block0'](point)
        return self.net.dec['dec0']['block1'](point).feat


@pytest.fixture
def models(monkeypatch):
    import fpsgen.hier_occ_flow.ptv3_backbone as backbone
    from fpsgen.hier_occ_flow.model import HierarchicalOccupancyModel
    monkeypatch.setattr(backbone, 'ParentPTv3', FakePTv3)
    geometry = SparseHierarchyGeometry(parent_resolution=(4,4,2), child_split=(2,2,2))
    kwargs = dict(width=16, checkpointing=True)
    b0 = HierarchicalOccupancyModel(geometry, **kwargs)
    m1 = HierarchicalOccupancyModel(geometry, state_condition_interaction='residual_v1', **kwargs)
    load_shared_baseline(m1, b0.state_dict())
    ids = torch.tensor([17, 2, 13, 5])
    condition = dict(state=torch.randn(4,8), time=torch.tensor(.35),
                     parent_score=torch.rand(4), lidar_features=torch.randn(4,384),
                     observed_child=torch.rand(4,8)>.7, valid_child_mask=torch.ones(4,8,dtype=torch.bool))
    return b0, m1, ids, condition


def test_zero_initialization_is_exact_and_preserves_baseline_state(models):
    b0, m1, ids, condition = models
    assert torch.equal(b0(ids, **condition), m1(ids, **condition))
    for key, value in b0.state_dict().items():
        assert torch.equal(value, m1.state_dict()[key])
    assert sum(p.numel() for p in m1.interaction.parameters()) < 100000


def test_checkpointed_gradients_reach_gates_then_interaction(models):
    _, m1, ids, condition = models
    optimizer = torch.optim.SGD(m1.parameters(), lr=.5)
    m1(ids, **condition).square().sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad) for x in m1.interaction.scales)
    assert any(x.grad.abs().item() > 0 for x in m1.interaction.scales)
    optimizer.step(); optimizer.zero_grad()
    # Both injections are outside the checkpointed blocks. A second training
    # update must retain the state/condition gradients without a double-backward.
    m1(ids, **condition).square().sum().backward()
    for module in (m1.interaction.static_encoder, m1.interaction.dynamic_encoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_ema_deepcopy_hooks_use_ema_parameters(models):
    _, m1, ids, condition = models
    ema = copy.deepcopy(m1).eval()
    with torch.no_grad():
        before = ema(ids, **condition)
        m1.interaction.scales[0].fill_(1.)
        assert torch.equal(before, ema(ids, **condition))
        ema.interaction.scales[0].fill_(.5)
        assert not torch.equal(before, ema(ids, **condition))


def test_parent_permutation_child_mask_and_output_count(models):
    _, m1, ids, condition = models
    with torch.no_grad():
        m1.interaction.scales[0].fill_(.3)
        m1.interaction.scales[1].fill_(.7)
    condition['valid_child_mask'][1, 5] = False
    out = m1(ids, **condition)
    order = torch.tensor([2,0,3,1])
    permuted = {k: v[order] if v.ndim and v.shape[0]==len(ids) else v for k,v in condition.items()}
    assert torch.allclose(out[order], m1(ids[order], **permuted), atol=1e-8, rtol=1e-6)
    assert out.shape == (4,8) and out[1,5] == 0 and torch.isfinite(out).all()


def test_state_dict_roundtrip_and_strict_shared_key_validation(models):
    b0, m1, ids, condition = models
    restored = copy.deepcopy(m1)
    with torch.no_grad():
        m1.interaction.scales[0].fill_(.8)
    restored.load_state_dict(m1.state_dict(), strict=True)
    assert torch.equal(m1(ids, **condition), restored(ids, **condition))
    state = dict(b0.state_dict()); state.pop('fuse.0.weight')
    with pytest.raises(ValueError, match='keys differ'):
        load_shared_baseline(restored, state)


def test_no_duplicate_hooks_or_silent_missing_token_field():
    interaction = StateConditionInteraction(4,4,16)
    net = FakePTv3(16, False).net
    interaction.install(net)
    with pytest.raises(ValueError, match='exactly once'):
        interaction.install(net)
    with pytest.raises(ValueError, match='original-resolution'):
        net.enc['enc0']['block0'](PointStub(feat=torch.zeros(3,16)))


def test_half_residual_enters_original_fp32_trunk_without_dtype_mismatch():
    interaction=StateConditionInteraction(4,4,16)
    net=FakePTv3(16,False).net
    interaction.install(net)
    point=PointStub(feat=torch.zeros(3,16),state_condition_residual=torch.ones(3,16,dtype=torch.float16))
    result=net.enc['enc0']['block0'](point)
    assert result.feat.dtype==torch.float32 and torch.isfinite(result.feat).all()


def test_cohort_changes_only_interaction_and_requires_new_evidence(tmp_path):
    from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,require_training_gate
    root=Path(__file__).resolve().parents[1]/'configs/research_v2/hier_occ_flow'
    b0,g=load_stage2_config(root/'stage2_384_desescene_sci_b0_20k.yaml')
    m1,_=load_stage2_config(root/'stage2_384_desescene_sci_m1_20k.yaml')
    b0compare=copy.deepcopy(b0);m1compare=copy.deepcopy(m1)
    for value in (b0compare,m1compare):
        value.pop('research_protocol');value['model'].pop('state_condition_interaction')
    assert b0compare==m1compare
    assert b0['candidates']['stage1_nfe']==2
    assert b0['training']['successful_frame_exposures']==20000
    assert b0['training']['successful_updates']==10000
    gate={key:True for key in ('full_scene_gpu','maximum_scene_gpu','ptv3_permutation',
        'checkpointing_parity','packed_batch2_gpu','research_model_smoke')}
    gate.update(geometry_sha256=g.sha256,stage1_checkpoint_sha256=b0['stage1']['checkpoint_sha256'])
    path=tmp_path/'gate.json';path.write_text(json.dumps(gate));m1['engineering_gate']=str(path)
    with pytest.raises(ValueError,match='state_condition_gpu_verified'):
        require_training_gate(m1,g)
