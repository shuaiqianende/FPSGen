"""CPU checks for the evaluation/continuation identity and precision policy."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from fpsgen.hier_occ_flow.explicit_fp16_training_policy import FP32Masters, apply_explicit_policy, update_fp32_ema

ROOT = Path(__file__).resolve().parents[1]


def test_u1_budget_and_explicit_precision():
    cfg = yaml.safe_load((ROOT / 'configs/research_v2/hier_occ_flow/u1_fulltrain_explicit_fp16_100k.yaml').read_text())
    assert cfg['model']['encoder_variant'] == 'patch_u1'
    assert cfg['model']['initialization'] == 'scratch'
    assert cfg['training']['batch_per_gpu'] * cfg['training']['successful_updates'] == 100000
    assert cfg['training']['precision'] == 'explicit_fp16_fp32_master'
    assert cfg['training']['compile_target'] == 'encoder.down_blocks.0.forward'


def test_gpu3_followup_changes_only_registered_condition_combination():
    cfg = yaml.safe_load((ROOT / 'configs/research_v2/hier_occ_flow/stage2_child_b4_attention.yaml').read_text())
    assert cfg['model']['local_state_encoding'] == 'position'
    assert cfg['model']['late_lidar_injection'] is True
    assert cfg['candidates']['negative_fraction'] == 0
    assert cfg['training']['batch_per_gpu'] == 4
    assert cfg['training']['successful_updates'] * cfg['training']['batch_per_gpu'] == 100000


def test_fp16_network_fp32_master_and_ema():
    torch.manual_seed(3)
    model = nn.Sequential(nn.Conv3d(2, 4, 1), nn.GroupNorm(2, 4))
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    master = FP32Masters(model, initial)
    apply_explicit_policy(model)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    ema = {name: value.clone() for name, value in initial.items()}
    optimizer = torch.optim.AdamW(master.optimizer_groups([{'params': list(model.parameters()), 'lr': .01}]))
    # PyTorch 2.0 does not provide CPU half-precision Conv3d. Exercise the
    # master/gradient path directly; real convolution is covered by GPU gate.
    sum(p.float().square().mean() for p in model.parameters()).backward()
    master.copy_gradients()
    optimizer.step()
    master.copy_to_network()
    update_fp32_ema(ema, ema_model, model, master)
    assert all(p.dtype == torch.float16 for p in model.parameters())
    assert all(p.dtype == torch.float32 for p in master.params.values())
    assert all(v.dtype == torch.float32 for v in ema.values() if v.is_floating_point())
    assert any(not torch.equal(initial[n], p.detach()) for n, p in master.params.items())
    assert all(torch.equal(p, master.params[n].to(p.dtype)) for n, p in model.named_parameters())


def test_seq08_skip10_is_408_distinct_frames():
    frames = list(range(0, 4071, 10))
    assert len(frames) == len(set(frames)) == 408
    assert frames[0] == 0 and frames[-1] == 4070


def test_voxel_resampling_exact_count_and_geometry():
    path = ROOT / 'scripts/research_v2/hier_occ_inference/voxel_guided_resampling.py'
    spec = importlib.util.spec_from_file_location('resampling', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bounds = {'minimum': [-1., -1., -1.], 'maximum': [1., 1., 1.], 'sphere_radius': 1.}
    out, meta = module.voxel_guided_uniform_resample(np.array([[0., 0., 0.], [.3, .3, .3]]),
        bounds=bounds, voxel_size=.15625, target_points=10000, seed=20261019)
    assert out.shape == (10000, 3) and meta['actual_output_points'] == 10000
    assert np.isfinite(out).all() and (np.linalg.norm(out.astype(np.float64), axis=1) < 1.).all()
