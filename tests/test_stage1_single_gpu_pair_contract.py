from pathlib import Path
import copy
import hashlib
import json

import pytest
import yaml

from fpsgen.hier_occ_flow.stage1_training import (
    SINGLE_GPU_20K_SCOPE,
    _verified_smoke_data_audit,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/research_v2/hier_occ_flow'


def _load(name):
    p = CONFIGS / name
    cfg = yaml.safe_load(p.read_text())
    cfg['geometry'] = yaml.safe_load((CONFIGS / cfg['geometry']).read_text())
    return cfg


def test_r0_e3_single_gpu_arms_share_exposure_and_schedule():
    r0 = _load('fidelity_v2_single_gpu_r0_20k_gpu2.yaml')
    e3 = _load('fidelity_v2_single_gpu_e3_20k_gpu3.yaml')
    validate_config(r0)
    validate_config(e3)
    assert r0['scope'] == e3['scope'] == SINGLE_GPU_20K_SCOPE
    assert r0['training']['successful_frame_exposures'] == e3['training']['successful_frame_exposures'] == 20000
    assert r0['training']['successful_updates'] == e3['training']['successful_updates'] == 10000
    assert r0['training']['batch_per_gpu'] == e3['training']['batch_per_gpu'] == 2
    assert r0['training']['warmup_updates'] * r0['training']['batch_per_gpu'] == 2000
    assert e3['training']['warmup_updates'] * e3['training']['batch_per_gpu'] == 2000
    assert r0['training']['physical_gpus'] == [2]
    assert e3['training']['physical_gpus'] == [3]
    for key in ('seed', 'lr_encoder', 'lr_flow', 'ema_decay', 'weight_decay', 'precision'):
        assert r0['training'][key] == e3['training'][key]
    assert r0['model']['encoder_variant'] == 'original'
    assert e3['model']['encoder_variant'] == 'lightweight_skip'


@pytest.mark.parametrize('mutate', [
    lambda c: c['training'].update(successful_frame_exposures=100000),
    lambda c: c['training'].update(successful_updates=5000),
    lambda c: c['training'].update(batch_per_gpu=4),
    lambda c: c['training'].update(physical_gpus=[1]),
    lambda c: c['model'].update(encoder_variant='bottleneck96'),
])
def test_single_gpu_scope_rejects_unregistered_changes(mutate):
    cfg = copy.deepcopy(_load('fidelity_v2_single_gpu_r0_20k_gpu2.yaml'))
    mutate(cfg)
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_verified_smoke_audit_checks_roots_manifests_and_finite_updates(tmp_path):
    roots = {}
    for split in ('train', 'development'):
        root = tmp_path / split
        root.mkdir()
        (root / 'manifest.json').write_text('{"verified":true}\n')
        roots[split] = str(root)
    sources = []
    for name in ('r0', 'e3'):
        smoke = tmp_path / f'smoke_{name}'
        smoke.mkdir()
        (smoke / 'config.json').write_text(json.dumps({
            'train_dataset': roots['train'],
            'development_dataset': roots['development'],
        }))
        (smoke / 'summary.json').write_text(json.dumps({
            'status': 'COMPLETE_REQUESTED_BUDGET',
            'successful_updates': 2,
            'successful_frame_exposures': 4,
        }))
        (smoke / 'train_rank0.jsonl').write_text(
            '{"finite":true}\n{"finite":true}\n'
        )
        sources.append(str(smoke))
    manifests = {
        'train': hashlib.sha256((Path(roots['train']) / 'manifest.json').read_bytes()).hexdigest(),
        'development': hashlib.sha256((Path(roots['development']) / 'manifest.json').read_bytes()).hexdigest(),
    }
    audit = tmp_path / 'verified_data_audit.json'
    audit.write_text(json.dumps({
        'schema': 'hs-occflow-verified-smoke-data-v1',
        'roots': roots,
        'manifest_sha256': manifests,
        'source_smokes': sources,
    }))
    cfg = {'train_dataset': roots['train'], 'development_dataset': roots['development']}
    assert len(_verified_smoke_data_audit(audit, cfg)) == 64


def test_verified_smoke_audit_rejects_changed_manifest(tmp_path):
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({
        'schema': 'hs-occflow-verified-smoke-data-v1',
        'roots': {'train': '/wrong/train', 'development': '/wrong/dev'},
        'manifest_sha256': {'train': 'x', 'development': 'y'},
        'source_smokes': [],
    }))
    with pytest.raises(ValueError, match='roots'):
        _verified_smoke_data_audit(audit, {
            'train_dataset': '/actual/train',
            'development_dataset': '/actual/dev',
        })
