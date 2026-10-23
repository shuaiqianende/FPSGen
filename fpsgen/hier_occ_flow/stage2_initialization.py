"""Strict source-weight initialization for a separately registered matched cohort."""
import json
from pathlib import Path

import torch

from .state_condition_interaction import load_shared_baseline
from .storage import file_sha


def initialize_matched_finetune(model, cfg, geometry, manifests):
    spec = cfg.get('initialization')
    if spec is None:
        return None
    if spec.get('type') != 'paired_finetune' or spec.get('weights') not in ('model', 'ema'):
        raise ValueError('Explicit paired fine-tuning type and source-weight policy required')
    checkpoint = Path(spec['checkpoint'])
    if file_sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Source checkpoint SHA mismatch')
    saved = torch.load(checkpoint, map_location='cpu')
    identity = saved['identity']
    from .stage2_runtime import config_sha
    source_cfg = json.loads((Path(identity['output']) / 'config.json').read_text())
    if config_sha(source_cfg) != identity['config_sha256']:
        raise ValueError('Source checkpoint/config identity mismatch')
    if (identity.get('experiment') != 'hs-occflow-stage2-384-v1'
            or identity.get('geometry_sha256') != geometry.sha256
            or identity.get('stage1_checkpoint_sha256') != cfg['stage1']['checkpoint_sha256']
            or any(identity.get(k) != v for k, v in manifests.items())
            or saved.get('parameter_sync_pass') is not True):
        raise ValueError('Source data/geometry/Stage1/synchronization identity mismatch')
    if (source_cfg.get('target_data_version', 'hs-occ-surface-hit-v1') != 'hs-occ-surface-hit-v1'
            or cfg.get('target_data_version', 'hs-occ-surface-hit-v1') != 'hs-occ-surface-hit-v1'):
        raise ValueError('Matched cohort uses original raw-map labels, never morphology weights')
    for key in ('objective', 'backbone', 'width', 'lidar_channels', 'condition_mode', 'patch_sizes'):
        if source_cfg['model'][key] != cfg['model'][key]:
            raise ValueError(f'Source architecture mismatch: {key}')
    if (source_cfg['model'].get('context_strategy', 'dynamic') != 'dynamic'
            or source_cfg['model'].get('child_bridge', 'none') != 'none'):
        raise ValueError('Source must be original dynamic FM without Child decoder')
    state = saved[spec['weights']]
    if state is None:
        raise ValueError('Source weight policy unavailable')
    loaded = load_shared_baseline(model, state)
    return {'type': 'paired_finetune', 'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': spec['checkpoint_sha256'], 'weights': spec['weights'],
            'source_identity': identity, 'source_candidates': source_cfg['candidates'],
            'source_successful_updates': saved['successful_updates'],
            'source_successful_frame_exposures': saved['sampler']['successful_frame_exposures'],
            'new_successful_frame_exposures': cfg['training']['successful_frame_exposures'],
            'optimizer_ema_scheduler_restarted': True, 'loaded': loaded,
            'claim': '20K additional matched adaptation, not 20K total scratch training'}
