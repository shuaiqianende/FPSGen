"""Prepare two immutable additional-20K fine-tuning configs from one raw-GT source."""
from common import ROOT
import argparse
import copy
import json
from pathlib import Path

import yaml

from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import file_sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--source-checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    base, _ = load_stage2_config(a.template)
    base['model'] = dict(objective='fm', backbone='ptv3', width=64, lidar_channels=384,
                         condition_mode='pyramid_observed', checkpointing=True, patch_sizes=[64,128],
                         context_strategy='dynamic', state_condition_interaction='none')
    base['candidates']['stage1_nfe'] = 2
    base['candidates']['calibration_status'] = 'frozen_uncalibrated_threshold_0.5'
    base['initialization'] = {'type': 'paired_finetune',
        'checkpoint': str(a.source_checkpoint.resolve()), 'checkpoint_sha256': file_sha(a.source_checkpoint),
        'weights': 'model'}
    base['training'].update(scope='research_pilot', successful_updates=10000,
        successful_frame_exposures=20000, batch_per_gpu=2, seed=20261035, warmup_updates=500)
    base['engineering_gate'] = 'experiments/desescene_state_condition_v1/gate.json'
    base['research_protocol'] = 'desescene_sci_dynamic_b0_v1'
    base['validation']['enabled'] = False
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for name, mode in (('b0','none'), ('m1','residual_v1')):
        cfg = copy.deepcopy(base)
        cfg['research_protocol'] = 'desescene_sci_dynamic_' + name + '_v1'
        cfg['model']['state_condition_interaction'] = mode
        for suffix in ('20k', 'smoke'):
            output = copy.deepcopy(cfg)
            if suffix == 'smoke':
                output['training'].update(scope='engineering_smoke', successful_updates=32,
                    successful_frame_exposures=64, warmup_updates=8, checkpoint_every=32, milestone_every=32)
            path = a.output_dir / ('stage2_384_desescene_sci_' + name + '_' + suffix + '.yaml')
            if path.exists():
                raise FileExistsError(path)
            path.write_text(yaml.safe_dump(output, sort_keys=False))
            load_stage2_config(path)
            print(path)


if __name__ == '__main__':
    main()
