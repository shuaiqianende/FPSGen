"""Register the user's100K successful-frame budget from the actual latest phase."""
from train_stage1_occ import ROOT,load_config
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import torch
import yaml
from fpsgen.hier_occ_flow.stage1_frame_budget import final_update_for_frames,verify_budget_transition,learning_rate_multiplier
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',type=Path,default=Path('experiments/hier_occ_flow/stage1_occ384_fm_100k_gb4/checkpoints/step020000.pt'))
    args=p.parse_args();source=args.source.resolve();old_root=source.parent.parent
    value=torch.load(source,map_location='cpu');old=json.loads((old_root/'config.json').read_text())
    frames=value['sampler']['successful_frame_exposures'];updates=value['successful_updates'];global_batch=old['training']['batch_per_gpu']*value['identity']['world_size']
    terminal=final_update_for_frames(updates,frames,global_batch,100000)
    old['training'].update(successful_updates=terminal,successful_frame_exposures=100000)
    old['engineering_gate']='experiments/hier_occ_flow/stage1_occ384_frames_engineering_gate.json'
    old['budget_transition']={'checkpoint':str(source),'checkpoint_sha256':file_sha(source),'source_config':str(old_root/'config.json'),
        'source_updates':updates,'source_successful_frames':frames,'previous_update_budget':value['training_budget']['successful_updates'] if value.get('training_budget') else 100000,
        'lr_policy':'continuous_cosine_to_frame_budget','anchor_lr_factor':float(value['optimizer']['param_groups'][0]['lr']/old['training']['lr_encoder'])}
    old['geometry']='geometry_384_8way.yaml'
    config=Path('configs/research_v2/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4.yaml');config.write_text(yaml.safe_dump(old,sort_keys=False))
    cfg=load_config(config);identity=dict(value['identity']);identity['output']=str((ROOT/'experiments/hier_occ_flow/stage1_occ384_fm_100k_frames_gb4').resolve())
    identity['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    verify_budget_transition(value,identity,cfg,source)
    suites=ET.parse('experiments/hier_occ_flow/stage1_frame_budget_cpu_tests.xml').getroot()
    tests={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    if tests['failures'] or tests['errors']:raise ValueError('Frame-budget CPU tests did not pass')
    result={'status':'PASS','config_sha256':identity['config_sha256'],'tests':tests,'source_updates':updates,'source_successful_frames':frames,
        'global_batch':global_batch,'target_successful_frames':100000,'target_successful_updates':terminal,
        'remaining_updates':terminal-updates,'source_checkpoint_sha256':old['budget_transition']['checkpoint_sha256'],
        'lr_at_boundary':[g['lr'] for g in value['optimizer']['param_groups']],
        'lr_factor_after_restore':learning_rate_multiplier(updates,cfg['training'],cfg['budget_transition']),
        'lr_factor_at_final':learning_rate_multiplier(terminal,cfg['training'],cfg['budget_transition'])}
    atomic_json('experiments/hier_occ_flow/stage1_frame_budget_preflight.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
