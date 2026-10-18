"""Launch audit rejects inherited state or changes confined to output/buffers."""
import copy,importlib.util,json
from pathlib import Path
import pytest
import torch


def audit_function(monkeypatch):
    path=Path('scripts/research_v2/hier_occ_flow/start_stage2_morph_training.py').resolve()
    monkeypatch.syspath_prepend(str(path.parent))
    spec=importlib.util.spec_from_file_location('hs_morph_launch_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify_launch


def artifacts(root):
    root.mkdir();(root/'checkpoints').mkdir()
    names=[x+'.weight' for x in ('local','trunk','decoder','score')]
    start={'model':{name:torch.zeros(2) for name in names},'optimizer':{'state':{}},
        'successful_updates':0,'rank_hashes':[{'online':'initial','ema':'initial'}]}
    start['ema']=copy.deepcopy(start['model'])
    current={'model':{name:torch.ones(2) for name in names},'successful_updates':20,
        'sampler':{'successful_frame_exposures':80},'parameter_sync_pass':True,'rank_rng':[{},{}]}
    (root/'launch.json').write_text(json.dumps({'trainable_parameter_names':names,'optimizer_entries':0,'transition':None,'global_batch':4}))
    for rank in (0,1):
        row={'finite':True,'successful_updates':20,'successful_frame_exposures':80,'loss':1.9,
            'gradient_norm':.3,'amp_skips':0,'seconds':4.,'tokens':400000,
            'peak_allocated_bytes':1000,'peak_reserved_bytes':2000}
        (root/f'train_rank{rank}.jsonl').write_text(json.dumps(row)+'\n')
    return start,current


@pytest.mark.parametrize('fault',[None,'inherited_optimizer','untrained_trunk','wrong_initial_ema','wrong_frame_budget'])
def test_actual_launch_requires_fresh_parameters_and_real_updates(tmp_path,monkeypatch,fault):
    root=tmp_path/'run';start,current=artifacts(root)
    if fault=='inherited_optimizer':start['optimizer']['state']={0:{'step':torch.tensor(5.)}}
    if fault=='untrained_trunk':current['model']['trunk.weight']=torch.zeros(2)
    if fault=='wrong_initial_ema':start['ema']['score.weight']=torch.ones(2)
    if fault=='wrong_frame_budget':current['sampler']['successful_frame_exposures']=79
    torch.save(start,root/'checkpoints/step000000.pt');torch.save(current,root/'checkpoints/step000020.pt')
    verify=audit_function(monkeypatch)
    if fault:
        with pytest.raises(ValueError):verify(root)
    else:
        report=verify(root)
        assert report['status']=='PASS_FRESH_RUNNING' and report['changed_trainable_parameter_tensors']==4
        assert report['parameter_delta_max_abs_by_module']['trunk']==1.
