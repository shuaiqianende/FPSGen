import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'scripts/research_v2/hier_occ_flow'


@pytest.fixture
def queue_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec=importlib.util.spec_from_file_location('matched_queue',SCRIPTS/'run_desescene_m1_matched_queue.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def records(folder, *, unstable=False, exposures=200):
    (folder/'summary.json').write_text(json.dumps({'successful_updates':100,'successful_frame_exposures':exposures}))
    rows=[{'successful_updates':i,'successful_frame_exposures':i*2,'global_batch':2,'batch_per_gpu':2,
           'finite':True,'loss':.9,'gradient_norm':.3,'amp_skips':0,'attempts':i,'seconds':2.}
          for i in range(1,101)]
    if unstable:rows[-1]['finite']=False
    (folder/'train_rank0.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')


def test_health_accepts_only_actual_successful_batch2_budget(queue_module,tmp_path):
    records(tmp_path)
    assert queue_module.health_report(tmp_path,100)['status']=='PASS'
    records(tmp_path,exposures=199)
    with pytest.raises(RuntimeError,match='Health check failed'):
        queue_module.health_report(tmp_path,100)


def test_health_blocks_recent_amp_failure_and_preserves_report(queue_module,tmp_path):
    records(tmp_path,unstable=True)
    with pytest.raises(RuntimeError,match='Health check failed'):
        queue_module.health_report(tmp_path,100)
    assert json.loads((tmp_path/'health_000100.json').read_text())['status']=='FAIL'


def test_user_authorized_gpu1_still_requires_idle_admission(monkeypatch):
    from fpsgen.hier_occ_flow import resources
    from types import SimpleNamespace
    def run(args,**kwargs):
        return SimpleNamespace(returncode=0,stdout=('20000,24576,0,580.159.03\n'
            if 'memory.free' in ' '.join(args) else '12345\n'),stderr='')
    monkeypatch.setattr(resources.subprocess,'run',run)
    result=resources.gpu_preflight((1,))
    assert result['status']=='BLOCKED'
    assert result['gpus'][0]['existing_compute_pids']==[12345]
    with pytest.raises(ValueError):resources.gpu_preflight((0,))


def test_smoke_gate_counts_successes_and_discloses_amp_overflow(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from register_desescene_m1_gate import validate_smoke
    rows=[{'successful_updates':i,'finite':True,'amp_skips':int(i>10)} for i in range(1,33)]
    rows.insert(10,{'successful_updates':10,'finite':False,'amp_skips':1})
    summary={'successful_updates':32,'successful_frame_exposures':64,'attempts':33}
    launch={'global_batch':2,'identity':{'source_sha256':'source'},'initialization':'paired_finetune_stage2'}
    cfg={'training':{'seed':20261032}}
    result=validate_smoke(summary,launch,cfg,rows,'source')
    assert result['amp_skips']==1 and result['attempts']==33
    assert result['successful_frame_exposures']==64
    rows[-1]['finite']=False
    with pytest.raises(ValueError):validate_smoke(summary,launch,cfg,rows,'source')
