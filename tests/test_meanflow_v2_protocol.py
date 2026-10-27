import copy
import json
from pathlib import Path
import pytest
import yaml
from fpsgen.hier_occ_flow.meanflow_v2_training import load_config
from scripts.research_v2.meanflow_v2.analyze import gap_name, summary, bootstrap

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('arm', ['a','b'])
def test_registered_config_exact_budget_and_separate_identity(arm):
    c,g=load_config(ROOT/f'configs/research_v2/hier_occ_flow/meanflow_v2_{arm}_gpu2.yaml')
    assert c['training']['successful_updates']==10000
    assert c['training']['successful_frame_exposures']==20000
    assert c['training']['batch_per_gpu']==2
    assert c['physical_gpu']==2 and c['initialization_mode']=='scratch'
    assert c['meanflow_v2']['tangent']==('conditional' if arm=='a' else 'detached_boundary')


@pytest.mark.parametrize('change', ['seed','tangent','epsilon','budget','source','batch'])
def test_rejects_unregistered_change(tmp_path,change):
    c=yaml.safe_load((ROOT/'configs/research_v2/hier_occ_flow/meanflow_v2_a_gpu2.yaml').read_text())
    if change=='seed':c['training']['seed']+=1
    if change=='tangent':c['meanflow_v2']['tangent']='detached_boundary'
    if change=='epsilon':c['meanflow_v2']['adaptive_epsilon']=1e-3
    if change=='budget':c['training']['successful_frame_exposures']=100000
    if change=='source':c['source_stage1']='forbidden'
    if change=='batch':c['training']['batch_per_gpu']=1
    p=tmp_path/'bad.yaml';p.write_text(yaml.safe_dump(c))
    with pytest.raises(ValueError):load_config(p)


def test_delta_bins_are_exhaustive_with_exact_diagonal():
    assert gap_name(0)=='0'
    assert gap_name(.125)=='(0.0,0.125]'
    assert gap_name(.126)=='(0.125,0.25]'
    assert gap_name(1)=='(0.75,1.0]'
    with pytest.raises(ValueError):gap_name(-.1)
    assert summary([None])['mean'] is None


def test_single_seed_paired_bootstrap_marks_scope():
    result=bootstrap([.01]*41)
    assert result['ci95'][0]==pytest.approx(.01)
    assert 'not cross-seed' in result['method']


def test_sequential_queue_does_not_start_b_after_a_failure(tmp_path,monkeypatch):
    from scripts.research_v2.meanflow_v2 import sequence
    from types import SimpleNamespace
    import sys
    calls=[]
    monkeypatch.setattr(sequence,'own_sha',lambda:'source')
    monkeypatch.setattr(sequence,'load_config',lambda p:({'arm':'MFV2-A'},None))
    def fail(*args,**kwargs):
        calls.append(args[0]);return SimpleNamespace(pid=123,wait=lambda:1)
    monkeypatch.setattr(sequence.subprocess,'Popen',fail)
    out=tmp_path/'new_sequence'
    monkeypatch.setattr(sys,'argv',['sequence','--output',str(out)])
    with pytest.raises(RuntimeError,match='later arms BLOCKED'):sequence.main()
    assert len(calls)==1
    assert json.loads((out/'status.json').read_text())['status']=='BLOCKED_REVIEW_REQUIRED'
