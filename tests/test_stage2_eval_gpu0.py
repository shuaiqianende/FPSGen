"""GPU0 authorization, bounded sharing and preserving other workloads."""
import importlib.util
from pathlib import Path
import pytest

path=Path('scripts/research_v2/hier_occ_inference/eval_gpu.py')
spec=importlib.util.spec_from_file_location('stage2_eval_gpu0',path)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


def setup_queries(monkeypatch, *, memory=346, command='uvicorn', utilization=0):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    def query(args,**kwargs):
        if args[0]=='ps':return command+'\n'
        if '--query-compute-apps=pid,used_memory' in args:return f'3399428, {memory}\n'
        return f'23751, {utilization}\n'
    monkeypatch.setattr(entry.subprocess,'check_output',query)


def test_gpu0_requires_explicit_visibility_and_background_identity(monkeypatch):
    setup_queries(monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    with pytest.raises(ValueError):entry.admit_gpu0(3399428)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(RuntimeError):entry.admit_gpu0()
    assert entry.admit_gpu0(3399428)['allocator_memory_fraction']==.25


@pytest.mark.parametrize('settings',[{'memory':600},{'command':'torchrun'},{'utilization':30}])
def test_gpu0_rejects_busy_or_unidentified_workloads(monkeypatch,settings):
    setup_queries(monkeypatch,**settings)
    with pytest.raises(RuntimeError):entry.admit_gpu0(3399428)
