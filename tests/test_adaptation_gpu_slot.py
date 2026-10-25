import pytest
from fpsgen.hier_occ_flow import adaptation_gpu_slot as slot


def test_default_admission_retains_exclusive_policy(monkeypatch):
    monkeypatch.delenv('DESESCENE_SHARED_ADAPTATION',raising=False)
    calls=[]
    monkeypatch.setattr(slot,'gpu_preflight',lambda ids,**kw:calls.append((ids,kw)) or {'status':'BLOCKED'})
    assert slot.adaptation_admission(2)['status']=='BLOCKED'
    assert calls==[((2,),{})]


def test_shared_admission_has_headroom_without_reducing_batch(monkeypatch):
    monkeypatch.setenv('DESESCENE_SHARED_ADAPTATION','1')
    calls=[]
    monkeypatch.setattr(slot,'gpu_preflight',lambda ids,**kw:calls.append((ids,kw)) or {'status':'AVAILABLE'})
    result=slot.adaptation_admission(3)
    assert calls==[((3,),dict(minimum_free_gib=14.,allow_shared_compute=True))]
    assert result['allocator_limit_gib']==12 and result['no_token_truncation']
    with pytest.raises(ValueError): slot.adaptation_admission(1)


def test_memory_cap_only_applies_when_admitted(monkeypatch):
    monkeypatch.setenv('DESESCENE_SHARED_ADAPTATION','1')
    monkeypatch.setattr(slot,'gpu_preflight',lambda *a,**kw:{'status':'BLOCKED'})
    assert slot.adaptation_admission(2,configure_allocator=True)['status']=='BLOCKED'


def test_memory_cap_uses_visible_gpu_and_preserves_external_reserve(monkeypatch):
    import torch
    from types import SimpleNamespace
    monkeypatch.setenv('DESESCENE_SHARED_ADAPTATION','1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','3')
    monkeypatch.setattr(slot,'gpu_preflight',lambda *a,**kw:{'status':'AVAILABLE'})
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(total_memory=24*2**30))
    calls=[]
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda fraction,device:calls.append((fraction,device)))
    slot.adaptation_admission(3,configure_allocator=True)
    assert calls==[(.5,0)]
    with pytest.raises(ValueError):slot.adaptation_admission(2,configure_allocator=True)
