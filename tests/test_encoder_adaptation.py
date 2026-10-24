import copy
from types import SimpleNamespace
import pytest
import torch
import json
import yaml
from pathlib import Path
from torch import nn

from fpsgen.hier_occ_flow.encoder_adaptation import (EncoderAdaptationSystem,
    frozen_state_sha,parameter_sha,tensor_state_sha,update_adaptation_ema,stage2_differentiable_batch)
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import make_optimizer,verify_adaptation_resume,load_adaptation_config
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage2_batching import packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.trainer import rng_state,restore_rng


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv3d(1,128,1);self.bn=nn.BatchNorm3d(128)
    def forward(self,xyz):
        x=torch.stack([p[:,0].reshape(1,2,2,2) for p in xyz])
        f=self.bn(self.conv(x))
        return (f,f*.7,f*.3)


class Flow(nn.Module):
    def __init__(self):super().__init__();self.conv=nn.Conv3d(128,1,1);self.bn=nn.BatchNorm3d(1)
    def forward(self,state,observed,condition,time):return self.bn(self.conv(condition[0]))+.1*state


class Stage1(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.flow=Flow();self.register_buffer('valid',torch.ones(1,1,2,2,2,dtype=torch.bool),persistent=False)
        self.geometry=SimpleNamespace(observed_mask=lambda xyz:torch.zeros(1,1,2,2,2,dtype=torch.bool))


class Stage2(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(384,8);self.bn=nn.BatchNorm1d(8)
    def forward(self,ids,*,state,time,lidar_features,valid_child_mask,**other):return self.bn(self.linear(lidar_features.float()))*valid_child_mask+.1*state


T={'lr_encoder':2e-5,'lr_flow':1e-5,'lr_stage2':2e-4,'weight_decay':.01}


@pytest.mark.parametrize('arm',['EF1','EF2'])
def test_stage1_loss_reaches_encoder_through_frozen_flow_and_ema_policy(arm):
    torch.manual_seed(4);model=EncoderAdaptationSystem(Stage1(),arm);ema=copy.deepcopy(model).eval().requires_grad_(False)
    frozen=frozen_state_sha(model);before=parameter_sha(model.stage1.encoder)
    opt,_=make_optimizer(model,T);xyz=[torch.randn(8,3),torch.randn(8,3)]
    loss,_=model.stage1_loss(xyz,torch.ones(2,1,2,2,2),torch.Generator().manual_seed(5));loss.backward()
    assert model.stage1.encoder.conv.weight.grad.abs().sum()>0
    if arm=='EF1':
        assert not model.stage1.flow.training and all(p.grad is None for p in model.stage1.flow.parameters())
    opt.step();update_adaptation_ema(ema,model,.999)
    assert parameter_sha(model.stage1.encoder)!=before
    assert frozen_state_sha(model)==frozen_state_sha(ema)==frozen


@pytest.mark.parametrize('arm',['S2-E1','S2-E2'])
def test_stage2_grid_sample_and_frozen_ptv3_surrogate_reach_encoder(arm):
    torch.manual_seed(11);m=EncoderAdaptationSystem(Stage1(),arm,Stage2());frozen=frozen_state_sha(m)
    g=SparseHierarchyGeometry(parent_resolution=(2,2,2),child_split=(2,2,2))
    xyz=[torch.randn(8,3),torch.randn(8,3)];scenes=[]
    for i in range(2):
        ids=torch.tensor([0,1,2,3]);c={'parent_score':torch.ones(4),'observed_child':torch.zeros(4,8,dtype=torch.bool),
            'valid_child_mask':torch.ones(4,8,dtype=torch.bool),'lidar_features':torch.zeros(4,384)}
        scenes.append((ids,c,torch.ones(4,8),{'source':'input_only_teacher'}))
    ids,c,target,lengths,pyramid=stage2_differentiable_batch(m,[None,None],xyz,{},g,None,teacher_scenes=scenes)
    assert lengths==[4,4] and c['lidar_features'].requires_grad
    pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],torch.Generator().manual_seed(8))
    loss=scene_mean_mse(m.stage2(ids,state=pair['state'],time=pair['time'],**c),pair['velocity'],c['valid_child_mask'],c['batch'])
    loss.backward();assert m.stage1.encoder.conv.weight.grad.abs().sum()>0
    opt,_=make_optimizer(m,T);opt.step()
    assert frozen_state_sha(m)==frozen
    if arm=='S2-E1':assert all(p.grad is None for p in m.stage2.parameters()) and not m.stage2.bn.training


def test_adaptation_resume_preserves_optimizer_rng_and_rejects_scratch_identity():
    torch.manual_seed(3);m=EncoderAdaptationSystem(Stage1(),'EF1');opt,sched=make_optimizer(m,T)
    ema=copy.deepcopy(m).eval().requires_grad_(False)
    gen=torch.Generator().manual_seed(7);xyz=[torch.randn(8,3),torch.randn(8,3)]
    def step():
        opt.zero_grad();loss,_=m.stage1_loss(xyz,torch.ones(2,1,2,2,2),gen);loss.backward();opt.step();sched.step();update_adaptation_ema(ema,m,.999);return float(loss)
    step();identity={'experiment':'new_adaptation','arm':'EF1'}
    saved={'identity':identity,'initialization_mode':'pretrained_ema_adaptation','freeze_policy':m.trainability_audit(),
        'successful_updates':1,'successful_frame_exposures':2,'model':copy.deepcopy(m.state_dict()),
        'ema':copy.deepcopy(ema.state_dict()),'attempts':1,'amp_skips':0,'sample_cursor':2,
        'optimizer':copy.deepcopy(opt.state_dict()),'scheduler':copy.deepcopy(sched.state_dict()),'rng':rng_state(gen,torch.device('cpu'))}
    expected=step();expected_params=parameter_sha(m.stage1.encoder);expected_ema=tensor_state_sha(ema.state_dict())
    verify_adaptation_resume(saved,identity,m)
    m.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);restore_rng(saved['rng'],gen,torch.device('cpu'))
    assert step()==expected and parameter_sha(m.stage1.encoder)==expected_params and tensor_state_sha(ema.state_dict())==expected_ema
    saved['initialization_mode']='scratch'
    with pytest.raises(ValueError,match='adaptation'):verify_adaptation_resume(saved,identity,m)


@pytest.mark.parametrize('field,value',[('batch_per_gpu',1),('successful_updates',50000),
    ('successful_frame_exposures',100000),('seed',123),('optimizer','SGD'),('lr_encoder',1e-4)])
def test_registered_protocol_rejects_silent_changes(tmp_path,field,value):
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'configs/research_v2/hier_occ_flow/encoder_adaptation_ef1_20k.yaml').read_text())
    cfg['training'][field]=value
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_adaptation_config(path)


def test_stage2_only_control_freezes_encoder_stats_and_ema():
    torch.manual_seed(19);m=EncoderAdaptationSystem(Stage1(),'S2-E0',Stage2())
    ema=copy.deepcopy(m).eval().requires_grad_(False);before=frozen_state_sha(m)
    x=[torch.randn(8,3),torch.randn(8,3)]
    features=m.stage1.encoder(x)[0].flatten(2).transpose(1,2).reshape(16,128).repeat(1,3)
    prediction=m.stage2.linear(features);prediction.square().mean().backward()
    assert not m.stage1.encoder.training and all(p.grad is None for p in m.stage1.encoder.parameters())
    opt,_=make_optimizer(m,T);opt.step();update_adaptation_ema(ema,m,.999)
    assert frozen_state_sha(m)==frozen_state_sha(ema)==before


def test_resume_cannot_change_freeze_policy_or_successful_exposure_identity():
    m=EncoderAdaptationSystem(Stage1(),'EF1');identity={'arm':'EF1'}
    saved={'identity':identity,'initialization_mode':'pretrained_ema_adaptation',
        'freeze_policy':m.trainability_audit(),'successful_updates':32,'successful_frame_exposures':65}
    with pytest.raises(ValueError,match='exposure'):verify_adaptation_resume(saved,identity,m)
    saved['successful_frame_exposures']=64
    other=EncoderAdaptationSystem(Stage1(),'EF2')
    with pytest.raises(ValueError,match='Freeze'):verify_adaptation_resume(saved,identity,other)
