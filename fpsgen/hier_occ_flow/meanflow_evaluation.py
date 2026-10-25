"""Same-checkpoint NFE sweeps; fixed other stage, absolute-Parent noise, native centers."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .meanflow_training import MeanFlowSystem,config_sha
from .meanflow_sampling import sample_meanflow,stage1_score
from .meanflow_control_audit import audit_control
from .encoder_adaptation import load_stage2_source
from .stage1_network import OccupancyUNet
from .stage2_condition import input_conditions,scene_seed
from .voxelization import pack_children,occupied_fine_coords
from .storage import file_sha


class EncodedInput:
    def __init__(self,pyramid,score,provenance):self.pyramid=pyramid;self.score=score;self.provenance=provenance
    def __call__(self,xyz,*,nfe,seed,with_scores=True):return self.pyramid,self.score


class EvaluationModels:
    def __init__(self,cfg,g,checkpoint,device):
        self.cfg=cfg;self.g=g;self.device=torch.device(device)
        self.system=MeanFlowSystem(cfg,g,device);self.mf=self.system.network.eval()
        saved=torch.load(checkpoint,map_location='cpu')
        if (saved['identity']['config_sha256']!=config_sha(cfg) or saved['counts']['updates']!=10000
            or saved['counts']['exposures']!=20000):raise ValueError('Frozen final matching20K checkpoint required')
        self.mf.load_state_dict(saved['ema'],strict=True)
        self.checkpoint_sha=file_sha(checkpoint);self.identity=saved['identity'];del saved
        self.control_audit=audit_control(cfg)
        if self.control_audit['status']!='MATCHED_CONTROL':raise ValueError('Cannot label unmatched historical checkpoint a paired control')
        control=torch.load(self.control_audit['checkpoint'],map_location='cpu')
        if cfg['arm']=='S1-MF20K':
            source=json.loads(Path(cfg['source_stage1']['config']).read_text())
            self.fm=OccupancyUNet('fm',widths=source['model']['widths'],activation_checkpointing=False).to(device).eval()
            self.fm.load_state_dict({k[len('stage1.flow.'):]:v for k,v in control['ema'].items() if k.startswith('stage1.flow.')},strict=True)
            self.fine,_=load_stage2_source(cfg,g);self.fine=self.fine.to(device).eval()
        else:
            self.fm,_=load_stage2_source(cfg,g);self.fm=self.fm.to(device).eval()
            self.fm.load_state_dict(control['ema'],strict=True);self.fine=None
        del control
        self.teacher=self.system.teacher;self.teacher.model.encoder.eval()
        if cfg['arm']=='S1-MF20K':self.mf.eval()
        else:self.teacher.model.eval()
        for model in (self.mf,self.fm,self.fine):
            if model is None:continue
            base=getattr(model,'base',model)
            if hasattr(base,'ptv3_topology_cache'):base.ptv3_topology_cache='multilevel'
        self.valid=self.teacher.model.valid

    @torch.no_grad()
    def generate(self,xyz,sequence,frame,*,variant,nfe):
        if variant not in ('meanflow','matched_fm'):raise ValueError('Independent registered models only')
        # IO and metrics stay outside this function. It includes Encoder,
        # candidate/condition preparation, all Flow steps and native decode.
        torch.cuda.synchronize();begin=time.perf_counter()
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)];ev[0].record()
        with torch.autocast('cuda',dtype=torch.float16):
            pyramid=self.teacher.model.encoder([xyz]);ev[1].record()
            seed=scene_seed(sequence,frame,self.system.policy['candidates']['seed'])
            generator=torch.Generator(device=self.device).manual_seed(seed)
            observed=self.teacher.model.geometry.observed_mask(xyz)
            source=torch.randn(self.valid.shape,device=self.device,generator=generator)*self.valid
            s1_nfe=nfe if self.cfg['arm']=='S1-MF20K' else 2
            if self.cfg['arm']=='S1-MF20K':
                selected=self.mf if variant=='meanflow' else self.fm
                if variant=='meanflow':u=lambda x,t,s:selected(x,observed,pyramid,t,time_end=s,validate_times=False)
                else:u=lambda x,t,s:selected(x,observed,pyramid,t)
            else:u=lambda x,t,s:self.teacher.model.flow(x,observed,pyramid,t)
            endpoint=sample_meanflow(u,source,self.valid,nfe=s1_nfe,scenes=1);score=stage1_score(endpoint)*self.valid;ev[2].record()
            policy=copy.deepcopy(self.system.policy['candidates']);policy['stage1_nfe']=s1_nfe
            ids,condition,provenance=input_conditions(xyz,self.g,EncodedInput(pyramid,score[0,0],self.teacher.provenance),policy,seed=seed)
            valid=condition['valid_child_mask']
            # All shared absolute Parent IDs receive identical Gaussian values,
            # even when Stage1 MF changes candidate membership or ordering.
            table=torch.randn((self.g.num_parents,self.g.k),device=self.device,
                generator=torch.Generator(device=self.device).manual_seed(scene_seed(sequence,frame,self.cfg['validation']['seed'])))
            source=table[ids]*valid;del table
            s2_nfe=nfe if self.cfg['arm']=='S2-MF20K' else 4
            selected=(self.mf if variant=='meanflow' else self.fm) if self.cfg['arm']=='S2-MF20K' else self.fine
            context=selected.prepare_inference_context(ids,**condition)
            if self.cfg['arm']=='S2-MF20K' and variant=='meanflow':
                u=lambda x,t,s:selected.forward_with_inference_context(context,x,t,time_end=s)
            else:u=lambda x,t,s:selected.forward_with_inference_context(context,x,t)
            endpoint=sample_meanflow(u,source,valid,nfe=s2_nfe,scenes=1);occupied=(endpoint>0)&valid;ev[3].record()
        ids_cpu=ids.cpu().numpy();occupied_cpu=occupied.cpu().numpy()
        fine=occupied_fine_coords(ids_cpu,pack_children(occupied_cpu,self.g),self.g)
        points=self.g.centers(fine,fine=True).astype(np.float32)
        torch.cuda.synchronize();wall=time.perf_counter()-begin
        timing=dict(encoder_ms=ev[0].elapsed_time(ev[1]),stage1_flow_ms=ev[1].elapsed_time(ev[2]),
            candidate_condition_stage2_ms=ev[2].elapsed_time(ev[3]),network_and_preparation_ms=ev[0].elapsed_time(ev[3]),
            generation_wall_ms=wall*1000,scope='same PyTorch original AMP; includes candidate/condition/native-center decode; excludes input IO/metrics')
        return dict(score=score,ids=ids_cpu,occupied=occupied_cpu,points=points,
            parent_ids_sha256=hashlib.sha256(ids_cpu.tobytes()).hexdigest(),timing=timing,
            stage1_nfe=s1_nfe,stage2_nfe=s2_nfe)
