#!/usr/bin/env python3
"""Read-only fixed-seed numerical diagnosis, not headline evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fpsgen.datasets.dataloader.semantic_kitti import TemporalKITTISet
from fpsgen.models.gen_img import FlowIMG
from fpsgen.utils.collations import SparseSegmentCollationGen


def stats(tensor):
    finite = torch.isfinite(tensor)
    valid = tensor[finite].float()
    return dict(dtype=str(tensor.dtype), nonfinite=int((~finite).sum()),
                max_abs=float(valid.abs().max()) if valid.numel() else None)


@torch.no_grad()
def trace(model, raw, *, dense, precision, cfg=1, hooks=False, seed=42,
          steps=10, layout=None):
    generator = torch.Generator(device=raw.device).manual_seed(seed)
    batch = raw.shape[0]
    xt = torch.randn((batch,3,256,256),device=raw.device,generator=generator)
    if layout is None:
        layout = torch.zeros((batch,2,256,256),device=raw.device)
    records=[];invalid=[];handles=[]
    if hooks:
        def make_hook(name):
            def check(module, inputs, output):
                tensors=output if isinstance(output,(tuple,list)) else (output,)
                for value in tensors:
                    if isinstance(value,torch.Tensor) and not torch.isfinite(value).all():
                        invalid.append(dict(module=name,type=module.__class__.__name__,output=stats(value)))
            return check
        for name,module in model.model.named_modules():
            if name: handles.append(module.register_forward_hook(make_hook(name)))
    try:
        for step in range(steps):
            t=torch.full((batch,),step/steps,device=raw.device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=precision=='fp16'):
                if cfg==1:
                    velocity=dense(xt,t,raw,layout)
                else:
                    values=dense(torch.cat([xt,xt]),torch.cat([t,t]),
                                 torch.cat([torch.zeros_like(raw),raw]),torch.cat([layout,layout]))
                    unconditional,conditional=values.chunk(2)
                    velocity=unconditional+cfg*(conditional-unconditional)
            xt=xt+(1.0/steps)*velocity
            record=dict(step=step,t=step/steps,velocity=stats(velocity),state=stats(xt))
            records.append(record)
            if record['state']['nonfinite']:break
    finally:
        for h in handles:h.remove()
    return dict(finite=all(r['state']['nonfinite']==0 for r in records),steps=records,
                first_invalid_modules=invalid[:15])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3':raise ValueError('GPU3 only')
    args.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu');cfg=copy.deepcopy(state['hyper_parameters'])
    cfg.setdefault('runtime',{})['compile_dense']=False
    model=FlowIMG(cfg);model.load_state_dict(state['state_dict'],strict=True);del state
    model.cuda().eval();compiled=torch.compile(model.model.forward,backend='inductor')
    dataset=TemporalKITTISet('/data-12/M2024-HWZ/KITTI_Odometry',['08'],'test',.05,180000,50.,gt_dir='gt_possion')
    loader=DataLoader(Subset(dataset,range(100)),batch_size=1,num_workers=0,collate_fn=SparseSegmentCollationGen())
    result=dict(checkpoint=str(args.checkpoint),generation_seed=42,cfg=1,checked=[],failure=None)
    with torch.no_grad():
        for index,batch in enumerate(loader):
            part=batch['pcd_part'].cuda().float();raw=model.model.get_raw_pc_bev(part)
            frame=Path(dataset.points_datapath[index]).stem
            baseline=trace(model,raw,dense=compiled,precision='fp16')
            result['checked'].append(dict(frame=frame,finite=baseline['finite']))
            print('CHECK',frame,'fp16_inductor_finite',baseline['finite'],flush=True)
            if not baseline['finite']:
                failure=dict(frame=frame,part=stats(part),raw_condition=stats(raw),fp16_inductor=baseline)
                for precision in ['fp16','fp32']:
                    failure[precision+'_eager']=trace(model,raw,dense=model.model.forward,precision=precision,hooks=True)
                failure['fp32_inductor']=trace(model,raw,dense=compiled,precision='fp32')
                failure['fp16_inductor_cfg2']=trace(model,raw,dense=compiled,precision='fp16',cfg=2)
                result['failure']=failure
                break
    (args.output/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DIAGNOSIS',json.dumps(result),flush=True)


if __name__=='__main__':main()
