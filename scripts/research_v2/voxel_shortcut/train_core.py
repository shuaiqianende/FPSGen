"""Staged base/CFM/shortcut/surface/residual training, pilot guard, finite AMP/DDP."""

import json
import math
from pathlib import Path
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from common import parser,read_config,frames_for,select_spaced,runtime,save_json,load_base,sha_file
from fpsgen.voxel_shortcut.dataset import ShortcutDataset
from fpsgen.voxel_shortcut.base_density import BaseDensity,base_loss
from fpsgen.voxel_shortcut.count_mapping import CountMapping
from fpsgen.voxel_shortcut.source_builder import build_source
from fpsgen.voxel_shortcut.shortcut_unet import ShortcutUNet
from fpsgen.voxel_shortcut.shortcut_flow import EMA,linear_pair,bootstrap_target,rollout
from fpsgen.voxel_shortcut.shortcut_losses import masked_mse,surface_loss


def main(stage):
    args=parser(__doc__,training=True).parse_args();cfg,spec,sha=read_config(args.config)
    steps=args.max_steps or cfg['train']['updates']
    if steps<1:raise ValueError('Training updates must be positive')
    if cfg['runtime']['batch_per_gpu']!=1:raise ValueError('This launcher currently implements B1/GPU only')
    if stage!='base' and not cfg['source_calibration'].get('frozen',False):
        raise ValueError('Freeze development support calibration before flow training')
    if args.smoke and steps>16 or args.pilot and steps>512:
        raise ValueError("Smoke<=16, pilot<=512 successful updates")
    if not args.smoke and not args.pilot:
        if not args.gate_report or json.loads(args.gate_report.read_text()).get(stage)!='READY':
            raise ValueError("Formal training requires explicit passing stage gate report")
    device,rank,world=runtime();torch.manual_seed(cfg['seed'])
    precision=args.precision or cfg['runtime']['precision'];amp=precision=='fp16'
    frames=select_spaced(frames_for(cfg,'train'),args.max_frames)
    # A rejected scene is never silently substituted; audit the manifest first.
    dataset=ShortcutDataset(frames,spec,**cfg['data'].get('gt_validation',{}))
    base_sha=None
    if stage=='base':
        raw=BaseDensity(spec,activation_checkpointing=cfg['runtime']['activation_checkpointing']).to(device)
        raw.initialize_density_checkpoint(cfg['initialize_density_checkpoint'])
    else:
        base,base_sha,base_checkpoint=load_base(cfg,spec,device)
        if not (args.smoke or args.pilot) and (base_checkpoint['pilot_only'] or base_checkpoint['mechanics_smoke_only']):
            raise ValueError('Formal flow training requires a qualified formal base, not a pilot/smoke checkpoint')
        raw=ShortcutUNet(**cfg['network']).to(device)
    if args.initialize:
        initial=torch.load(args.initialize,map_location='cpu')
        if initial['geometry']!=spec.to_dict() or initial['mapping']!=cfg.get('mapping'):
            raise ValueError("Initialization mapping/geometry mismatch")
        if stage!='base' and (initial['base_checkpoint_sha256']!=base_sha or initial['source']!=cfg['source']):
            raise ValueError("Shortcut/surface must keep identical frozen base/source")
        raw.load_state_dict(initial['model'],strict=True)
    model=DDP(raw,device_ids=[device.index]) if world>1 else raw
    model.train();ema=EMA(raw,decay=cfg['train']['ema_decay']).to(device) if stage in ('shortcut','surface') else None
    if ema is not None and args.initialize and 'ema' in initial and initial['ema'] is not None:
        ema.model.load_state_dict(initial['ema'],strict=True)
    mapping=CountMapping(**cfg['mapping']) if stage!='base' else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['train']['lr'],weight_decay=cfg['train']['weight_decay'])
    scaler=torch.cuda.amp.GradScaler(enabled=amp,init_scale=cfg['runtime'].get('initial_amp_scale',4096.))
    successful=0;attempts=0;skipped=0;records=[];started=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=True)
    if rank==0:
        save_json(args.output/'config_at_run.json',cfg)
        save_json(args.output/'frames_at_run.json',frames)
        save_json(args.output/'code_provenance.json',{
            str(p.relative_to(__import__('common').ROOT)):sha_file(p)
            for folder in ('fpsgen/voxel_shortcut','scripts/research_v2/voxel_shortcut')
            for p in (__import__('common').ROOT/folder).glob('*.py')})
    valid=spec.field_mask(device=device)
    while successful<steps:
        frame_index=(attempts*world+rank)%len(dataset)
        sample=dataset[frame_index];xyz=sample['input_xyz'].to(device);gt=sample['gt_counts'][None].to(device)
        optimizer.zero_grad(set_to_none=True);step_started=time.perf_counter()
        generator=torch.Generator(device=device).manual_seed(cfg['seed']+attempts*world+rank)
        values={};off_path=False;half=None;t_sc=None;endpoint_t=None
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            if stage=='base':
                output=model(xyz);terms=base_loss(output,gt,spec,**cfg['base_loss']);loss=terms['total']
                values={key:float(value.detach()) for key,value in terms.items()}
            else:
                with torch.no_grad():
                    frozen=base(xyz)
                    source,source_info=build_source(frozen['density'],frozen['occupancy_probability'],sample['observed'][None].to(device),valid,mapping,generator=generator,**cfg['source'])
                    condition=tuple(c.detach() for c in frozen['condition']);target=mapping.encode(gt)
                t=torch.rand((),device=device,generator=generator)
                state,velocity=linear_pair(source,target,t)
                if stage=='residual':
                    predicted=model(source,0.,1.,condition)
                    loss=masked_mse(predicted,target-source,valid)
                    values={'residual_mse':float(loss.detach())}
                else:
                    predicted=model(state,t,0.,condition,instantaneous=True)
                    fm=masked_mse(predicted,velocity,valid);loss=cfg['loss']['fm_weight']*fm
                    values={'fm':float(fm.detach())}
                    if ema is not None:
                        # Anchor the smallest positive step to analytical FM.
                        # h=0 alone would leave EMA's positive-h leaf untrained.
                        leaf_h=cfg['train']['small_step_anchor']
                        leaf_t=float(torch.randint(round(1/leaf_h),(),device=device,generator=generator))*leaf_h
                        leaf_state,_=linear_pair(source,target,leaf_t)
                        leaf_loss=masked_mse(model(leaf_state,leaf_t,leaf_h,condition),velocity,valid)
                        loss=loss+cfg['loss']['small_step_anchor_weight']*leaf_loss
                        values['small_step_anchor']=float(leaf_loss.detach())
                    if ema is not None and successful>=cfg['train']['shortcut_after']:
                        half_steps=cfg['train']['bootstrap_half_steps']
                        half=float(half_steps[int(torch.randint(len(half_steps),(),device=device,generator=generator))])
                        positions=round((1-2*half)/half)+1
                        t_sc=float(torch.randint(positions,(),device=device,generator=generator))*half
                        sc_state,_=linear_pair(source,target,t_sc)
                        if t_sc>0 and successful>=cfg['train']['off_path_after'] and float(torch.rand((),device=device,generator=generator))<cfg['train']['off_path_fraction']:
                            off_path=True
                            if t_sc:
                                with torch.no_grad():
                                    sc_state=source.clone()
                                    for i in range(round(t_sc/half)):
                                        sc_state=sc_state+half*ema.model(sc_state,i*half,half,condition)*valid
                        boot=bootstrap_target(ema.model,sc_state,t_sc,half,condition,valid=valid)
                        shortcut=masked_mse(model(sc_state,t_sc,2*half,condition),boot,valid)
                        loss=loss+cfg['loss']['shortcut_weight']*shortcut
                        # Half the endpoint examples explicitly see inference(0,1).
                        endpoint_t=0. if attempts%2==0 else float(t)
                        end_state,_=linear_pair(source,target,endpoint_t)
                        endpoint=end_state+(1-endpoint_t)*model(end_state,endpoint_t,1-endpoint_t,condition)*valid
                        end_loss=masked_mse(endpoint,target,valid)
                        loss=loss+cfg['loss']['endpoint_weight']*end_loss
                        values.update(shortcut=float(shortcut.detach()),endpoint=float(end_loss.detach()))
                        if stage=='surface' and successful>=cfg['train']['surface_after']:
                            structural,structure=surface_loss(endpoint,gt,valid,mapping,**cfg['surface'])
                            loss=loss+cfg['loss']['surface_weight']*structural
                            values.update(surface=float(structural.detach()),**{k:float(v.detach()) for k,v in structure.items()})
                values['total']=float(loss.detach())
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in raw.parameters())
        finite_flag=torch.tensor(int(finite),device=device)
        if world>1:torch.distributed.all_reduce(finite_flag,op=torch.distributed.ReduceOp.MIN)
        finite=bool(finite_flag)
        if finite:
            torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['train']['gradient_clip'])
            scaler.step(optimizer);scaler.update();successful+=1
            if ema is not None:ema.update(raw)
        elif amp:
            scaler.update(new_scale=scaler.get_scale()/2);skipped+=1
        else:raise FloatingPointError("Nonfinite FP32 gradient")
        attempts+=1
        if attempts>max(64,steps*4):raise FloatingPointError("Excessive AMP skips")
        torch.cuda.synchronize(device)
        record={'successful_updates':successful,'attempt':attempts,'gradients_finite':finite,'frame':sample['frame'],
                'gt_validation':sample['gt_validation'],
                'source_mode':None if stage=='base' else cfg['source']['mode'],
                'seconds':time.perf_counter()-step_started,'losses':values,'half_step':half,'shortcut_t':t_sc,
                'endpoint_t':endpoint_t,'off_path':off_path,'base_encodings':1 if stage!='base' else None}
        records.append(record)
        with (args.output/f'train_rank{rank}.jsonl').open('a') as file:file.write(json.dumps(record,allow_nan=False)+'\n')
        print(f'rank{rank} {stage} update{successful}/{steps} loss{float(loss.detach()):.6f} finite{finite} sec{record["seconds"]:.2f}',flush=True)
        if stage=='base':del output,terms
        else:
            del frozen,source,source_info,condition,target,state,velocity,predicted
            if ema is not None:del leaf_state,leaf_loss
            if half is not None:del boot,sc_state,end_state,endpoint,shortcut,end_loss
        del loss,gt,xyz
    if rank==0:
        path=args.output/f'checkpoint_{successful:06d}.pt';temporary=path.with_suffix('.pt.tmp')
        torch.save({'stage':'ssvsf_base' if stage=='base' else stage,'geometry':spec.to_dict(),'config_sha256':sha,
            'config':cfg,'model':raw.state_dict(),'ema':ema.model.state_dict() if ema is not None else None,
            'base_checkpoint_sha256':base_sha,'mapping':cfg.get('mapping'),'source':cfg.get('source'),
            'successful_updates':successful,'mechanics_smoke_only':args.smoke,'pilot_only':args.pilot},temporary)
        temporary.replace(path)
    save_json(args.output/f'summary_rank{rank}.json',{'stage':stage,'successful_updates':successful,'attempts':attempts,
        'amp_skipped_attempts':skipped,'successful_gradients_finite':sum(r['gradients_finite'] for r in records)==successful,
        'parameters':sum(p.numel() for p in raw.parameters()),'config_sha256':sha,'base_checkpoint_sha256':base_sha,
        'physical_visible':__import__('os').environ['CUDA_VISIBLE_DEVICES'],'precision':precision,'world_size':world,
        'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3,'seconds':time.perf_counter()-started,
        'first_loss':records[0]['losses'],'last_loss':records[-1]['losses'],'pilot_only':args.pilot,'mechanics_smoke_only':args.smoke})
    if world>1:torch.distributed.barrier();torch.distributed.destroy_process_group()
