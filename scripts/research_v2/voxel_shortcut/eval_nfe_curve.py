"""Same frozen base/source/seed for each NFE; development and strict manifests."""

import json
from pathlib import Path
import statistics
import time
import torch
from common import parser,read_config,frames_for,select_spaced,runtime,save_json,load_base,sha_file
from fpsgen.voxel_shortcut.dataset import ShortcutDataset
from fpsgen.voxel_shortcut.count_mapping import CountMapping
from fpsgen.voxel_shortcut.source_builder import build_source
from fpsgen.voxel_shortcut.shortcut_unet import ShortcutUNet
from fpsgen.voxel_shortcut.shortcut_flow import rollout,path_diagnostics
from fpsgen.voxel_shortcut.evaluator import field_metrics,endpoint_density,anchor_metrics
from fpsgen.voxel_shortcut.count_allocator import allocate_anchors


def main():
    p=parser(__doc__);p.add_argument('--split',choices=('development','test'),default='development')
    p.add_argument('--anchors',action='store_true');p.add_argument('--nfe',type=int,nargs='+');args=p.parse_args()
    cfg,spec,sha=read_config(args.config);device,rank,world=runtime()
    if world!=1:raise ValueError('Use explicit manifests for independent evaluations')
    if not cfg['source_calibration']['frozen']:raise ValueError('Freeze source gate on development before NFE comparison')
    base,base_sha,base_ckpt=load_base(cfg,spec,device)
    model=ShortcutUNet(**cfg['network']).to(device).eval()
    checkpoint=torch.load(cfg['flow_checkpoint'],map_location='cpu')
    if checkpoint['geometry']!=spec.to_dict() or checkpoint['base_checkpoint_sha256']!=base_sha or checkpoint['mapping']!=cfg['mapping'] or checkpoint['source']!=cfg['source']:
        raise ValueError('Flow checkpoint vs base/mapping/source mismatch')
    stage=checkpoint['stage'];shortcut=stage in ('shortcut','surface')
    model.load_state_dict(checkpoint['ema'] if checkpoint['ema'] is not None else checkpoint['model'],strict=True)
    if args.split=='test' and any(c['pilot_only'] or c['mechanics_smoke_only'] for c in (base_ckpt,checkpoint)):
        raise ValueError('Pilot/smoke cannot claim formal seq08 results')
    dataset=ShortcutDataset(select_spaced(frames_for(cfg,args.split),args.max_frames),spec,**cfg['data'].get('gt_validation',{}))
    nfes=args.nfe or ([1] if stage=='residual' else cfg['evaluation']['nfe'])
    valid=spec.field_mask(device=device);mapping=CountMapping(**cfg['mapping']);rows=[];source_rows=[]
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for i in range(len(dataset)):
            sample=dataset[i];gt=sample['gt_counts'][None].to(device);known=sample['observed'][None].to(device)
            frame=sample['frame'];seed=cfg['evaluation']['source_seed']+int(frame['sequence'])*1000000+frame['frame']
            torch.cuda.synchronize(device);base_started=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['precision']=='fp16'):
                frozen=base(sample['input_xyz'].to(device))
            torch.cuda.synchronize(device);base_ms=(time.perf_counter()-base_started)*1000
            source_started=time.perf_counter()
            source,info=build_source(frozen['density'],frozen['occupancy_probability'],known,valid,mapping,
                generator=torch.Generator(device=device).manual_seed(seed),**cfg['source'])
            torch.cuda.synchronize(device);source_ms=(time.perf_counter()-source_started)*1000
            condition=frozen['condition'];source_density,_,_=endpoint_density(source,mapping,valid)
            source_row={'frame':frame,'seed':seed,'density_metrics':field_metrics(source_density,gt,spec,observed=known,threshold=cfg['evaluation']['occupancy_threshold'])}
            if args.anchors:
                source_bridge=allocate_anchors(source_density.cpu(),spec,generator=torch.Generator().manual_seed(seed))
                source_row['anchor_metrics']=anchor_metrics(source_bridge['anchors'].to(device),sample['gt_xyz'].to(device),spec,backend=cfg['runtime']['knn_backend'])
            source_rows.append(source_row)
            endpoints={};scene=[]
            for nfe in [0]+list(nfes):
                torch.cuda.synchronize(device);flow_started=time.perf_counter()
                if nfe==0:
                    density=info['base_density'];raw=frozen['density'];diagnostics={};actual=0
                else:
                    with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['precision']=='fp16'):
                        if stage=='residual':
                            endpoint=source+model(source,0.,1.,condition)*valid;detail={'actual_nfe':1,'path':None}
                        else:
                            endpoint,detail=rollout(model,source,condition,nfe=nfe,shortcut=shortcut,
                                return_path=nfe>=16,valid=valid)
                    actual=detail['actual_nfe'];endpoints[nfe]=endpoint
                    density,raw,diagnostics=endpoint_density(endpoint,mapping,valid)
                    if detail['path'] is not None:diagnostics.update(path_diagnostics(detail['path']))
                torch.cuda.synchronize(device);flow_ms=(time.perf_counter()-flow_started)*1000
                metrics=field_metrics(density,gt,spec,observed=known,threshold=cfg['evaluation']['occupancy_threshold'])
                raw_mass=float(raw.sum());metrics['raw_mass_calibration_error']=abs(raw_mass-float(gt.sum()))/float(gt.sum())
                row={'frame':frame,'gt_validation':sample['gt_validation'],'seed':seed,'method':'base' if nfe==0 else stage,'requested_nfe':nfe,'actual_nfe':actual,
                    'base_encodings':1,'base_ms':base_ms,'source_ms':source_ms,'flow_and_decode_ms':flow_ms,'density_metrics':metrics,
                    'raw_density_mass':raw_mass,'rollout_diagnostics':diagnostics}
                if args.anchors:
                    anchor_started=time.perf_counter()
                    bridge=allocate_anchors(density.cpu(),spec,generator=torch.Generator().manual_seed(seed))
                    row['anchor_allocation_ms']=(time.perf_counter()-anchor_started)*1000
                    row['anchor_metrics']=anchor_metrics(bridge['anchors'].to(device),sample['gt_xyz'].to(device),spec,backend=cfg['runtime']['knn_backend'])
                scene.append(row)
                print(f'{stage} {frame["sequence"]}/{frame["frame"]:06d} NFE{actual} massTV{metrics["mass_tv"]:.4f} F1{metrics["occupancy_f1"]:.4f}',flush=True)
            if 16 in endpoints:
                for row in scene:
                    n=row['requested_nfe']
                    if n in endpoints:
                        row['rollout_diagnostics']['mapped_endpoint_rmse_to_nfe16']=float((((endpoints[n]-endpoints[16]).square()*valid).sum()/valid.sum()).sqrt())
                        row['rollout_diagnostics']['mapped_endpoint_gt_rmse']=float((((endpoints[n]-mapping.encode(gt)).square()*valid).sum()/valid.sum()).sqrt())
            rows.extend(scene)
            save_json(args.output/'nfe_curve.partial.json',{'status':'RUNNING','rows':rows})
    means={}
    for n in [0]+list(nfes):
        subset=[r for r in rows if r['requested_nfe']==n]
        means[str(n)]={'density_metrics':{key:statistics.mean(r['density_metrics'][key] for r in subset) for key in subset[0]['density_metrics']},
            'base_ms':statistics.mean(r['base_ms'] for r in subset),'source_ms':statistics.mean(r['source_ms'] for r in subset),
            'flow_and_decode_ms':statistics.mean(r['flow_and_decode_ms'] for r in subset)}
        if len(subset)>1:
            means[str(n)]['warm_base_ms']=statistics.mean(r['base_ms'] for r in subset[1:])
            means[str(n)]['warm_flow_and_decode_ms']=statistics.mean(r['flow_and_decode_ms'] for r in subset[1:])
        if args.anchors:means[str(n)]['anchor_metrics']={key:statistics.mean(r['anchor_metrics'][key] for r in subset) for key in subset[0]['anchor_metrics']}
    save_json(args.output/'nfe_curve.json',{'status':'COMPLETE','stage':stage,'config_sha256':sha,'base_sha256':base_sha,
        'flow_sha256':sha_file(cfg['flow_checkpoint']),'source':cfg['source'],'split':args.split,'frames':len(dataset),'means':means,'rows':rows,'source_rows':source_rows,
        'base_checkpoint_pilot_only':base_ckpt['pilot_only'],'flow_pilot_only':checkpoint['pilot_only'],
        'formal_quality_claim':not any(c['pilot_only'] or c['mechanics_smoke_only'] for c in (base_ckpt,checkpoint)),
        'physical_visible':__import__('os').environ['CUDA_VISIBLE_DEVICES'],'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3})


if __name__=='__main__':main()
