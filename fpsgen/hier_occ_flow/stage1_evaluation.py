"""Parent support metrics and fine-recall ceiling; metrics never enter training."""
from pathlib import Path
import time
import numpy as np
import torch
from .storage import SparseShardDataset
from .dataset import input_xyz


def parent_metrics(score,record,geometry,valid,threshold=.5):
    score=score.detach().float().cpu().numpy() if isinstance(score,torch.Tensor) else score
    valid=valid.cpu().numpy() if isinstance(valid,torch.Tensor) else valid
    predicted=(score.ravel()>threshold)&valid.ravel()
    truth=record['parent_ids'].astype(np.int64)
    selected=predicted[truth]
    total=int(predicted.sum());tp=int(selected.sum());fp=total-tp;fn=len(truth)-tp
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1)
    masks=record['child_occ_mask']; child_counts=np.stack([(masks>>i)&1 for i in range(geometry.k)],1).sum(1)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
        'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1),
        'active_parents':total,'gt_parents':len(truth),'predicted_occupied_ratio':total/max(int(valid.sum()),1),
        'gt_occupied_ratio':len(truth)/max(int(valid.sum()),1),
        'fine_recall_ceiling':float(child_counts[selected].sum()/max(child_counts.sum(),1))}


def summarize(rows):
    tp=sum(r['tp'] for r in rows); fp=sum(r['fp'] for r in rows); fn=sum(r['fn'] for r in rows)
    return {'frames':len(rows),'macro':{k:float(np.mean([r[k] for r in rows])) for k in
        ('precision','recall','f1','iou','active_parents','fine_recall_ceiling')},
        'micro':{'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
                 'f1':2*tp/max(2*tp+fp+fn,1),'iou':tp/max(tp+fp+fn,1)}}


@torch.no_grad()
def evaluate_stage1(model,cfg,geometry,device,*,max_frames=None,sample_output=None):
    was_training=model.training; model.eval()
    try:
        ds=SparseShardDataset(cfg['development_dataset'],geometry,verify=False)
        budget=min(len(ds),max_frames or len(ds)); val=cfg['validation']
        # Spread light checks through the frozen development index, not seq08.
        indices=np.linspace(0,len(ds)-1,budget,dtype=int)
        nfes=val['nfes'] if model.objective=='fm' else [1]
        rows={str(n):[] for n in nfes}; thresholds=val.get('threshold_sweep',[val['threshold']]);valid_cpu=model.valid.cpu().numpy()
        for position,index in enumerate(indices):
            record=ds[int(index)]; path=Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy"
            xyz=torch.from_numpy(input_xyz(path)).to(device)
            start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
                condition=model.encode_condition(xyz)
            torch.cuda.synchronize(device); encoder_seconds=time.perf_counter()-start
            for nfe in nfes:
                generator=torch.Generator(device=device).manual_seed(val['seed']+int(index))
                tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['training']['precision']=='fp16'):
                    score,state=model.sample(xyz,generator=generator,nfe=nfe,condition=condition)
                torch.cuda.synchronize(device)
                if not bool(torch.isfinite(score).all() and torch.isfinite(state).all()):
                    raise RuntimeError('Nonfinite generated occupancy')
                sampling_seconds=time.perf_counter()-tick;metric_start=time.perf_counter();score_cpu=score.float().cpu().numpy()
                metrics=parent_metrics(score_cpu,record,geometry,valid_cpu,val['threshold'])
                sweep={str(t):parent_metrics(score_cpu,record,geometry,valid_cpu,t) for t in thresholds}
                if sample_output is not None and index==indices[0] and nfe==val.get('sample_nfe',4):
                    dest=Path(sample_output);dest.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest/'fixed_sample_occupancy_score.npz',score=score_cpu[0,0])
                    np.save(dest/'fixed_sample_parent_ids.npy',np.flatnonzero((score_cpu.ravel()>val['threshold'])&valid_cpu.ravel()).astype(np.uint32),allow_pickle=False)
                    from .storage import atomic_json
                    atomic_json(dest/'fixed_sample_manifest.json',{'sequence':record['sequence'],'frame':record['frame'],
                        'input':str(path),'geometry_sha256':geometry.sha256,'velocity_nfe':nfe,
                        'prediction_condition':'LiDAR XYZ and seeded Gaussian only','gt_used_for_generation':False,
                        'threshold':val['threshold'],'target_definition':'raw_pre_poisson_surface_hit'})
                rows[str(nfe)].append({'sequence':record['sequence'],'frame':record['frame'],**metrics,
                    'encoder_seconds':encoder_seconds,'sampling_seconds':sampling_seconds,
                    'metrics_seconds':time.perf_counter()-metric_start,
                    'actual_velocity_nfe':nfe if model.objective=='fm' else 0,
                    'classifier_forwards':1 if model.objective=='bce' else 0,
                    'thresholds':sweep})
            if position==0 or (position+1)%10==0 or position+1==len(indices):
                print(f'validation_scenes={position+1}/{len(indices)} objective={model.objective}',flush=True)
        return {'objective':model.objective,'threshold':val['threshold'],'threshold_status':'uncalibrated_development_diagnostic',
            'target':'raw_pre_poisson_surface_hit','geometry_sha256':geometry.sha256,
            'nfe_results':{k:{'summary':summarize(v),'rows':v} for k,v in rows.items()}}
    finally: model.train(was_training)
