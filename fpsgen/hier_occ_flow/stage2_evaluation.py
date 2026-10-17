"""Fine metrics include all missed parents; same immutable raw-map labels."""
import time
from pathlib import Path
import numpy as np
import torch
from .stage2_dataset import assemble_stage2_scene
from .dataset import input_xyz
from .voxelization import voxelize_surface
from .inference import sample
from .evaluation import fine_metrics,aggregate_frames


@torch.no_grad()
def evaluate_stage2(model,cfg,g,device,adapter,dataset,*,indices=None,cache=None,visualization_root=None):
    indices=indices if indices is not None else np.linspace(0,len(dataset)-1,cfg['validation']['max_frames'],dtype=int).tolist()
    was_training=model.training;model.eval();rows={str(n):[] for n in cfg['validation']['nfes'] if model.objective=='fm' or n==1}
    begin=time.perf_counter()
    try:
        for index in indices:
            record=dataset[int(index)];tick=time.perf_counter()
            ids,condition,_,source=assemble_stage2_scene(record,cfg,g,device,adapter,cache)
            if device.type=='cuda':torch.cuda.synchronize(device)
            condition_seconds=time.perf_counter()-tick
            xyz=input_xyz(Path(cfg['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy")
            observed_ids,observed_masks,_=voxelize_surface(xyz,g)
            for nfe in rows:
                for seed in cfg['validation'].get('sampling_seeds',[cfg['validation']['seed']]):
                    generator=torch.Generator(device=device).manual_seed(seed+int(index))
                    tick=time.perf_counter()
                    dtype=torch.bfloat16 if cfg['training']['precision']=='bf16' else torch.float16
                    with torch.autocast(device.type,dtype=dtype,enabled=device.type=='cuda' and cfg['training']['precision']!='fp32'):
                        _,occupancy,calls=sample(model,ids,condition,nfe=int(nfe),generator=generator,threshold=cfg['validation']['threshold'])
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    inference_seconds=time.perf_counter()-tick
                    tick=time.perf_counter()
                    metric=fine_metrics(ids.cpu().numpy(),occupancy.cpu().numpy(),record['parent_ids'],record['child_occ_mask'],g,
                        observed_parent_ids=observed_ids,observed_masks=observed_masks,structural=cfg['validation'].get('structural_metrics',True))
                    gt_positive=metric['fine']['tp']+metric['fine']['fn']
                    candidate_positive=gt_positive-metric['missed_parent_fine_false_negatives']
                    rows[nfe].append({'sequence':record['sequence'],'frame':record['frame'],'seed':seed,**metric,
                        'actual_stage2_nfe':calls,'condition_seconds':condition_seconds,'stage2_seconds':inference_seconds,
                        'metrics_seconds':time.perf_counter()-tick,
                        'valid_child_count':int(condition['valid_child_mask'].sum()),
                        'all_candidate_children_occupied_f1':2*candidate_positive/max(int(condition['valid_child_mask'].sum())+gt_positive,1),
                        'candidate_source':source})
                    # Optional one real scene, first fixed source seed, all NFE.
                    # Large diagnostic PLYs never enter the Git report directory.
                    if visualization_root is not None and index==indices[0] and seed==cfg['validation'].get('sampling_seeds',[cfg['validation']['seed']])[0]:
                        from .stage2_visualization import export_evaluation_diagnostics
                        export_evaluation_diagnostics(Path(visualization_root)/f'{record["sequence"]}_{record["frame"]:06d}_nfe{nfe}',
                            xyz,ids.cpu().numpy(),occupancy.cpu().numpy(),record['parent_ids'],record['child_occ_mask'],g)
            print(f'Stage2 evaluation {record["sequence"]}/{record["frame"]:06d} tokens={len(ids)}',flush=True)
        summaries={key:{'summary':aggregate_frames(value),'rows':value} for key,value in rows.items()}
        return {'geometry_sha256':g.sha256,'scope':cfg['training']['scope'],'candidate_mode':cfg['candidates']['mode'],
            'oracle_is_deployable':False,'gt_is_condition':False,'gt_target':'fixed raw-map fine surface-hit occupancy',
            'stage1':adapter.provenance,'seed_variability':'separate rows per frozen sampling seed',
            'nfe_results':summaries,'seconds':time.perf_counter()-begin}
    finally:model.train(was_training)
