"""Fixed development NFE evaluation; reuse one condition encoding per scene."""
import time
import torch
from .storage import SparseShardDataset,atomic_json
from .dataset import assemble_scene,input_xyz
from .voxelization import voxelize_surface
from .evaluation import physical_support_ids
from pathlib import Path
from .inference import sample
from .evaluation import fine_metrics,aggregate_frames


@torch.no_grad()
def evaluate_model(model,config,geometry,device,encoder,*,max_frames=None):
    was_training=model.training; model.eval()
    try:
        dataset=SparseShardDataset(config['development_dataset'],geometry,verify=True)
        budget=min(len(dataset),max_frames or config['validation'].get('max_frames',len(dataset)))
        nfes=(1,) if model.objective=='bce' else tuple(config['validation']['nfes'])
        rows={str(nfe):[] for nfe in nfes}
        raw_source=None
        if config['validation'].get('common_raw_gt'):
            from .raw_gt import RawMapSource
            raw_source=RawMapSource(config['data_root'])
        for index in range(budget):
            record=dataset[index]; start=time.perf_counter()
            ids,condition,_,source=assemble_scene(record,config,geometry,device,encoder,config['validation']['seed']+index)
            if device.type=='cuda': torch.cuda.synchronize(device)
            condition_seconds=time.perf_counter()-start
            partial=input_xyz(Path(config['data_root'])/record['sequence']/'input_'/f"{record['frame']:06d}.npy")
            observed_ids,observed_masks,_=voxelize_surface(partial,geometry)
            common_gt=None
            if raw_source is not None:
                raw=raw_source.load(record['sequence'],record['frame']); _,roi=geometry.voxelize_points(raw)
                common_gt=physical_support_ids(raw[roi])
            for nfe in nfes:
                generator=torch.Generator(device=device).manual_seed(config['validation']['seed']+index)
                start=time.perf_counter()
                endpoint,occupancy,calls=sample(model,ids,condition,nfe=nfe,generator=generator,threshold=config['validation']['threshold'])
                if device.type=='cuda': torch.cuda.synchronize(device)
                inference_seconds=time.perf_counter()-start
                metrics=fine_metrics(ids.cpu().numpy(),occupancy.cpu().numpy(),record['parent_ids'],record['child_occ_mask'],geometry,
                    observed_parent_ids=observed_ids,observed_masks=observed_masks,common_raw_gt_ids=common_gt,
                    structural=config['validation'].get('structural_metrics',True))
                rows[str(nfe)].append({'sequence':record['sequence'],'frame':record['frame'],**metrics,'actual_network_calls':calls,
                    'condition_seconds':condition_seconds,'stage2_seconds':inference_seconds,'candidate_source':source})
        return {'scope':config.get('scope','pilot'),'candidate_policy':config['candidates'],'gt_is_model_condition':False,
            'oracle_support_used':config['candidates']['mode']=='oracle','objective':model.objective,'threshold':config['validation']['threshold'],
            'threshold_calibration':config['validation'].get('threshold_calibration','uncalibrated_default'),
            'geometry_sha256':geometry.sha256,
            'nfe_results':{key:{'summary':aggregate_frames(value),'rows':value} for key,value in rows.items()}}
    finally: model.train(was_training)
