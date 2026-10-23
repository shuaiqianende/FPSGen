from common import ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,config_sha,source_sha
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel as HierarchicalOccupancyModel
from fpsgen.hier_occ_flow.stage2_evaluation import evaluate_stage2
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ema',action='store_true')
    p.add_argument('--visualization-output',type=Path);p.add_argument('--max-frames',type=int,
        help='Limit diagnostic evaluation after exact training-config identity validation')
    p.add_argument('--sampling-seeds',type=int,nargs='+',
        help='Use a paired diagnostic RNG subset after exact training-config identity validation')
    p.add_argument('--nfes',type=int,nargs='+',choices=(1,2,4,8))
    p.add_argument('--continuous-metrics',action='store_true',help='Raw-map CD, corrected physical IoU and FPSGen JSD')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous evaluation output')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('1','2','3'):raise ValueError('Select only authorized physical GPU1/2/3')
    admission=gpu_preflight((int(visible),))
    if admission['status']!='AVAILABLE':atomic_json(args.output,{'status':'BLOCKED','gpu':admission});return
    cfg,g=load_stage2_config(args.config);_,dev,manifests=dataset_identity(cfg,g,verify=True)
    torch.set_num_threads(4)
    saved=torch.load(args.checkpoint,map_location='cpu');identity=saved['identity']
    if identity['experiment']!='hs-occflow-stage2-384-v1' or identity['geometry_sha256']!=g.sha256 or identity['config_sha256']!=config_sha(cfg) or any(identity[k]!=v for k,v in manifests.items()):
        raise ValueError('Stage2 checkpoint/config/data identity mismatch')
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).cuda().eval()
    model=HierarchicalOccupancyModel(g,**cfg['model']).cuda().eval()
    state=saved['ema'] if args.ema else saved['model']
    if state is None:raise ValueError('Requested EMA absent')
    model.load_state_dict(state,strict=True)
    if args.max_frames is not None:
        if args.max_frames<=0:raise ValueError('--max-frames must be positive')
        cfg['validation']['max_frames']=args.max_frames
    if args.sampling_seeds is not None:
        if not args.sampling_seeds:raise ValueError('At least one sampling seed is required')
        cfg['validation']['sampling_seeds']=args.sampling_seeds
    if args.nfes is not None:cfg['validation']['nfes']=args.nfes
    continuous=None
    if args.continuous_metrics:
        from fpsgen.hier_occ_flow.raw_gt import RawMapSource
        from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords,pack_children
        from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
        sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
        from unified_pointcloud_metrics import evaluate_prediction
        raw=RawMapSource(cfg['data_root'],fast_viewpoint=True)
        metric=GPUKDTreeMetrics(('cd',),geometry=g)
        cached={}
        def continuous(record,ids,occupancy):
            key=(record['sequence'],int(record['frame']))
            if cached.get('key')!=key:
                target=raw.load(*key);_,valid=g.voxelize_points(target)
                target=np.ascontiguousarray(target[valid],dtype=np.float32)
                cached.update(key=key,target=target,sha256=hashlib.sha256(target.tobytes()).hexdigest())
            fine=occupied_fine_coords(ids.cpu().numpy(),pack_children(occupancy.cpu().numpy(),g),g)
            points=g.centers(fine,fine=True).astype(np.float32)
            if not len(points):raise ValueError('Empty native occupancy; report failure instead of dropping frame')
            return {**evaluate_prediction(points,cached['target'],g,metric,device='cuda'),
                    'raw_gt_xyz_sha256':cached['sha256'],'point_mode':'native_fine_voxel_centers'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output.with_suffix('.protocol.json'),{'training_identity':identity,'evaluation_source_sha256':source_sha(),
        'validation':cfg['validation'],'continuous_metrics':args.continuous_metrics,'gpu_admission':admission,
        'checkpoint_sha256':file_sha(args.checkpoint),'input_manifest_sha256':manifests})
    with args.output.with_suffix('.jsonl').open('x',buffering=1) as stream:
        def progress(nfe,row):stream.write(json.dumps({'nfe':nfe,**row})+'\n')
        result=evaluate_stage2(model,cfg,g,torch.device('cuda'),adapter,dev,visualization_root=args.visualization_output,
                              continuous_metrics=continuous,progress_callback=progress)
    result.update(successful_updates=saved['successful_updates'],checkpoint_sha256=file_sha(args.checkpoint),ema=args.ema,
                  training_source_sha256=identity['source_sha256'],evaluation_source_sha256=source_sha())
    atomic_json(args.output,result)


if __name__=='__main__':main()
