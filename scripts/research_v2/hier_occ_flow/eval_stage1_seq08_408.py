"""Stage1 seq08/skip10: unchanged Parent-centre points against full raw-map GT."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
import numpy as np
import torch
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
from unified_pointcloud_metrics import evaluate_prediction

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-frames',type=int,default=408);a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Only physical GPU2 is registered')
 if not 1<=a.max_frames<=408:raise ValueError('Expected 1..408 frames')
 cfg=json.loads(a.config.read_text());g=SparseHierarchyGeometry(**cfg['geometry']);m=cfg['model'];frames=list(range(0,4071,10))[:a.max_frames]
 ck=torch.load(a.checkpoint,map_location='cpu');weights=ck.get('ema')
 if not isinstance(weights,dict):raise ValueError('Checkpoint has no EMA state')
 exposure=ck.get('sampler',{}).get('successful_frame_exposures');steps=ck.get('successful_updates')
 if (exposure,steps) not in ((20000,10000),(100000,50000)):raise ValueError(f'Unexpected training budget {(exposure,steps)}')
 if ck.get('initialization_mode')!='scratch':raise ValueError('Scratch checkpoint required')
 torch.set_num_threads(4);torch.cuda.set_device(0)
 model=Stage1DirectOccupancy(g,objective='fm',widths=m['widths'],activation_checkpointing=False,encoder_variant=m['encoder_variant']).cuda()
 model.load_state_dict(weights,strict=True);model.eval().requires_grad_(False);precision=apply_explicit_policy(model);del ck,weights
 protocol={'frames':frames,'sequence':'08','stride':10,'nfe':2,'threshold':.5,'variant':m['encoder_variant'],'checkpoint_sha256':file_sha(a.checkpoint),'config_sha256':file_sha(a.config),'successful_exposures':exposure,'successful_updates':steps,'weights':'EMA','network_precision':precision['mode'],'geometry_and_euler':'original FP32','point_mode':'Parent voxel centers; no resampling','gt':'RawMapSource fast_viewpoint, all original points in frozen ROI','seed':'scene_seed(08,frame,20261031)','metrics':'exact GPU KD-tree CD; physical IoU; audited sparse JSD','code_sha256':file_sha(Path(__file__))}
 a.output.mkdir(parents=True,exist_ok=True);protocol_path=a.output/'protocol.json';journal=a.output/'per_frame.jsonl'
 if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Existing protocol differs')
 atomic_json(protocol_path,protocol);rows=[json.loads(x) for x in journal.read_text().splitlines()] if journal.exists() else [];done={r['frame'] for r in rows}
 if len(done)!=len(rows) or not done.issubset(frames):raise ValueError('Inconsistent existing rows')
 raw=RawMapSource(cfg['data_root'],fast_viewpoint=True);metric=GPUKDTreeMetrics(('cd',),geometry=g)
 with torch.no_grad(),journal.open('a',buffering=1) as log:
  for frame in frames:
   if frame in done:continue
   start=time.perf_counter();path=Path(cfg['data_root'])/'08/input_'/f'{frame:06d}.npy';xyz=torch.from_numpy(input_xyz(path)).cuda();gt=raw.load('08',frame);_,valid=g.voxelize_points(gt);target=np.ascontiguousarray(gt[valid],dtype=np.float32)
   with torch.autocast('cuda',dtype=torch.float16):
    score,_=model.sample(xyz,nfe=2,generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',frame,20261031)))
    ids=torch.nonzero(score[0,0].reshape(-1)>.5).flatten();points=g.centers(g.unflatten(ids)).float().cpu().numpy()
   if not len(points):raise ValueError(f'Empty prediction frame {frame}')
   row={'frame':frame,'elapsed_seconds':time.perf_counter()-start,**evaluate_prediction(points,target,g,metric,device='cuda')};log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row);done.add(frame)
   atomic_json(a.output/'progress.json',{'status':'RUNNING','completed':len(rows),'total':len(frames),'last_frame':frame});print(json.dumps(row,allow_nan=False),flush=True)
 if len(rows)!=len(frames):raise ValueError('Incomplete frame set')
 keys=('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d','predicted_points','gt_points');summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
 with (a.output/'per_frame.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 atomic_json(a.output/'results.json',{'status':'COMPLETE','protocol':protocol,'summary':summary,'rows':rows});atomic_json(a.output/'progress.json',{'status':'COMPLETE','completed':len(rows),'total':len(frames)})

if __name__=='__main__':main()
