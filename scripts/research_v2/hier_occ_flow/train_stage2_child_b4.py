"""Two independent real-B4 Child-state Stage2 scratch experiments on GPU2/3."""
import argparse,copy,json,math,os,signal,sys,time,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from fpsgen.hier_occ_flow.geometry import SparseHierarchyGeometry
from fpsgen.hier_occ_flow.storage import SparseShardDataset,atomic_json,file_sha
from fpsgen.hier_occ_flow.morphology import MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.voxelization import unpack_children
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_remove_score import remove_score
from fpsgen.hier_occ_flow.stage2_no_lidar import remove_lidar_and_score
from fpsgen.hier_occ_flow.stage2_fp16_migration import migrate_fp32_runtime
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import update_fp32_ema,apply_explicit_policy
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.stage2_runtime import sampler_index
from fpsgen.hier_occ_flow.trainer import seed_all,atomic_checkpoint,rng_state,restore_rng
from fpsgen.hier_occ_flow.static_mask_cache import StaticMaskCache
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
CACHE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-stage2-condition-cache-v1/experiments/stage2_static_masks_full_v1');BASE=ROOT/'experiments/stage2_child_b4_followup_v1'

def code_sha():
 files=[Path(__file__)]+[ROOT/'fpsgen/hier_occ_flow'/n for n in ('stage2_model.py','model.py','stage2_no_lidar.py','stage2_remove_score.py','stage2_fp16_migration.py','explicit_fp16_training_policy.py','static_mask_cache.py','t1_stage1_adapter.py','depth_moment_encoder.py','patch_unet_condition.py','child_local_state.py')]
 h=hashlib.sha256()
 for p in files:h.update(p.name.encode());h.update(p.read_bytes())
 return h.hexdigest()

def data(c,g):
 ds=SparseShardDataset(c['train_dataset'],g,data_version=MORPH_DATA_VERSION);meta=json.loads((CACHE/'status.json').read_text());manifest=file_sha(Path(c['train_dataset'])/'manifest.json')
 if len(ds)!=19130 or meta.get('status')!='COMPLETE' or meta.get('source_manifest_sha256')!=manifest or {r['sequence'].decode() for r in ds.index}!={f'{i:02}' for i in (0,1,2,3,4,5,6,7,9,10)}:raise ValueError('Full train/cache identity mismatch')
 return ds,manifest

def no_encoder_scene(record,g,manifest):
 path=CACHE/'frames'/record['sequence']/f"{record['frame']:06d}.npz"
 with np.load(path,allow_pickle=False) as z:
  ids=z['ids'];targets=z['target']
  if str(z['manifest_sha'])!=manifest or not np.array_equal(ids,record['parent_ids']) or not np.array_equal(targets,record['child_occ_mask']):raise ValueError('Cache mismatch')
  obs=unpack_children(z['observed'],g);valid=unpack_children(z['valid'],g);target=unpack_children(targets,g)
 parents=torch.as_tensor(ids.astype(np.int64),device='cuda');cond={'observed_child':torch.as_tensor(obs,device='cuda'),'valid_child_mask':torch.as_tensor(valid,device='cuda')}
 return parents,cond,torch.as_tensor(target,device='cuda'),{'mode':'gt_only','encoder':False,'score':False}

def make_model(c,g,arm):
 seed_all(c['training']['seed']);opts=dict(c['model']);opts['condition_mode']='pyramid_observed'
 model=Stage2OccupancyModel(g,**opts).cuda().train()
 # The matched starting template has a zero score contribution before removing it.
 with torch.no_grad():model.score[2].weight.zero_();model.score[2].bias.zero_()
 dummy=torch.optim.AdamW(model.parameters(),lr=c['training']['lr'],weight_decay=c['training']['weight_decay'])
 dummy,_=remove_score(model,dummy,model.state_dict())
 if hasattr(model,'score') or not hasattr(model,'lidar'):raise ValueError('Condition branch mismatch')
 masters,opt,ema,precision=migrate_fp32_runtime(model,dummy,copy.deepcopy(model.state_dict()))
 # Batch4 means the time encoder sees a stable [4,32] tensor. PTv3 is not compiled.
 model.time.forward=torch.compile(model.time.forward,backend='inductor',dynamic=False)
 return model,masters,opt,ema,precision

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['position','attention'],required=True);parser.add_argument('--mode',choices=['gate','smoke','formal'],required=True);parser.add_argument('--resume',action='store_true');a=parser.parse_args()
 gpu='2' if a.arm=='position' else '3'
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Wrong physical GPU')
 cfgpath=ROOT/f'configs/research_v2/hier_occ_flow/stage2_child_b4_{a.arm}.yaml';cfg=yaml.safe_load(cfgpath.read_text());g=SparseHierarchyGeometry(**cfg['geometry']);t=cfg['training'];out=BASE/a.arm/a.mode;out.mkdir(parents=True,exist_ok=True);status=out/'status.json';resume=out/'checkpoints/resume_latest.pt';stop=[False]
 def request_stop(*_):stop[0]=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 try:
  if t['batch_per_gpu']!=4 or t['successful_updates']!=25000 or t['successful_frame_exposures']!=100000 or t['seed']!=20261032 or cfg['candidates']['negative_fraction']!=0:raise ValueError('B4 full training protocol mismatch')
  torch.set_num_threads(4);torch.cuda.set_device(0);ds,manifest=data(cfg,g);confsha=file_sha(cfgpath);codesha=code_sha()
  if a.mode=='formal':
   for name in ('gate','smoke'):
    proof=json.loads((BASE/a.arm/name/'status.json').read_text())
    if proof.get('status')!='PASS' or proof.get('config_sha256')!=confsha or proof.get('source_sha256')!=codesha:raise ValueError(name+' must pass for this config')
  if resume.exists() and not a.resume:raise FileExistsError('Existing checkpoint; explicitly resume')
  adapter=None;static=None
  if a.arm in ('position','attention'):
   adapter=FrozenT1Stage1Adapter(**cfg['stage1']['adapter']).cuda().eval();apply_explicit_policy(adapter.model)
   adapter.model.encoder.down_blocks[0].forward=torch.compile(adapter.model.encoder.down_blocks[0].forward,backend='inductor',dynamic=False)
   static=StaticMaskCache(CACHE,cfg['train_dataset'],g)
  model,masters,optimizer,ema,precision=make_model(cfg,g,a.arm)
  def multiplier(i):return (i+1)/1000 if i<1000 else .5*(1+math.cos(math.pi*min(1.,(i-1000)/24000)))
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier);scaler=torch.cuda.amp.GradScaler(init_scale=16384.);gen=torch.Generator(device='cuda').manual_seed(t['seed']+1000);step=frames=attempts=cursor=skips=0
  if a.resume:
   ck=torch.load(resume,map_location='cpu')
   if ck['arm']!=a.arm or ck['mode']!=a.mode or ck['config_sha256']!=confsha or ck['manifest_sha256']!=manifest or ck['source_sha256']!=codesha:raise ValueError('Resume identity mismatch')
   masters.load_state_dict(ck['masters']);masters.copy_to_network();optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);ema=ck['ema'];restore_rng(ck['rng'],gen,torch.device('cuda'));step=ck['step'];frames=ck['frames'];attempts=ck['attempts'];cursor=ck['cursor'];skips=ck['skips']
  ema={n:v.cuda() for n,v in ema.items()};ema_model=copy.deepcopy(model).eval().requires_grad_(False);ema_model.load_state_dict(ema,strict=True)
  def save():
   atomic_checkpoint(resume,dict(arm=a.arm,mode=a.mode,initialization='scratch',batch_per_gpu=4,config_sha256=confsha,manifest_sha256=manifest,source_sha256=codesha,masters=masters.state_dict(),ema=ema,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(gen,torch.device('cuda')),step=step,frames=frames,attempts=attempts,cursor=cursor,skips=skips,precision=precision,ptv3_compile=False))
  atomic_json(status,dict(status='RUNNING',arm=a.arm,mode=a.mode,step=step,frames=frames,config_sha256=confsha,manifest_sha256=manifest,source_sha256=codesha,batch=4))
  if a.mode=='gate':indices=np.argsort(ds.index['length'])[-4:][::-1].tolist();limit=1
  else:indices=None;limit=32 if a.mode=='smoke' else 25000
  torch.cuda.reset_peak_memory_stats();log=(out/'train.jsonl').open('a',buffering=1);failures=0
  while step<limit:
   begin=time.perf_counter();model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
   picked=indices if indices is not None else [sampler_index(cursor+i,0,1,len(ds),t['seed']) for i in range(4)];records=[ds[int(i)] for i in picked]
   scenes=[static.assemble(r,cfg,torch.device('cuda'),adapter) if static is not None else no_encoder_scene(r,g,manifest) for r in records]
   ids,cond,target,lengths=pack_scenes(scenes)
   cond.pop('parent_score',None)
   if len(lengths)!=4 or len(ids)>t['max_batch_parent_tokens']:raise RuntimeError('Full Batch4 required; no parent truncation')
   pair=packed_training_pair(target,cond['valid_child_mask'],cond['batch'],gen)
   with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):pred=model(ids,state=pair['state'],time=pair['time'],**cond);loss=scene_mean_mse(pred,pair['velocity'],cond['valid_child_mask'],cond['batch'])
   scaler.scale(loss).backward();masters.copy_gradients();scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(masters.params.values(),t['grad_clip']);finite=bool(torch.isfinite(loss) and torch.isfinite(grad) and float(grad)>0);attempts+=1;cursor+=4
   if finite:
    scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    if scaler.get_scale()<scale:raise RuntimeError('Hidden AMP skip')
    masters.copy_to_network();scheduler.step();update_fp32_ema(ema,ema_model,model,masters,t['ema_decay']);step+=1;frames+=4;failures=0
   else:
    skips+=1;failures+=1;scaler.update(new_scale=scaler.get_scale()/2)
    if failures>=3:raise RuntimeError('Three consecutive nonfinite updates')
   torch.cuda.synchronize();row=dict(step=step,frames=frames,attempts=attempts,finite=finite,skips=skips,loss=float(loss) if torch.isfinite(loss) else None,grad_norm=float(grad) if torch.isfinite(grad) else None,seconds=time.perf_counter()-begin,parents=lengths,records=[{'sequence':r['sequence'],'frame':r['frame']} for r in records],lr=optimizer.param_groups[0]['lr'],peak_allocated=torch.cuda.max_memory_allocated(),batch=4,encoder=True,score=False);log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
   if a.mode=='formal' and (step%500==0 or stop[0]):save();atomic_json(status,dict(status='PAUSED' if stop[0] else 'RUNNING',arm=a.arm,mode=a.mode,config_sha256=confsha,manifest_sha256=manifest,source_sha256=codesha,**row))
   if stop[0]:break
  if a.mode=='formal':save()
  atomic_json(status,dict(status='PAUSED' if stop[0] else 'PASS' if a.mode!='formal' else 'COMPLETE',arm=a.arm,mode=a.mode,config_sha256=confsha,manifest_sha256=manifest,source_sha256=codesha,step=step,frames=frames,skips=skips,peak_allocated_bytes=torch.cuda.max_memory_allocated(),precision=precision,batch=4,initialization='scratch',score=False,encoder=True,compile='time_mlp_inductor',ptv3_compile=False))
 except Exception as e:atomic_json(status,dict(status='BLOCKED',arm=a.arm,mode=a.mode,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
