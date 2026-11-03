"""Continue T1 Stage2 from registered FP32 checkpoint using explicit FP16 masters."""
import copy,json,math,os,signal,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,sampler_index
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
from fpsgen.hier_occ_flow.stage2_fp16_migration import migrate_fp32_runtime
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy,update_fp32_ema
from fpsgen.hier_occ_flow.trainer import restore_rng,rng_state,atomic_checkpoint
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-t1-stage2-fulltrain-v1');OLD=BASE/'experiments/desescene_t1_stage2_fulltrain_v1/stage2/formal/checkpoints/resume_latest.pt';CFG=BASE/'configs/research_v2/hier_occ_flow/stage2_t1_fulltrain_morph.yaml';OUT=ROOT/'experiments/stage2_explicit_inductor_v1';P=OUT/'checkpoints';STATUS=OUT/'status.json'

def main():
 OUT.mkdir(parents=True,exist_ok=True);P.mkdir(exist_ok=True);resume=P/'resume_latest.pt';stopped=[False]
 def stop(sig,frame):stopped[0]=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('GPU2 only')
  torch.set_num_threads(4);torch.cuda.set_device(0);cfg,g=load_stage2_config(CFG);train=cfg['training'];ds,_,_=dataset_identity(cfg,g,verify=False)
  if resume.exists():
   ck=torch.load(resume,map_location='cpu');source=ck['source_checkpoint_sha256'];is_resume=True
   if ck.get('experiment')!='t1_explicit_inductor_v1' or ck['config_sha256']!=file_sha(CFG):raise ValueError('Resume identity mismatch')
  else:source=file_sha(OLD);ck=torch.load(OLD,map_location='cpu');is_resume=False
  adapter=FrozenT1Stage1Adapter(**cfg['stage1']['adapter']).cuda().eval();apply_explicit_policy(adapter.model)
  # Explicit Inductor of Stage1 flow. PTv3 remains eager with native CUDA ops.
  adapter.model.flow.forward=torch.compile(adapter.model.flow.forward,backend='inductor',dynamic=False)
  model=Stage2OccupancyModel(g,**cfg['model']).cuda().train()
  if is_resume:
   model.load_state_dict(ck['model_fp32'],strict=True)
  else:model.load_state_dict(ck['model'],strict=True)
  optimizer=torch.optim.AdamW(model.parameters(),lr=train['lr'],weight_decay=train['weight_decay']);optimizer.load_state_dict(ck['optimizer'])
  masters,optimizer,ema_state,precision=migrate_fp32_runtime(model,optimizer,ck['ema'])
  def multiplier(update):
   if update<train['warmup_updates']:return (update+1)/max(train['warmup_updates'],1)
   p=(update-train['warmup_updates'])/max(train['successful_updates']-train['warmup_updates'],1)
   return .5*(1+math.cos(math.pi*min(max(p,0),1)))
  preserved_lrs=[group['lr'] for group in optimizer.param_groups]
  scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier);scheduler.load_state_dict(ck['scheduler'])
  for group,lr in zip(optimizer.param_groups,preserved_lrs):group['lr']=lr
  scaler=torch.cuda.amp.GradScaler(enabled=True,init_scale=16384.);scaler.load_state_dict(ck['scaler'])
  generator=torch.Generator(device='cuda').manual_seed(train['seed']+1000)
  restore_rng(ck['rank_rng'][0],generator,torch.device('cuda'))
  successful=int(ck['successful_updates']);sampler=ck['sampler'];frames=int(sampler['successful_frame_exposures']);cursor=int(sampler['sample_cursor']);attempts=int(ck['attempts']);skips=int(ck.get('amp_skips',0));failures=0
  ema_state={n:v.cuda() for n,v in ema_state.items()};ema_model=copy.deepcopy(model).eval().requires_grad_(False);ema_model.load_state_dict(ema_state,strict=True)
  def save():
   state={n:(masters.params[n].detach().clone() if n in masters.params else v.detach().clone()) for n,v in model.state_dict().items()}
   atomic_checkpoint(resume,{'experiment':'t1_explicit_inductor_v1','config_sha256':file_sha(CFG),'source_checkpoint_sha256':source,'model_fp32':state,'ema':ema_state,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'successful_updates':successful,'attempts':attempts,'amp_skips':skips,'rank_rng':[rng_state(generator,torch.device('cuda'))],'sampler':{'successful_frame_exposures':frames,'sample_cursor':cursor},'precision_policy':precision,'compile':'stage1_flow_inductor','ptv3_compile':'not_compiled'})
  atomic_json(STATUS,{'status':'RUNNING','successful_updates':successful,'successful_frame_exposures':frames,'starting_checkpoint_sha256':source,'compile':'stage1_flow_inductor','ptv3_compile':'not_compiled','precision':precision})
  log=(OUT/'train.jsonl').open('a',buffering=1)
  while successful<50000 and frames<100000:
   start=time.perf_counter();model.zero_grad(set_to_none=True);optimizer.zero_grad(set_to_none=True)
   records=[ds[sampler_index(cursor+i,0,1,len(ds),train['seed'])] for i in range(2)]
   scenes=[assemble_stage2_scene(r,cfg,g,torch.device('cuda'),adapter) for r in records]
   ids,c,target,lengths=pack_scenes(scenes);pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],generator)
   with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):pred=model(ids,state=pair['state'],time=pair['time'],**c);loss=scene_mean_mse(pred,pair['velocity'],c['valid_child_mask'],c['batch'])
   scaler.scale(loss).backward();masters.copy_gradients();scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_(masters.params.values(),train['grad_clip'])
   finite=bool(torch.isfinite(loss) and torch.isfinite(grad) and float(grad)>0);attempts+=1;cursor+=2
   if finite:
    old=scaler.get_scale();scaler.step(optimizer);scaler.update()
    if scaler.get_scale()<old:raise RuntimeError('Unexpected hidden AMP skip')
    masters.copy_to_network();scheduler.step();update_fp32_ema(ema_state,ema_model,model,masters,train['ema_decay']);successful+=1;frames+=2;failures=0
   else:
    scaler.update(new_scale=scaler.get_scale()/2);skips+=1;failures+=1
    if failures>=3:raise RuntimeError('Three consecutive nonfinite updates')
   torch.cuda.synchronize();row=dict(successful_updates=successful,successful_frame_exposures=frames,attempts=attempts,amp_skips=skips,loss=float(loss) if torch.isfinite(loss) else None,gradient_norm=float(grad) if torch.isfinite(grad) else None,finite=finite,tokens_per_scene=lengths,seconds=time.perf_counter()-start,lr=optimizer.param_groups[0]['lr']);log.write(json.dumps(row,allow_nan=False)+'\n');print(json.dumps(row),flush=True)
   if successful%500==0 or stopped[0]:save();atomic_json(STATUS,dict(status='PAUSED' if stopped[0] else 'RUNNING',**row))
   if stopped[0]:break
  save();atomic_json(STATUS,dict(status='PAUSED' if stopped[0] else 'TRAINING_COMPLETE',successful_updates=successful,successful_frame_exposures=frames,amp_skips=skips))
 except Exception as e:atomic_json(STATUS,{'status':'BLOCKED','reason':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
