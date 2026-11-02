"""Real B2 preflight then fresh, resumable Stage2 FM on one authorized GPU."""
import os,sys,time,traceback,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,config_sha,source_sha
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
from fpsgen.hier_occ_flow.storage import atomic_json

P=ROOT/'experiments/desescene_t1_stage2_fulltrain_v1/stage2';CFG=ROOT/'configs/research_v2/hier_occ_flow/stage2_t1_fulltrain_morph.yaml'
def main():
 P.mkdir(parents=True,exist_ok=True);status=P/'launch_status.json'
 try:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('Physical GPU2 required')
  if (P/'formal/checkpoints/step000000.pt').exists():raise ValueError('Existing formal run; resume only with own identity')
  torch.set_num_threads(4);torch.cuda.set_device(0);cfg,g=load_stage2_config(CFG);ds,_,_=dataset_identity(cfg,g,verify=False)
  atomic_json(status,{'status':'RUNNING_REAL_BATCH2_PREFLIGHT'})
  adapter=FrozenT1Stage1Adapter(**cfg['stage1']['adapter']).cuda().eval();model=Stage2OccupancyModel(g,**cfg['model']).cuda().train()
  # Two genuine scenes; no clipping or surrogate data. Formal training always uses full scenes.
  records=[ds[i] for i in (0,1)];scenes=[assemble_stage2_scene(r,cfg,g,torch.device('cuda'),adapter) for r in records]
  ids,c,target,lengths=pack_scenes(scenes);pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],torch.Generator(device='cuda').manual_seed(20261032))
  torch.cuda.reset_peak_memory_stats();opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr']);opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.float16):
   pred=model(ids,state=pair['state'],time=pair['time'],**c);loss=scene_mean_mse(pred,pair['velocity'],c['valid_child_mask'],c['batch'])
  scaler=torch.cuda.amp.GradScaler(init_scale=16384.);scaler.scale(loss).backward();scaler.unscale_(opt)
  grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
  if not bool(torch.isfinite(loss) and torch.isfinite(grad)) or float(grad)<=0:raise ValueError('Nonfinite or zero Stage2 gradient')
  if any(x.requires_grad or x.grad is not None for x in adapter.parameters()):raise ValueError('T1 must remain frozen')
  torch.cuda.synchronize();report={'status':'PASS_REAL_BATCH2','config_sha256':config_sha(cfg),'source_sha256':source_sha(),'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],'records':[{'sequence':r['sequence'],'frame':r['frame']} for r in records],'tokens_per_scene':lengths,'loss':float(loss),'gradient_norm':float(grad),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'no_token_truncation':True}
  atomic_json(P/'preflight.json',report);del adapter,model,scenes,ids,c,target,pair,pred,loss,opt;torch.cuda.empty_cache()
  admission=P/'admission.json';atomic_json(admission,{'status':'AVAILABLE','physical_ids':[2],'checked_at_epoch':time.time(),'reason':'user explicitly authorized GPU2 shared use; real B2 preflight passed'})
  env=os.environ.copy();env['HS_STAGE2_ADMISSION']=str(admission)
  with (P/'formal.log').open('x',buffering=1) as log:
   child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/train_stage2_384.py'),'--config',str(CFG),'--output',str(P/'formal')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   atomic_json(status,{'status':'RUNNING_FORMAL','child_pid':child.pid,'preflight':str(P/'preflight.json')})
   code=child.wait()
  if code:raise RuntimeError('Formal training failed; inspect formal.log')
  atomic_json(status,{'status':'FORMAL_PROCESS_EXITED','code':code})
 except Exception as e:
  atomic_json(status,{'status':'BLOCKED','reason':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
