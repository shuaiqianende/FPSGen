"""Real-scene FP16 Stage2 training probe; do not alter source checkpoint."""
import sys,json,os,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.stage2_batching import pack_scenes,packed_training_pair,scene_mean_mse
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
from fpsgen.hier_occ_flow.stage2_fp16_migration import migrate_fp32_runtime
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.storage import atomic_json
BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-t1-stage2-fulltrain-v1');CFG=BASE/'configs/research_v2/hier_occ_flow/stage2_t1_fulltrain_morph.yaml';CK=BASE/'experiments/desescene_t1_stage2_fulltrain_v1/stage2/formal/checkpoints/resume_latest.pt';OUT=ROOT/'experiments/stage2_fp16_inductor_probe_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=False);report={'status':'RUNNING','checkpoint':str(CK),'precision':'fp16_model_fp32_masters','compile_target':'stage1_flow.forward','actual_training':False};atomic_json(OUT/'status.json',report)
 try:
  torch.set_num_threads(4);torch.cuda.set_device(0);cfg,g=load_stage2_config(CFG);ds,_,_=dataset_identity(cfg,g,verify=False)
  ck=torch.load(CK,map_location='cpu')
  # Preserve time embedding frequencies and other FP32 buffers from original checkpoint.
  adapter=FrozenT1Stage1Adapter(**cfg['stage1']['adapter']).cuda().eval();apply_explicit_policy(adapter.model)
  adapter.model.flow.forward=torch.compile(adapter.model.flow.forward,backend='inductor',dynamic=False)
  model=Stage2OccupancyModel(g,**cfg['model']).cuda().train();model.load_state_dict(ck['model'],strict=True)
  optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay']);optimizer.load_state_dict(ck['optimizer'])
  masters,optimizer,ema,precision=migrate_fp32_runtime(model,optimizer,ck['ema'])
  scenes=[assemble_stage2_scene(ds[i],cfg,g,torch.device('cuda'),adapter) for i in (0,1)]
  ids,c,target,lengths=pack_scenes(scenes);gen=torch.Generator(device='cuda').manual_seed(20261032);pair=packed_training_pair(target,c['valid_child_mask'],c['batch'],gen)
  torch.cuda.reset_peak_memory_stats();scaler=torch.cuda.amp.GradScaler(init_scale=1024.);t=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):pred=model(ids,state=pair['state'],time=pair['time'],**c);loss=scene_mean_mse(pred,pair['velocity'],c['valid_child_mask'],c['batch'])
  scaler.scale(loss).backward();masters.copy_gradients();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(masters.params.values(),1.)
  if not bool(torch.isfinite(loss) and torch.isfinite(norm)) or float(norm)<=0:raise RuntimeError('invalid gradient')
  scaler.step(optimizer);scaler.update();masters.copy_to_network();torch.cuda.synchronize()
  report.update(status='PROBE_PASS',loss=float(loss),gradient_norm=float(norm),tokens_per_scene=lengths,seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),precision_policy=precision,warning='Single batch; no exact resume, no trained checkpoint')
 except Exception as exc:report.update(status='BLOCKED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
 atomic_json(OUT/'status.json',report);print(json.dumps(report,indent=2),flush=True)
 if report['status']!='PROBE_PASS':raise SystemExit(1)
if __name__=='__main__':main()
