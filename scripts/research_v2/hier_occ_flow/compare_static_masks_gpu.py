"""Compare real Stage2 scene assembly with frozen T1, with and without cache."""
import sys,time,json,statistics,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity
from fpsgen.hier_occ_flow.t1_stage1_adapter import FrozenT1Stage1Adapter
from fpsgen.hier_occ_flow.explicit_fp16_training_policy import apply_explicit_policy
from fpsgen.hier_occ_flow.stage2_dataset import assemble_stage2_scene
from fpsgen.hier_occ_flow.static_mask_cache import StaticMaskCache
from fpsgen.hier_occ_flow.storage import atomic_json
CFG=ROOT/'configs/research_v2/hier_occ_flow/stage2_t1_no_score_scratch.yaml';OUT=ROOT/'experiments/stage2_static_masks_gpu_comparison_v1';CACHE=ROOT/'experiments/stage2_static_masks_full_v1'
def main():
 OUT.mkdir(parents=True,exist_ok=False);atomic_json(OUT/'status.json',{'status':'RUNNING'});rows=[]
 try:
  torch.set_num_threads(4);torch.cuda.set_device(0);cfg,g=load_stage2_config(CFG);ds,_,_=dataset_identity(cfg,g,verify=False);cache=StaticMaskCache(CACHE,cfg['train_dataset'],g)
  adapter=FrozenT1Stage1Adapter(**cfg['stage1']['adapter']).cuda().eval();apply_explicit_policy(adapter.model)
  adapter.model.encoder.down_blocks[0].forward=torch.compile(adapter.model.encoder.down_blocks[0].forward,backend='inductor',dynamic=False)
  indices=[int(x) for x in __import__('numpy').linspace(0,len(ds)-1,12,dtype=int)]
  def measure(fn):
   torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();return out,1000*(time.perf_counter()-t)
  with torch.no_grad():
   for i in indices:
    r=ds[i];a,ta=measure(lambda:assemble_stage2_scene(r,cfg,g,torch.device('cuda'),adapter));b,tb=measure(lambda:cache.assemble(r,cfg,torch.device('cuda'),adapter))
    equal=torch.equal(a[0],b[0]) and torch.equal(a[2],b[2]) and all(torch.equal(a[1][k],b[1][k]) for k in ('observed_child','valid_child_mask','lidar_features','parent_score'))
    if not equal:raise ValueError(f'Parity failed {r["sequence"]}/{r["frame"]}')
    row=dict(sequence=r['sequence'],frame=r['frame'],parents=len(a[0]),original_ms=ta,cache_ms=tb,parity=equal);rows.append(row);print(json.dumps(row),flush=True)
  # first two frames exclude lazy GPU/compile initialization; ordered, not a three-round benchmark
  timed=rows[2:];result=dict(status='PASS',scope='Scene condition assembly only, GPU3, batch1 x2 equivalent, no PTv3 backward/optimizer; first two warmup; fixed input and GT support',rows=rows,original_mean_ms=statistics.mean(r['original_ms'] for r in timed),cache_mean_ms=statistics.mean(r['cache_ms'] for r in timed));atomic_json(OUT/'result.json',result);atomic_json(OUT/'status.json',{'status':'COMPLETE'})
 except Exception as e:atomic_json(OUT/'status.json',{'status':'BLOCKED','reason':str(e),'traceback':traceback.format_exc(),'completed':len(rows)});raise
if __name__=='__main__':main()
