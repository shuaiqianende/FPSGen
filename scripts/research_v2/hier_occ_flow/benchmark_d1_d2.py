"""Separate 41-frame GPU timing without quality/profiler hooks."""
from pathlib import Path
import argparse,json,os,statistics,sys,time
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import torch,yaml
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Prior timing retained')
 c=yaml.safe_load(a.config.read_text());c['geometry']=yaml.safe_load((a.config.parent/c['geometry']).read_text());g=validate_config(c)
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(c['training']['physical_gpus'][0]):raise ValueError('wrong GPU')
 ck=torch.load(a.checkpoint,map_location='cpu')
 if ck['successful_updates']!=50000 or ck['sampler']['successful_frame_exposures']!=100000:raise ValueError('100K checkpoint required')
 m=Stage1DirectOccupancy(g,objective='fm',widths=c['model']['widths'],activation_checkpointing=False,encoder_variant=c['model']['encoder_variant']).cuda().eval().requires_grad_(False);m.load_state_dict(ck['ema'],strict=True);torch.set_num_threads(4)
 def one(frame):
  x=torch.from_numpy(input_xyz(Path(c['data_root'])/'08/input_'/f'{frame:06d}.npy')).cuda();torch.cuda.synchronize();start=time.perf_counter();e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);e2=torch.cuda.Event(enable_timing=True)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
   e0.record();cond=m.encoder(x);e1.record();m.sample(x,generator=torch.Generator(device='cuda').manual_seed(scene_seed('08',frame,20261031)),nfe=2,condition=cond);e2.record()
  e2.synchronize();return dict(frame=frame,encoder_ms=e0.elapsed_time(e1),flow_and_sampling_ms=e1.elapsed_time(e2),stage1_wall_ms=1000*(time.perf_counter()-start))
 for f in (0,800,1600,2400,3200):one(f)
 torch.cuda.reset_peak_memory_stats();rows=[one(f) for f in range(0,4001,100)]
 atomic_json(a.output,dict(status='COMPLETE',protocol='batch1, FP32 weights, FP16 autocast, NFE2; 5 warmup + seq08/skip100 41; shared GPU; CPU input IO excluded; no quality computation',checkpoint_sha256=file_sha(a.checkpoint),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),rows=rows,mean={k:statistics.mean(r[k] for r in rows) for k in ('encoder_ms','flow_and_sampling_ms','stage1_wall_ms')}))
if __name__=='__main__':main()
