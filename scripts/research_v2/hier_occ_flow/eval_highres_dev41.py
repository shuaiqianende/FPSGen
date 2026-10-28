"""Fixed non-seq08 41-frame development screen; never runs the 900-frame set."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,os,time
import numpy as np,torch,yaml
from fpsgen.hier_occ_flow.stage1_training import validate_config
from fpsgen.hier_occ_flow.stage1_occupancy import Stage1DirectOccupancy
from fpsgen.hier_occ_flow.stage1_evaluation import evaluate_stage1
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,choices=(2,3));a=p.parse_args()
    c=yaml.safe_load(a.config.read_text());c['geometry']=yaml.safe_load((a.config.parent/c['geometry']).read_text());g=validate_config(c);gpu=a.gpu if a.gpu is not None else c['training']['physical_gpus'][0]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Device mismatch')
    admission=gpu_preflight((gpu,),minimum_free_gib=16.,allow_shared_compute=False)
    if admission['status']!='AVAILABLE':raise RuntimeError(json.dumps(admission))
    if a.output.exists():raise FileExistsError('Preserve screen result')
    a.output.mkdir(parents=True);torch.set_num_threads(4);v=torch.load(a.checkpoint,map_location='cpu')
    if v['initialization_mode']!='scratch' or v['successful_updates'] not in (1000,2500) or v['sampler']['successful_frame_exposures']!=2*v['successful_updates'] or v['identity']['config_sha256']!=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest():raise ValueError('Own2K/5K scratch checkpoint and config required')
    m=Stage1DirectOccupancy(g,objective='fm',widths=c['model']['widths'],activation_checkpointing=False,encoder_variant=c['model']['encoder_variant']).cuda().eval()
    m.load_state_dict(v['ema'],strict=True);m.requires_grad_(False);torch.cuda.reset_peak_memory_stats();start=time.time()
    d=evaluate_stage1(m,c,g,'cuda',max_frames=41);rows=d['nfe_results']['2']['rows']
    if len(rows)!=41 or any(r['sequence']=='08' for r in rows):raise ValueError('Only fixed41 non-seq08 development frames allowed')
    atomic_json(a.output/'results.json',dict(status='COMPLETE_DEV41',checkpoint_sha256=file_sha(a.checkpoint),checkpoint_identity=v['identity'],successful_updates=v['successful_updates'],successful_exposures=v['sampler']['successful_frame_exposures'],gpu=admission,scope='fixed41 independent development; coarse screening only',elapsed_seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=sum(p.numel() for p in m.parameters()),result=d))
    print(json.dumps(d['nfe_results']['2']['summary']['macro']),flush=True)

if __name__=='__main__':main()
