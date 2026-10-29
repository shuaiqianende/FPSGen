"""Fresh scratch100K per-GPU sequence; preserve earlier runs and model identity."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight
PRIOR=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-patch-flow-v1/experiments/desescene_patch_flow_v1')

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or a.output.exists():raise ValueError('GPU or queue identity invalid')
 a.output.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
 arms=('equal','u1a') if a.gpu==2 else ('patch',);dep=PRIOR/f'gpu{a.gpu}_recovery/status.json'
 gate=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_u1_100k_seq08.py'
 files=[Path(__file__),gate,train,ev,*[ROOT/'fpsgen/hier_occ_flow'/x for x in ('stage1_training.py','stage1_occupancy.py','stage1_network.py','patch_fm_backbone.py','patch_unet_condition.py','gctv_encoder.py')],*[ROOT/f'configs/research_v2/hier_occ_flow/scratch100k_{x}.yaml' for x in arms]];pins={str(x):file_sha(x) for x in files};results={}
 def status(s,**kw):atomic_json(a.output/'status.json',dict(status=s,gpu=a.gpu,pid=os.getpid(),results=results,source_sha256=pins,**kw))
 def run(label,args):
  if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned source changed')
  with (a.output/(label+'.log')).open('x') as f:
   child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+label,child_pid=child.pid)
   if child.wait():raise RuntimeError(label+' failed; preserve log and stop queue')
 try:
  if a.gpu==3:
   status('WAITING_PRIOR_20K_AND_41')
   while True:
    d=json.loads(dep.read_text());s=d['status']
    if s=='COMPLETE_20K_AND_41':break
    if s=='BLOCKED':raise RuntimeError('Prior GPU3 task blocked')
    time.sleep(20)
  for arm in arms:
   status('WAITING_GPU',arm=arm)
   while gpu_preflight((a.gpu,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
   root=a.output/arm;root.mkdir();cfg=ROOT/f'configs/research_v2/hier_occ_flow/scratch100k_{arm}.yaml'
   run(arm+'_gate',[gate,'--config',cfg,'--output',root/'gate'])
   smoke=root/'smoke';base=[train,'--config',cfg,'--output',smoke,'--smoke'];run(arm+'_smoke2',base+['--max-updates','2']);run(arm+'_smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
   s=json.loads((smoke/'summary.json').read_text());i=json.loads((smoke/'initialization_audit.json').read_text())
   if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34 or i['pretrained_loaded']:raise ValueError('Scratch smoke health failed')
   if arm!='patch' and i['shared_flow_initialization_sha256']!='6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd':raise ValueError('Original FM Flow initialization differs')
   run(arm+'_train100k',[train,'--config',cfg,'--output',root/'formal']);v=json.loads((root/'formal/summary.json').read_text())
   if v['successful_updates']!=50000 or v['successful_frame_exposures']!=100000:raise ValueError('100K incomplete')
   run(arm+'_seq08_41',[ev,'--config',cfg,'--checkpoint',root/'formal/checkpoints/resume_latest.pt','--output',root/'seq08_41']);results[arm]='COMPLETE_100K_AND_41';atomic_json(a.output/'progress.json',dict(results=results))
  status('COMPLETE')
 except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
