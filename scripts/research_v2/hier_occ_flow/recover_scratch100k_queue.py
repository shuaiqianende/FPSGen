"""Recover interrupted 100K arms and preserve the GPU2 U1+A successor."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);a=p.parse_args();old=ROOT/f'experiments/desescene_u1_scratch100k_v1/gpu{a.gpu}_queue';out=ROOT/f'experiments/desescene_u1_scratch100k_v1/gpu{a.gpu}_recovery';oldstatus=json.loads((old/'status.json').read_text());arm='equal' if a.gpu==2 else 'patch'
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or out.exists() or oldstatus['status']!=f'RUNNING_{arm}_train100k':raise ValueError('Unexpected recovery identity')
 if subprocess.run(['ps','-p',str(oldstatus['child_pid'])],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Existing child still running')
 pins=oldstatus['source_sha256']
 if any(file_sha(Path(k))!=v for k,v in pins.items()):raise ValueError('Original source changed')
 ck=old/arm/'formal/checkpoints/resume_latest.pt'
 if not ck.exists():raise FileNotFoundError('Checkpoint missing')
 out.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json';gate=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_u1_100k_seq08.py';results={}
 def status(s,**kw):atomic_json(out/'status.json',dict(status=s,gpu=a.gpu,pid=os.getpid(),results=results,**kw))
 def run(label,args):
  if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned source changed')
  with (out/(label+'.log')).open('x') as f:
   child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+label,child_pid=child.pid)
   if child.wait():raise RuntimeError(label+' failed; inspect log')
 def admit(label):
  status('WAITING_GPU',arm=label)
  while gpu_preflight((a.gpu,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
 try:
  admit(arm);cfg=ROOT/f'configs/research_v2/hier_occ_flow/scratch100k_{arm}.yaml';formal=old/arm/'formal'
  run(arm+'_resume100k',[train,'--config',cfg,'--output',formal,'--resume',ck]);v=json.loads((formal/'summary.json').read_text())
  if v['successful_updates']!=50000 or v['successful_frame_exposures']!=100000:raise ValueError('100K incomplete')
  run(arm+'_seq08_41',[ev,'--config',cfg,'--checkpoint',ck,'--output',out/(arm+'_seq08_41')]);results[arm]='COMPLETE_100K_AND_41'
  if a.gpu==2:
   arm='u1a';admit(arm);root=out/arm;root.mkdir();cfg=ROOT/'configs/research_v2/hier_occ_flow/scratch100k_u1a.yaml'
   run('u1a_gate',[gate,'--config',cfg,'--output',root/'gate']);smoke=root/'smoke';base=[train,'--config',cfg,'--output',smoke,'--smoke'];run('u1a_smoke2',base+['--max-updates','2']);run('u1a_smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
   v=json.loads((smoke/'summary.json').read_text());i=json.loads((smoke/'initialization_audit.json').read_text())
   if v['successful_updates']!=32 or v['attempts']>34 or i['pretrained_loaded'] or i['shared_flow_initialization_sha256']!='6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd':raise ValueError('U1A smoke failed')
   formal=root/'formal';run('u1a_train100k',[train,'--config',cfg,'--output',formal]);v=json.loads((formal/'summary.json').read_text())
   if v['successful_updates']!=50000 or v['successful_frame_exposures']!=100000:raise ValueError('U1A 100K incomplete')
   run('u1a_seq08_41',[ev,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',out/'u1a_seq08_41']);results[arm]='COMPLETE_100K_AND_41'
  status('COMPLETE')
 except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
