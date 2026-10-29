"""Resume a vanished Patch FM process from its last full checkpoint."""
from pathlib import Path
import argparse,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);a=p.parse_args();old=ROOT/f'experiments/desescene_patch_flow_v1/gpu{a.gpu}_queue';out=ROOT/f'experiments/desescene_patch_flow_v1/gpu{a.gpu}_recovery';s=json.loads((old/'status.json').read_text())
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or out.exists() or s['status']!='RUNNING_train20k':raise ValueError('Not an interrupted training queue')
 if subprocess.run(['ps','-p',str(s['child_pid'])],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('Previous training process still exists')
 if any(file_sha(Path(k))!=v for k,v in s['source_sha256'].items()):raise ValueError('Pinned model source changed')
 ck=old/'formal/checkpoints/resume_latest.pt'
 if not ck.exists():raise ValueError('Checkpoint missing')
 out.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json';cfg=ROOT/f"configs/research_v2/hier_occ_flow/patch_{s['arm']}_20k.yaml";train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py'
 def status(state,**kw):atomic_json(out/'status.json',dict(status=state,gpu=a.gpu,arm=s['arm'],pid=os.getpid(),**kw))
 def run(label,args):
  with (out/(label+'.log')).open('x') as f:
   child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+label,child_pid=child.pid)
   if child.wait():raise RuntimeError(label+' failed; inspect log')
 try:
  status('WAITING_SHARED_GPU')
  while gpu_preflight((a.gpu,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
  run('train20k',[train,'--config',cfg,'--output',old/'formal','--resume',ck]);v=json.loads((old/'formal/summary.json').read_text())
  if v['successful_updates']!=10000 or v['successful_frame_exposures']!=20000:raise ValueError('20K incomplete')
  run('seq08_41',[ev,'--config',cfg,'--checkpoint',ck,'--output',out/'seq08_41']);status('COMPLETE_20K_AND_41')
 except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
