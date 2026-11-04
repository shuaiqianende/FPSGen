"""Wait for existing GPU2 training/evaluation, then run the registered 408-frame Stage1 set."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/desescene_eval408_nextstage2_v1'
PREV=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-stage2-child-b4-followup-v1/experiments/stage2_child_b4_followup_v1/position/formal/status.json')
W=Path('/data-12/M2024-HWZ/FPSGen/worktrees')
ITEMS=(
 ('u1_adapted_20k',W/'desescene-u1-triplane-v1/experiments/desescene_u1_triplane_v1/pilot_u1/formal'),
 ('u1a_100k',W/'desescene-u1-scratch100k-v1/experiments/desescene_u1_scratch100k_v1/gpu2_recovery/u1a/formal'),
 ('depth_moment_100k',W/'desescene-depth-moment-v1/experiments/desescene_depth_moment_v1/gpu2_t1/formal'))
def status(v):
 BASE.mkdir(parents=True,exist_ok=True);tmp=BASE/'stage1_queue.tmp';tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,BASE/'stage1_queue.json')
def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='2':raise ValueError('GPU2 only')
 while True:
  prev=json.loads(PREV.read_text()) if PREV.exists() else {}
  if prev.get('status')=='COMPLETE' and prev.get('step')==25000 and prev.get('frames')==100000:break
  if prev.get('status') in ('FAILED','BLOCKED','PAUSED'):raise RuntimeError('Existing GPU2 training failed or paused; inspect before evaluation')
  status({'status':'WAITING_PREVIOUS_GPU2_TRAINING','previous':prev.get('status')});time.sleep(30)
 while True:
  free,util=map(float,subprocess.check_output(['nvidia-smi','--id=2','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
  active=subprocess.check_output(['nvidia-smi','--id=2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if free>=18000 and util<=10 and not active:break
  status({'status':'WAITING_GPU2','free_mib':free,'utilization':util,'compute_pids':active});time.sleep(30)
 env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1');script=ROOT/'scripts/research_v2/hier_occ_flow/eval_stage1_seq08_408.py'
 for name,formal in ITEMS:
  ck=formal/'checkpoints'/('resume_latest.pt' if name=='u1_adapted_20k' else 'step050000.pt');cfg=formal/'config.json'
  for phase,count in (('smoke',1),('seq08_skip10_408',408)):
   out=BASE/name/phase;result=out/'results.json'
   if result.exists():
    d=json.loads(result.read_text())
    if d.get('status')=='COMPLETE' and len(d.get('rows',[]))==count:continue
    raise ValueError(f'Invalid existing result {result}')
   status({'status':'RUNNING_STAGE1','variant':name,'phase':phase,'frames':count})
   out.parent.mkdir(parents=True,exist_ok=True)
   with (out.parent/(phase+'.log')).open('a') as log:
    subprocess.run([sys.executable,'-u',str(script),'--config',str(cfg),'--checkpoint',str(ck),'--output',str(out),'--max-frames',str(count)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   d=json.loads(result.read_text())
   if d.get('status')!='COMPLETE' or len(d.get('rows',[]))!=count:raise ValueError(f'Incomplete {result}')
 status({'status':'STAGE1_COMPLETE','note':'Joint 408 evaluation and U1 100K run in separate dependent queues.'})
if __name__=='__main__':
 try:main()
 except Exception as e:status({'status':'FAILED','reason':repr(e),'traceback':traceback.format_exc()});raise
