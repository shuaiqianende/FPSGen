"""Wait for one completed B4 training, then smoke and evaluate 41 frames."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/stage2_b4_paired_v1'
PY='/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python'
EVAL=ROOT/'scripts/research_v2/hier_occ_inference/eval_stage2_b4_pair.py'

def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['with_encoder','no_encoder'],required=True);p.add_argument('--train-pid',type=int,required=True);a=p.parse_args()
    gpu={'with_encoder':2,'no_encoder':3}[a.arm];root=BASE/a.arm;root.mkdir(parents=True,exist_ok=True);status=root/'evaluation_queue.json'
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong GPU')
    try:
        while True:
            train=json.loads((root/'formal/status.json').read_text())
            if train['status']=='COMPLETE':
                if train.get('step')!=25000 or train.get('frames')!=100000:raise ValueError('Training completion count mismatch')
                break
            if train['status'] in ('BLOCKED','PAUSED','FAILED'):raise RuntimeError('Training not completed: '+train['status'])
            try:os.kill(a.train_pid,0)
            except ProcessLookupError:raise RuntimeError('Training process exited before completion')
            atomic(status,{'status':'WAITING_FOR_TRAINING','train_pid':a.train_pid,'gpu':gpu,'last_reported_step':train.get('step'),'last_reported_frames':train.get('frames')});time.sleep(30)
        while True:
            free,util=map(float,subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            procs=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if free>=16000 and util<=10 and not procs:break
            atomic(status,{'status':'WAITING_FOR_GPU','gpu':gpu,'free_mib':free,'utilization':util,'compute_pids':procs});time.sleep(30)
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        for phase,extra in [('SMOKE',['--smoke']),('SEQ08_41',[])]:
            atomic(status,{'status':'RUNNING_'+phase,'gpu':gpu,'training_status':'COMPLETE','script':str(EVAL)})
            with (root/('eval_smoke.log' if extra else 'eval_seq08_41.log')).open('a') as f:
                subprocess.run([PY,'-u',str(EVAL),'--arm',a.arm,*extra],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        atomic(status,{'status':'COMPLETE','gpu':gpu,'training_status':'COMPLETE','result':str(root/'eval_seq08_41/results.json')})
    except Exception as e:
        atomic(status,{'status':'FAILED','gpu':gpu,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
