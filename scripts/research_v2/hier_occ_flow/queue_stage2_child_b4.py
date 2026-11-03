"""Serial one-GPU queue, gated on the preceding B4 41-frame evaluation."""
import argparse,json,os,subprocess,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'experiments/stage2_child_b4_followup_v1'
PREVIOUS=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-stage2-b4-paired-v1/experiments/stage2_b4_paired_v1')
PY='/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python'
TRAIN=ROOT/'scripts/research_v2/hier_occ_flow/train_stage2_child_b4.py'
EVAL=ROOT/'scripts/research_v2/hier_occ_inference/eval_stage2_child_b4.py'

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['position','attention'],required=True);a=p.parse_args()
    gpu,prior={'position':(2,'with_encoder'),'attention':(3,'no_encoder')}[a.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong physical GPU')
    out=BASE/a.arm;out.mkdir(parents=True,exist_ok=True);status=out/'queue_status.json';parent=PREVIOUS/prior/'evaluation_queue.json'
    try:
        while True:
            upstream=json.loads(parent.read_text())
            if upstream.get('status')=='COMPLETE':
                result=json.loads((PREVIOUS/prior/'eval_seq08_41/results.json').read_text())
                if result.get('status')!='COMPLETE' or len(result.get('rows',[]))!=41:raise ValueError('Prior 41-frame evaluation incomplete')
                break
            if upstream.get('status')=='FAILED':raise RuntimeError('Preceding evaluation failed; stop queue')
            write(status,{'status':'WAITING_PREVIOUS_EVALUATION','gpu':gpu,'previous_arm':prior,'previous_status':upstream.get('status')});time.sleep(30)
        while True:
            free,util=map(float,subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            active=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if free>=18000 and util<=10 and not active:break
            write(status,{'status':'WAITING_GPU','gpu':gpu,'free_mib':free,'utilization':util,'compute_pids':active});time.sleep(30)
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        for mode in ('gate','smoke','formal'):
            state=out/mode/'status.json'
            previous=json.loads(state.read_text()) if state.exists() else {}
            if mode!='formal' and previous.get('status')=='PASS':continue
            if previous.get('status') in ('BLOCKED','FAILED'):raise RuntimeError(f'{mode} previously failed; no silent retry')
            cmd=[PY,'-u',str(TRAIN),'--arm',a.arm,'--mode',mode]
            if mode=='formal' and (out/'formal/checkpoints/resume_latest.pt').exists():cmd.append('--resume')
            write(status,{'status':'RUNNING_'+mode.upper(),'gpu':gpu,'previous_evaluation':'COMPLETE'})
            with (out/(mode+'.log')).open('a') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            result=json.loads(state.read_text())
            if result.get('status')!=('COMPLETE' if mode=='formal' else 'PASS'):raise RuntimeError(f'{mode} did not complete: {result}')
        for phase,extra in [('EVAL_SMOKE',['--smoke']),('EVAL_41',[])]:
            write(status,{'status':'RUNNING_'+phase,'gpu':gpu})
            with (out/(phase.lower()+'.log')).open('a') as f:subprocess.run([PY,'-u',str(EVAL),'--arm',a.arm,*extra],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        write(status,{'status':'COMPLETE','gpu':gpu,'results':str(out/'eval_seq08_41/results.json')})
    except Exception as exc:
        write(status,{'status':'FAILED','gpu':gpu,'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
