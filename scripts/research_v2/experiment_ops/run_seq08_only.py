"""User-directed scheduling revision only: unchanged trained/gated model sources."""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--role',choices=('stage2_fm','stage1_eval'),required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage1-root',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.role=='stage2_fm':
        from fpsgen.hier_occ_flow.encoder_free_fm_training import load_fm_config
        cfg,_=load_fm_config(a.config);gpu=2;minimum=22.;status_name='supervisor.json'
        cpu=json.loads((a.output/'cpu_report.json').read_text());gate=json.loads(Path(cfg['engineering_report']).read_text())
        smoke=json.loads(Path(cfg['smoke_summary']).read_text());pin=source_sha()
        if (cpu.get('status')!='CPU_PASS' or cpu['source_sha256']!=pin
            or gate.get('status')!='GPU_PASS' or gate['source_sha256']!=pin or gate['config_sha256']!=config_sha(cfg)
            or [g['physical_gpu'] for g in gate['gpu_admission']['gpus']]!=[2] or not gate['one_forward_no_jvp']
            or smoke['status']!='SMOKE_PASS' or smoke['counts']['updates']!=32 or smoke['counts']['exposures']!=64
            or smoke['identity']['source_sha256']!=pin or smoke['identity']['config_sha256']!=config_sha(cfg)):
            raise ValueError('Own unchanged-source/config/GPU2 gate and discarded ordinary-FM Smoke required')
        prior=json.loads((a.stage1_root/'formal/summary.json').read_text())
        if prior['status']!='TRAINING_COMPLETE' or prior['counts']['updates']!=10000 or prior['counts']['exposures']!=20000:
            raise ValueError('Stage1 training must have completed; quality is no longer a blocker')
        if (a.output/'formal').exists():raise FileExistsError('No blind restart of existing formal training')
    else:
        from fpsgen.hier_occ_flow.meanflow_training import load_config
        cfg,_=load_config(a.config);gpu=3;minimum=16.;status_name='single_arm_supervisor.json';pin=source_sha()
        prior=json.loads((a.output/'formal/summary.json').read_text())
        if prior['status']!='TRAINING_COMPLETE' or prior['counts']['updates']!=10000 or prior['counts']['exposures']!=20000:
            raise ValueError('Own final20K Stage1 checkpoint required')
        if prior['identity']['source_sha256']!=pin or prior['identity']['config_sha256']!=config_sha(cfg):
            raise ValueError('Keep the trained Stage1 model/config identity unchanged')
    if a.check_only:
        print(json.dumps(dict(status='READY',role=a.role,physical_gpu=gpu,source_sha256=pin,evaluation_frames=list(range(0,4001,100)))));return
    marker=a.output/'seq08_only_launch.json'
    if marker.exists():raise FileExistsError('Already launched this policy revision; preserve evidence')
    old=json.loads((a.output/status_name).read_text())
    if old.get('status')!='SUPERSEDED_BY_USER_SEQ08_ONLY':raise ValueError('Explicit stopped-controller handoff required')
    lock_path=ROOT.parent.parent/f'experiments/desescene_meanflow_locks/meanflow_gpu{gpu}.lock'
    lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config_pin=file_sha(a.config);controller_pin=file_sha(__file__)
    def pinned():
        if source_sha()!=pin or file_sha(a.config)!=config_pin or file_sha(__file__)!=controller_pin:
            raise RuntimeError('Pinned model/config/scheduler changed')
    def mark(state,**fields):atomic_json(a.output/status_name,dict(status=state,pid=os.getpid(),epoch=time.time(),
        role=a.role,physical_gpu=gpu,source_sha256=pin,config_file_sha256=config_pin,
        evaluation_policy='seq08_skip100_41_only_user_20261026',**fields))
    atomic_json(marker,dict(pid=os.getpid(),role=a.role,physical_gpu=gpu,source_sha256=pin,controller_sha256=controller_pin,
        config_file_sha256=config_pin,epoch=time.time(),development900_skipped_by_user=True,formal_budget_changed=False))
    def run(name,script,args):
        while True:
            pinned();report=gpu_preflight((gpu,),minimum_free_gib=minimum,allow_shared_compute=False)
            if report['status']=='AVAILABLE':break
            mark('WAITING_GPU',admission=report);time.sleep(15)
        task=a.output/(name+'.task.json')
        if task.exists():raise FileExistsError('Existing task record; no automatic retry')
        command=[sys.executable,'-u',str(script),*map(str,args)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),DESESCENE_MF_SHARED='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        with (a.output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            mark('RUNNING_'+name.upper(),child_pid=child.pid,command=command)
            atomic_json(task,dict(status='RUNNING',pid=child.pid,command=command));code=child.wait()
        atomic_json(task,dict(status='COMPLETE' if code==0 else 'BLOCKED',exit_code=code,command=command))
        if code:raise RuntimeError(f'{name} exited {code}; review, no automatic retry')
    try:
        scripts=ROOT/'scripts/research_v2/hier_occ_flow'
        if a.role=='stage2_fm':
            for count in (100,500,10000):
                args=['--config',a.config,'--output',a.output/'formal','--max-updates',count]
                if count!=100:args+=['--resume',a.output/'formal/checkpoints/resume_latest.pt']
                run(f'seq08_policy_formal_{count}',scripts/'train_encoder_free_fm.py',args)
                proof=json.loads((a.output/'formal/summary.json').read_text())
                if proof['counts']['updates']!=count or proof['counts']['exposures']!=2*count:raise ValueError('Successful budget mismatch')
            run('seq08_policy_quality41',scripts/'eval_encoder_free_fm.py',['--config',a.config,'--checkpoint',a.output/'formal/checkpoints/resume_latest.pt','--output',a.output/'evaluation/seq08_41','--split','seq08_41'])
        else:
            run('seq08_policy_quality41',scripts/'eval_meanflow.py',['--config',a.config,'--checkpoint',a.output/'formal/checkpoints/resume_latest.pt','--output',a.output/'evaluation/seq08_41','--split','seq08_41'])
            benchmark=ROOT/'scripts/research_v2/experiment_ops/benchmark_meanflow_seq08_41.py'
            run('seq08_policy_paired_benchmark41',benchmark,['--config',a.config,'--checkpoint',a.output/'formal/checkpoints/resume_latest.pt','--output',a.output/'benchmark_seq08_41'])
        result=json.loads((a.output/'evaluation/seq08_41/results.json').read_text())
        if result['status']!='EVALUATED' or result['frames']!=41:raise ValueError('Frozen41-frame quality coverage incomplete')
        mark('COMPLETE20K_AND_SEQ08_41',development900='CANCELLED_BY_USER; retained partial evidence',automatic_extra_training=False)
    except Exception as exc:
        mark('BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
