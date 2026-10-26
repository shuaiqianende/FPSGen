"""Continue one already-registered MeanFlow arm, preserving failed peers.

Scheduling only: the training implementation and engineering identities remain
unchanged. This controller's own file hash is pinned separately.
"""
from pathlib import Path
import argparse,fcntl,json,os,subprocess,sys,time,traceback

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.meanflow_training import load_config
from fpsgen.hier_occ_flow.meanflow_control_audit import audit_control
from fpsgen.hier_occ_flow.meanflow_queue import project_jobs
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,choices=(2,3),required=True)
    a=p.parse_args();cfg,_=load_config(a.config);root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    pin=source_sha();config_pin=file_sha(a.config);controller_pin=file_sha(__file__)
    lock_path=ROOT.parent.parent/'experiments/desescene_meanflow_locks'/f'meanflow_gpu{a.physical_gpu}.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True);lock=lock_path.open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=root/'single_arm_supervisor.json'
    if status.exists():raise FileExistsError('Preserve existing controller identity; no automatic retry')
    def mark(state,**fields):
        atomic_json(status,dict(status=state,pid=os.getpid(),arm=cfg['arm'],physical_gpu=a.physical_gpu,
            source_sha256=pin,config_file_sha256=config_pin,controller_sha256=controller_pin,epoch=time.time(),**fields))
    def pinned():
        if source_sha()!=pin or file_sha(a.config)!=config_pin or file_sha(__file__)!=controller_pin:
            raise RuntimeError('Pinned model/config/controller changed')
    def wait_slot(exclusive=False):
        while True:
            pinned();r=gpu_preflight((a.physical_gpu,),minimum_free_gib=16.,allow_shared_compute=not exclusive)
            jobs=project_jobs(a.physical_gpu,r,ROOT.parent.parent)
            if r['status']=='AVAILABLE' and not jobs:return
            mark('WAITING_GPU',admission=r,project_jobs=jobs);time.sleep(15)
    def run(name,script,args,*,exclusive=False):
        wait_slot(exclusive=exclusive);pinned();task=root/(name+'.task.json')
        if task.exists():raise FileExistsError('Existing task evidence: '+str(task))
        command=[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow'/script),*map(str,args)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.physical_gpu),DESESCENE_MF_SHARED='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        with (root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            mark('RUNNING_'+name.upper(),child_pid=child.pid,command=command)
            atomic_json(task,dict(status='RUNNING',pid=child.pid,command=command,source_sha256=pin))
            code=child.wait()
        atomic_json(task,dict(status='COMPLETE' if code==0 else 'BLOCKED',exit_code=code,command=command,source_sha256=pin))
        if code:raise RuntimeError(f'{name} exited {code}; review required, no retry')
    try:
        cpu=json.loads((ROOT/'experiments/desescene_meanflow_jvp_v2/cpu_report.json').read_text())
        if cpu.get('status')!='CPU_PASS' or cpu.get('source_sha256')!=pin:
            raise ValueError('Source-matched CPU gate required')
        if Path(cfg['smoke_summary']).resolve()!=root/'smoke/summary.json':
            raise ValueError('Registered Smoke output path must match')
        if any((root/name).exists() for name in ('smoke','formal','evaluation','benchmark')):
            raise FileExistsError('This continuation only starts a never-trained arm; preserve existing artifacts')
        gate=json.loads(Path(cfg['engineering_report']).read_text())
        if any(gate.get(k)!=v for k,v in dict(status='GPU_PASS',arm=cfg['arm'],source_sha256=pin,config_sha256=config_sha(cfg)).items()):
            raise ValueError('Own source/config-matched engineering gate required')
        if [x['physical_gpu'] for x in gate['gpu_admission']['gpus']]!=[a.physical_gpu]:
            raise ValueError('Engineering must have been tested on the same physical GPU')
        control=audit_control(cfg);atomic_json(root/'single_arm_control_audit.json',control)
        if control['status']!='MATCHED_CONTROL':raise ValueError('Unmatched original-FM control')
        run('smoke32_gpu3','train_meanflow.py',['--config',a.config,'--output',root/'smoke','--max-updates',32,'--smoke'])
        proof=json.loads((root/'smoke/summary.json').read_text())
        if (proof.get('status')!='SMOKE_PASS' or proof.get('counts',{}).get('updates')!=32
            or proof['counts']['exposures']!=64 or proof['identity']['source_sha256']!=pin
            or proof['identity']['config_sha256']!=config_sha(cfg)
            or proof['identity']['engineering_report_sha256']!=file_sha(cfg['engineering_report'])):
            raise ValueError('Discarded Smoke did not pass with the same training identity')
        for count in (100,500,10000):
            args=['--config',a.config,'--output',root/'formal','--max-updates',count]
            if count!=100:args+=['--resume',root/'formal/checkpoints/resume_latest.pt']
            run(f'formal_gpu3_{count}','train_meanflow.py',args)
            proof=json.loads((root/'formal/summary.json').read_text())
            if proof['counts']['updates']!=count or proof['counts']['exposures']!=2*count:
                raise ValueError('Successful exposure budget mismatch')
        for split in ('development900','seq08_41'):
            run(split,'eval_meanflow.py',['--config',a.config,'--checkpoint',root/'formal/checkpoints/resume_latest.pt',
                '--output',root/'evaluation'/split,'--split',split])
        run('paired_timing','benchmark_meanflow.py',['--config',a.config,'--checkpoint',root/'formal/checkpoints/resume_latest.pt',
            '--output',root/'benchmark'],exclusive=True)
        mark('COMPLETE_TRAINING_AND_EVALUATION',exposures=20000,automatic_extra_training=False)
    except Exception as exc:
        mark('BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
