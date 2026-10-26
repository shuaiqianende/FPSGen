"""Dependency-verified, source-pinned followup queue. Never stops an existing GPU job."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,fcntl,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.queued20k_training import own_sha,load_config
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha
from fpsgen.hier_occ_flow.storage import file_sha,atomic_json
from fpsgen.hier_occ_flow.resources import gpu_preflight


def completed(dep):
    root=Path(dep['root']);summary=root/'formal/summary.json';quality=root/'evaluation/seq08_41/results.json'
    if (root/'formal/failure.json').exists():raise RuntimeError('Predecessor training failed; review, do not auto-retry')
    supervisor=root/'supervisor.json'
    if supervisor.exists() and json.loads(supervisor.read_text()).get('status','').startswith(('BLOCKED','USER_HOLD')):raise RuntimeError('Predecessor held or blocked; do not bypass')
    if not summary.exists() or not quality.exists():return False,'Waiting predecessor formal20K and41-frame quality'
    s=json.loads(summary.read_text());q=json.loads(quality.read_text());identity=s['identity']
    if s['status']!='TRAINING_COMPLETE' or s['counts']['updates']!=10000 or s['counts']['exposures']!=20000 or identity['initialization_mode']!='scratch':raise ValueError('Predecessor is not a completed scratch20K')
    if identity['arm']!=dep['arm'] or identity['config_sha256']!=dep['config_sha256'] or identity['source_sha256']!=dep['source_sha256']:raise ValueError('Predecessor code/config/arm mismatch')
    if q['status']!='EVALUATED' or q['frames']!=41 or q['protocol']['frames']!=list(range(0,4001,100)):raise ValueError('Require complete original41 quality, not900/partial')
    return True,'Verified predecessor20K+41'


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());pin=own_sha();plan_sha=file_sha(a.plan);queue=Path(plan['queue_root']);jobs=[j for j in plan['jobs'] if j['physical_gpu']==a.gpu]
    if len(jobs)!=(1 if a.gpu==2 else 2):raise ValueError('Only three user-authorized followups')
    proof=json.loads(Path(plan['cpu_report']).read_text())
    if proof['status']!='CPU_PASS' or proof['source_sha256']!=pin:raise ValueError('Current-source CPU_PASS required')
    for job in jobs:
        cfg,g=load_config(job['config'])
        if cfg['physical_gpu']!=a.gpu or cfg['arm']!=job['id'] or config_sha(cfg)!=job['config_sha256'] or file_sha(job['config'])!=job['config_file_sha256']:raise ValueError('Pinned queued job identity changed')
    if a.check_only:
        print(json.dumps(dict(status='QUEUE_READY_WAITING_DEPENDENCIES',physical_gpu=a.gpu,jobs=[j['id'] for j in jobs],source_sha256=pin,gpu_checks='PENDING each actual GPU release; never waived')));return
    queue.mkdir(parents=True,exist_ok=True)
    lock=(queue/f'gpu{a.gpu}.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=queue/f'gpu{a.gpu}.launch.json'
    if launch.exists():raise FileExistsError('Prior controller evidence exists; review before restart')
    atomic_json(launch,dict(pid=os.getpid(),physical_gpu=a.gpu,source_sha256=pin,plan_sha256=plan_sha,epoch=time.time(),jobs=[j['id'] for j in jobs]))
    def mark(status,job=None,**extra):atomic_json(queue/f'gpu{a.gpu}.json',dict(status=status,pid=os.getpid(),physical_gpu=a.gpu,job=job,epoch=time.time(),source_sha256=pin,**extra))
    def pinned():
        if own_sha()!=pin or file_sha(a.plan)!=plan_sha:raise RuntimeError('Pinned queue/model/plan changed')
        for job in jobs:
            if file_sha(job['config'])!=job['config_file_sha256']:raise ValueError('Queued config changed')
    try:
        for job in jobs:
            while True:
                pinned();dependencies=[completed(d) for d in job['depends_on']]
                if not all(x[0] for x in dependencies):mark('WAITING_PREDECESSOR_TRAIN_AND_QUALITY41',job['id'],reasons=[x[1] for x in dependencies if not x[0]]);time.sleep(20);continue
                gpu=gpu_preflight((a.gpu,),minimum_free_gib=22.,allow_shared_compute=False)
                if gpu['status']!='AVAILABLE':mark('WAITING_EXCLUSIVE_GPU',job['id'],admission=gpu);time.sleep(20);continue
                break
            task=queue/(job['id']+'.task.json')
            if task.exists():raise FileExistsError('Existing followup task; no duplicate')
            command=[sys.executable,'-u',str(ROOT/'scripts/research_v2/queued20k/worker.py'),'--config',job['config'],'--mode','pipeline']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            with (queue/(job['id']+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(task,dict(status='RUNNING_ENGINEERING_SMOKE_FORMAL_THEN41',pid=child.pid,command=command));mark('RUNNING_FOLLOWUP_PIPELINE',job['id'],child_pid=child.pid)
                code=child.wait()
            atomic_json(task,dict(status='COMPLETE' if code==0 else 'BLOCKED',exit_code=code,command=command))
            if code:raise RuntimeError(f'{job["id"]} exited{code}; preserve evidence, no automatic retry')
            dep=dict(root=job['output_root'],arm=job['id'],config_sha256=job['config_sha256'],source_sha256=pin)
            if not completed(dep)[0]:raise ValueError('Followup incomplete')
        mark('ALL_REGISTERED_FOLLOWUPS_COMPLETE',automatic_extra_training=False)
    except Exception as exc:
        mark('BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
