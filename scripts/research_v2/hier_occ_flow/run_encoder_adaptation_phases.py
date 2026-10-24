"""Bounded P0->P1->P2 supervisor. No P3 training and no preemption."""
from common import ROOT
import argparse,fcntl,json,os,shutil,subprocess,sys,threading,time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import load_adaptation_config

SCRIPTS=ROOT/'scripts/research_v2/hier_occ_flow'
CONFIGS=ROOT/'configs/research_v2/hier_occ_flow'
PROJECT=Path('/data-12/M2024-HWZ/FPSGen')
P0=PROJECT/'worktrees/desescene-child-local-state-v1/experiments/desescene_child_local_state_v1/pilot_pair_v1'
B0=PROJECT/'worktrees/desescene-state-condition-v1/experiments/desescene_state_condition_v1/matched20k_gpu1_gpu3_v2/b0_20k'
CENSUS=PROJECT/'worktrees/desescene-state-condition-v1/experiments/desescene_state_condition_v1/phase0/d3_fixed_budget/per_frame.jsonl'
BRANCH='research/desescene-encoder-adaptation-v1'
PRIVATE_URL='https://github.com/shuaiqianende/FPSGen-private.git'


def read(path):return json.loads(Path(path).read_text())


def health(folder,minimum):
    value=read(folder/'summary.json')
    if (value.get('status')!='COMPLETE_REQUESTED_BUDGET' or value['successful_updates']<minimum or value['successful_updates']>10000
        or value['successful_frame_exposures']!=value['successful_updates']*2
        or value.get('freeze_violation') or value.get('amp_skips',0)>max(2,int(value['attempts']*.01))):
        raise RuntimeError('Training health/budget/freeze check failed: '+str(folder))
    return value


class Queue:
    def __init__(self,folder,publish):
        if folder!=(ROOT/'experiments/desescene_encoder_adaptation_v1').resolve():
            raise ValueError('Artifact root is pinned by registered config; do not create an unregistered cohort')
        self.folder=folder;self.publish=publish;self.frozen=source_sha()
        self.gpu_lock=threading.Lock();self.gpu_leases=set()
        self.config_files={str(p):file_sha(p) for p in sorted(CONFIGS.glob('encoder_adaptation_*_20k.yaml'))}
        self.status_path=folder/'supervisor.json'
        self.lock=(folder/'supervisor.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=folder/'registration.json'
        if previous.exists() and read(previous)['source_sha256']!=self.frozen:raise RuntimeError('Registered queue source changed; do not overwrite its identity')
        if previous.exists() and read(previous).get('config_file_sha256')!=self.config_files:raise RuntimeError('Registered queue configuration changed')
        cfg,_=load_adaptation_config(CONFIGS/'encoder_adaptation_ef1_20k.yaml')
        sources={}
        for name in ('stage1','stage2'):
            spec=cfg['source_'+name];actual=file_sha(spec['checkpoint'])
            if actual!=spec['checkpoint_sha256']:raise RuntimeError('Source checkpoint hash mismatch: '+name)
            sources[name]={'path':spec['checkpoint'],'sha256':actual,'expected_sha256':spec['checkpoint_sha256'],'match':True}
        receipt=folder/'source_checkpoint_audit.json'
        if not receipt.exists():atomic_json(receipt,sources)
        atomic_json(folder/'registration.json',{'pid':os.getpid(),'source_sha256':self.frozen,
            'branch':BRANCH,'gpus':[2,3],'no_preemption':True,'phase_order':['P0','P1','P2'],
            'p3_formal_training':False,'seed':20261032,'successful_exposures_per_formal_arm':20000,
            'config_file_sha256':self.config_files,
            'checkpoint_sources_audited':sources})

    def status(self,state,**extra):
        atomic_json(self.status_path,{'status':state,'pid':os.getpid(),'epoch':time.time(),'source_sha256':self.frozen,**extra})

    def intact(self):
        if source_sha()!=self.frozen:raise RuntimeError('Registered source changed; stopped before starting another task')
        if any(file_sha(p)!=digest for p,digest in self.config_files.items()):raise RuntimeError('Registered configuration changed')

    def idle(self,gpu):
        while True:
            self.intact();admission=gpu_preflight((gpu,))
            if admission['status']=='AVAILABLE':return
            atomic_json(self.folder/f'gpu{gpu}_wait.json',{'status':'WAITING_GPU_IDLE','admission':admission,'epoch':time.time()})
            time.sleep(20)

    @contextmanager
    def lease(self,preferred=2):
        # Prefer the registered card, but make useful sequential progress on
        # the other idle card when an unrelated job occupies the first one.
        gpu=None
        while gpu is None:
            self.intact()
            with self.gpu_lock:
                for candidate in (preferred,3 if preferred==2 else 2):
                    if candidate not in self.gpu_leases and gpu_preflight((candidate,))['status']=='AVAILABLE':
                        gpu=candidate;self.gpu_leases.add(gpu);break
            if gpu is None:time.sleep(20)
        try:yield gpu
        finally:
            with self.gpu_lock:self.gpu_leases.remove(gpu)

    def run(self,name,gpu,script,args,result,*,expected=None):
        self.intact();status=self.folder/'tasks'/f'{name}.json';status.parent.mkdir(exist_ok=True)
        if result.exists():
            value=read(result)
            if value.get('status') in ('FAIL','FAILED','BLOCKED','STOPPED_HEALTH_FAILURE'):raise RuntimeError('Retained failure; manual review required: '+str(result))
            if expected is None or value.get('successful_updates',0)>=expected:return
        self.idle(gpu)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',
            HS_STAGE2_ALLOW_SHARED_COMPUTE='0',TMPDIR=str(PROJECT/'env/tmp'),
            TORCH_EXTENSIONS_DIR=str(PROJECT/'env/voxel_point_cache/torch_extensions'))
        command=[sys.executable,str(SCRIPTS/script),*map(str,args)]
        with (self.folder/'tasks'/f'{name}.log').open('a',buffering=1) as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(status,{'status':'RUNNING','pid':child.pid,'physical_gpu':gpu,'command':command,'epoch':time.time()})
            code=child.wait()
        value=read(result) if result.exists() else {}
        if code or not result.exists() or value.get('status') in ('FAIL','FAILED','BLOCKED','STOPPED_HEALTH_FAILURE'):
            atomic_json(status,{'status':'FAILED','exit_code':code,'epoch':time.time(),'reason':value.get('reason')})
            raise RuntimeError(name+' failed; no fallback/reduced batch/automatic retry')
        if expected is not None and value.get('successful_updates',0)<expected:raise RuntimeError('Successful budget not reached')
        atomic_json(status,{'status':'COMPLETE','exit_code':0,'epoch':time.time()})

    def config(self,arm):return CONFIGS/f"encoder_adaptation_{arm.lower().replace('-','_')}_20k.yaml"

    def engineering(self,arm,gpu):
        out=self.folder/'engineering'/arm
        self.run(arm+'_maximum_b2',gpu,'verify_encoder_adaptation_gpu.py',
            ['--config',self.config(arm),'--output',out,'--p0-root',P0,'--census',CENSUS],out/'result.json')
        report=read(out/'result.json')
        if report['source_sha256']!=self.frozen or report['status']!='PASS':raise RuntimeError('Invalid engineering artifact')
        out=self.folder/'smokes'/arm
        self.run(arm+'_smoke32',gpu,'train_encoder_adaptation.py',
            ['--config',self.config(arm),'--output',out,'--max-updates',32,'--smoke'],out/'summary.json',expected=32)
        health(out,32)

    def train(self,arm,gpu):
        if shutil.disk_usage(self.folder).free<50*2**30:raise RuntimeError('Less than50GiB free; preserve historical artifacts and stop new training')
        folder=self.folder/'runs'/arm
        for cap in (100,500,10000):
            args=['--config',self.config(arm),'--output',folder,'--max-updates',cap]
            checkpoint=folder/'checkpoints/resume_latest.pt'
            if checkpoint.exists():args+=['--resume',checkpoint]
            self.run(arm+'_train_'+str(cap),gpu,'train_encoder_adaptation.py',args,folder/'summary.json',expected=cap)
            health(folder,cap)

    def evaluate(self,arm,gpu,*,reference=False):
        output=self.folder/'evaluations'/f'{arm}.json'
        args=['--config',self.config('EF1' if arm=='EF0' else arm),'--output',output]
        args+=['--reference'] if reference else ['--checkpoint',self.folder/'runs'/arm/'checkpoints/resume_latest.pt']
        self.run(arm+'_eval900',gpu,'eval_encoder_adaptation.py',args,output)

    def timing(self,phase,arms,gpu,legacy=False):
        for existing in (self.folder/phase/'timing').glob('round*/*.json'):
            if read(existing)['physical_gpu']!=gpu:raise RuntimeError('Retained timing uses another GPU; resume on that same card, never mix hardware')
        for round_id in (1,2,3):
            # Alternating whole-arm order; every invocation rewarms5 frames.
            order=arms if round_id%2 else list(reversed(arms))
            for arm in order:
                output=self.folder/phase/'timing'/f'round{round_id}'/f'{arm}.json'
                if legacy:
                    folder=B0 if arm=='B0' else P0/({'L0':'position','L1':'attention'}[arm]+'_20k')
                    args=['--legacy-stage2','--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt']
                else:
                    args=['--config',self.config('EF1' if arm=='EF0' else arm)]
                    args+=['--reference'] if arm=='EF0' else ['--checkpoint',self.folder/'runs'/arm/'checkpoints/resume_latest.pt']
                self.run(f'{phase}_{arm}_timing{round_id}',gpu,'benchmark_adaptation_generation.py',args+['--output',output],output)

    def publish_phase(self,phase,report):
        # Only compact, allowlisted results are committed. Never stage data,
        # checkpoints, execution caches, or unrelated worktree changes.
        destination=ROOT/'docs/research_v2/desescene_encoder_adaptation_v1'
        destination.mkdir(exist_ok=True)
        value=read(report)
        if phase=='P0':
            compact={k:v for k,v in value.items() if k not in ('branch_activity','checkpoint_audits')}
            compact['checkpoint_audits']={k:{x:v[x] for x in ('checkpoint_sha256','successful_updates','successful_exposures','amp_skips','identity')} for k,v in value['checkpoint_audits'].items()}
            activities=value['branch_activity'];compact['activity_rows']=len(activities)
            compact['activity_mean']={arm:{m:sum(r[m] for r in activities if r['arm']==arm and m in r)/sum(m in r for r in activities if r['arm']==arm)
                for m in ('base_rms','residual_rms','relative_residual_rms','attention_entropy','attention_max_weight','attention_diagonal_mass')
                if any(m in r for r in activities if r['arm']==arm)} for arm in ('position','attention')}
        else:
            compact=value  # Aggregates/worst frames only; no900-frame tensors.
        atomic_json(destination/f'{phase}_results.json',compact)
        (destination/f'{phase}_results.md').write_text(report.with_suffix('.md').read_text() if report.with_suffix('.md').exists() else (report.parent/'REPORT.md').read_text())
        if not self.publish:return
        if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!=BRANCH:raise RuntimeError('Wrong publish branch')
        staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
        if staged:raise RuntimeError('Unexpected staged files; do not mix with phase report')
        for key in ('remote.private.url','remote.private.pushurl'):
            got=subprocess.run(['git','config','--get',key],cwd=ROOT,text=True,capture_output=True)
            if got.returncode==0 and got.stdout.strip()!=PRIVATE_URL:raise RuntimeError('Private remote changed')
        paths=[str((destination/f'{phase}_results.{suffix}').relative_to(ROOT)) for suffix in ('json','md')]
        subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            subprocess.run(['git','commit','-m',f'research: record {phase} encoder adaptation evidence'],cwd=ROOT,check=True)
        subprocess.run(['git','push',PRIVATE_URL,'HEAD:refs/heads/'+BRANCH],cwd=ROOT,check=True)

    def aggregate(self,phase,arms,reference):
        phase_root=self.folder/phase;(phase_root/'evaluations').mkdir(parents=True,exist_ok=True)
        for arm in arms:
            link=phase_root/'evaluations'/f'{arm}.json'
            if not link.exists():link.symlink_to(self.folder/'evaluations'/f'{arm}.json')
        output=phase_root/'results.json'
        if not output.exists():subprocess.run([sys.executable,str(SCRIPTS/'aggregate_encoder_adaptation.py'),'--phase',phase,
            '--root',str(phase_root),'--reference',str(reference),'--arms',*arms,'--output',str(output)],cwd=ROOT,check=True)
        self.publish_phase(phase,output)

    def pair(self,fn,arms):
        def worker(arm,preferred):
            with self.lease(preferred) as gpu:fn(arm,gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,arm,gpu) for arm,gpu in zip(arms,(2,3))]
            for future in futures:future.result()

    def execute(self):
        audit=self.folder/'data_access_audit.json'
        if not audit.exists():
            subprocess.run([sys.executable,str(SCRIPTS/'audit_encoder_adaptation_data.py'),'--config',str(self.config('EF1')),
                '--output',str(audit)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),check=True)
        value=read(audit)
        if value['status']!='PASS' or not value['shard_hashes_verified'] or value['missing_original_inputs']:
            raise RuntimeError('Original registered inputs/GT not available; no substitute dataset')
        self.status('WAITING_P0_EXISTING_QUEUE')
        while True:
            self.intact();summary=P0/'gpu2_transfer/summary.json'
            if summary.exists():
                value=read(summary)
                if value['status']=='FAILED':raise RuntimeError('Existing P0 queue failed; preserved without duplicate restart')
                if value['status']=='COMPLETE':break
            time.sleep(20)
        self.status('P0_COMPLETION_AUDIT')
        p0out=self.folder/'P0/audit'
        with self.lease(2) as gpu:
            self.run('P0_audit',gpu,'complete_local_state_audit.py',['--p0-root',P0,'--baseline-root',B0,'--output',p0out],p0out/'summary.json')
            self.timing('P0',['B0','L0','L1'],gpu,legacy=True)
        self.publish_phase('P0',p0out/'summary.json')
        self.status('P1_ENGINEERING_AND_SMOKE')
        self.pair(self.engineering,['EF1','EF2'])
        self.status('P1_MATCHED_20K');self.pair(self.train,['EF1','EF2'])
        self.status('P1_EVALUATION_900');self.pair(self.evaluate,['EF1','EF2'])
        with self.lease(2) as gpu:
            self.evaluate('EF0',gpu,reference=True)
            self.timing('P1',['EF0','EF1','EF2'],gpu)
        self.aggregate('P1',['EF1','EF2'],self.folder/'evaluations/EF0.json')
        self.status('P2_ENGINEERING_AND_SMOKE')
        self.pair(self.engineering,['S2-E1','S2-E2'])
        with self.lease(2) as gpu:self.engineering('S2-E0',gpu)
        # A two-scene smoke cannot establish equivalence for17870 training
        # frames. Retain B0 as a historical comparison; use a new registered
        # Stage2-only matched control for the B2 differentiable runner.
        atomic_json(self.folder/'P2_control_decision.json',{'reuse_b0_for_primary_claim':False,
            'reason':'Real B2 Encoder representation / backward path requires own matched control; maximum-pair comparison alone is insufficient',
            'registered_control':'S2-E0','budget':20000,'additional_architectures':False})
        self.status('P2_MATCHED_CONTROL')
        with self.lease(2) as gpu:self.train('S2-E0',gpu)
        self.status('P2_MATCHED_20K');self.pair(self.train,['S2-E1','S2-E2'])
        self.status('P2_EVALUATION_900');self.pair(self.evaluate,['S2-E1','S2-E2'])
        with self.lease(2) as gpu:
            self.evaluate('S2-E0',gpu)
            self.timing('P2',['EF0','S2-E0','S2-E1','S2-E2'],gpu)
        self.aggregate('P2',['S2-E1','S2-E2'],self.folder/'evaluations/S2-E0.json')
        self.status('COMPLETE_P1_P2_P3_RESEARCH_DECISION_REQUIRED',p3_formal_training_started=False)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=ROOT/'experiments/desescene_encoder_adaptation_v1')
    p.add_argument('--publish-phases',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    q=Queue(a.root.resolve(),a.publish_phases)
    try:q.execute()
    except Exception as exc:
        q.status('BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
