"""Retry only a GPU-admission race that stopped before any formal update."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight

OLD=ROOT/'experiments/desescene_u1_abc_v1/u1a_queue';OUT=ROOT/'experiments/desescene_u1_abc_v1/u1a_shared_retry'
def main():
    old=json.loads((OLD/'status.json').read_text());log=(OLD/'train20k.log').read_text()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or OUT.exists() or old['status']!='BLOCKED' or 'Allowed GPUs busy; existing training preserved' not in log or (OLD/'formal').exists():raise ValueError('Not a pre-update admission-only failure')
    if any(file_sha(Path(k))!=v for k,v in old['source_sha256'].items()):raise ValueError('Pinned source changed')
    OUT.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    cfg=ROOT/'configs/research_v2/hier_occ_flow/patch_u1a_20k.yaml';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';evalfile=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py';formal=OLD/'formal';pins=old['source_sha256']
    def status(s,**extra):atomic_json(OUT/'status.json',dict(status=s,pid=os.getpid(),gpu=2,**extra))
    def child(label,cmd):
        if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('Pinned source changed')
        with (OUT/(label+'.log')).open('x') as f:
            proc=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+label,child_pid=proc.pid)
            return proc.wait()
    try:
        attempt=0
        while True:
            status('WAITING_GPU2_SHARED_ADMISSION',attempt=attempt,minimum_free_gib=14.5)
            while gpu_preflight((2,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
            attempt+=1;name=f'train20k_attempt{attempt}'
            if child(name,[train,'--config',cfg,'--output',formal])==0:break
            text=(OUT/(name+'.log')).read_text()
            if 'Allowed GPUs busy; existing training preserved' not in text or formal.exists():raise RuntimeError('Training failed after admission; inspect log; no silent retry')
        s=json.loads((formal/'summary.json').read_text())
        if s['successful_updates']!=10000 or s['successful_frame_exposures']!=20000:raise ValueError('20K not completed')
        if child('seq08_nfe2',[evalfile,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',OUT/'seq08_41']):raise RuntimeError('41 frame evaluation failed')
        status('COMPLETE_20K_AND_41',attempt=attempt)
    except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
