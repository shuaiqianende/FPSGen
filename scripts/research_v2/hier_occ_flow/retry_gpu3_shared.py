"""Resume GPU3 arms stopped at admission, after explicit shared-GPU authorization."""
from pathlib import Path
import json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight

OLD=ROOT/'experiments/desescene_u1_bc_v1/gpu3_queue';OUT=ROOT/'experiments/desescene_u1_bc_v1/gpu3_shared_retry'
def main():
    old=json.loads((OLD/'status.json').read_text())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or OUT.exists() or old['status']!='COMPLETE':raise ValueError('Unexpected queue state')
    if any(file_sha(Path(k))!=v for k,v in old['source_sha256'].items()):raise ValueError('Existing model source changed')
    for arm in ('u1ac','u1abc'):
        if (OLD/arm/'formal').exists() or 'Allowed GPUs busy; existing training preserved' not in (OLD/(arm+'_train20k.log')).read_text():raise ValueError('Not a pre-training GPU admission failure')
        s=json.loads((OLD/arm/'smoke/summary.json').read_text());g=json.loads((OLD/arm/'gate/result.json').read_text())
        if s['successful_updates']!=32 or g['status']!='GPU_PASS':raise ValueError('Prior smoke or gate failed')
    OUT.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json';results={}
    def state(status,**kw):atomic_json(OUT/'status.json',dict(status=status,gpu=3,pid=os.getpid(),results=results,**kw))
    def run(label,args,admission=False):
        if any(file_sha(Path(k))!=v for k,v in old['source_sha256'].items()):raise RuntimeError('Pinned source changed')
        if admission:
            while True:
                evidence=gpu_preflight((3,),minimum_free_gib=10.,allow_shared_compute=True)
                if evidence['status']=='AVAILABLE':break
                state('WAITING_SHARED_GPU3',arm=label,available=evidence);time.sleep(20)
            atomic_json(OUT/(label+'_admission.json'),dict(status='AVAILABLE',checked_at_epoch=time.time(),physical_ids=[3],inspection=evidence,justification='user authorized shared GPU; 10 GiB minimum free; prior Batch2 gate peak reserved about 6.4 GiB'))
            env['HS_OCCFLOW_ADMISSION']=str((OUT/(label+'_admission.json')).resolve())
        else:env.pop('HS_OCCFLOW_ADMISSION',None)
        with (OUT/(label+'.log')).open('x') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);state('RUNNING_'+label,child_pid=child.pid)
            if child.wait():raise RuntimeError(label+' failed; no automatic numerical retry')
    try:
        for arm in ('u1ac','u1abc'):
            cfg=ROOT/f'configs/research_v2/hier_occ_flow/patch_{arm}_20k.yaml';formal=OLD/arm/'formal';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_patch_seq08_nfe2.py'
            run(arm+'_train',[train,'--config',cfg,'--output',formal],admission=True)
            s=json.loads((formal/'summary.json').read_text())
            if s['successful_updates']!=10000 or s['successful_frame_exposures']!=20000:raise ValueError('20K exposure incomplete')
            run(arm+'_seq08',[ev,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',OUT/(arm+'_seq08_41')])
            results[arm]='COMPLETE_20K_AND_41'
        state('COMPLETE')
    except Exception as e:state('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
