"""Resume healthy registered 2K GCTV runs to5K, then evaluate seq08/41."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--models',nargs='+',required=True);p.add_argument('--dependency',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu) or any(m not in ('g0','g1','g2') for m in a.models):raise ValueError('Registered GPU/models only')
    parent=a.dependency.parent;env=os.environ.copy();env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
    if a.root.exists():raise FileExistsError('Preserve earlier results')
    a.root.mkdir(parents=True)
    try:
        atomic_json(a.root/'status.json',dict(status='WAITING_FOR_2K_AND_41',gpu=a.gpu,pid=os.getpid()))
        while True:
            if a.dependency.exists():
                s=json.loads(a.dependency.read_text())
                if s['status']=='COMPLETE_2K_AND_41':break
                if s['status']=='BLOCKED':raise RuntimeError('2K dependency failed')
            time.sleep(20)
        for name in a.models:
            model=parent/name;formal=model/'formal';summary=json.loads((formal/'summary.json').read_text());quality=json.loads((model/'evaluation_2k_seq08_41/results.json').read_text())
            if summary['successful_updates']!=1000 or summary['successful_frame_exposures']!=2000 or quality['status']!='EVALUATED_SEQ08_41':raise ValueError('2K or41-frame evaluation incomplete')
            rows=[json.loads(line) for line in (formal/'train_rank0.jsonl').open()];finite=[r for r in rows if r['finite']]
            if len(finite)!=1000 or any(r['loss'] is None or r['loss']!=r['loss'] for r in finite) or rows[-1]['skipped_attempts']>max(2,int(len(rows)*.01)):
                raise ValueError('Training health failed; no continuation')
            cfg=ROOT/f'configs/research_v2/hier_occ_flow/gctv_{name}_20k.yaml';checkpoint=formal/'checkpoints/resume_latest.pt'
            commands=[('train5k',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py'),'--config',str(cfg),'--output',str(formal),'--max-updates','2500','--resume',str(checkpoint)]),('eval5k',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/eval_triplane_seq08.py'),'--config',str(cfg),'--checkpoint',str(checkpoint),'--output',str(model/'evaluation_5k_seq08_41')])]
            for phase,command in commands:
                with (a.root/(name+'_'+phase+'.log')).open('x') as f:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                    atomic_json(a.root/'status.json',dict(status='RUNNING_'+name+'_'+phase,gpu=a.gpu,pid=os.getpid(),child_pid=child.pid,source_checkpoint_sha256=file_sha(checkpoint),command=command))
                    if child.wait():raise RuntimeError(name+' '+phase+' failed; no automatic retry')
        atomic_json(a.root/'status.json',dict(status='COMPLETE_5K_AND_41',gpu=a.gpu,pid=os.getpid(),models=a.models,automatic_20k=False))
    except Exception as e:
        atomic_json(a.root/'status.json',dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),gpu=a.gpu,pid=os.getpid()));raise

if __name__=='__main__':main()
