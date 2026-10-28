"""Gate, independent32-step smoke/resume, fresh20K training, seq08/41 evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--data-audit',type=Path,required=True);a=p.parse_args()
    gate=json.loads((a.gate/'result.json').read_text())
    if gate['status']!='GPU_PASS' or gate['config_sha256']!=file_sha(a.config):raise ValueError('Own maximum Batch2 engineering gate required')
    for name,sha in gate['source_sha256'].items():
        if file_sha(ROOT/'fpsgen/hier_occ_flow'/name)!=sha:raise ValueError('Model source changed after gate')
    if a.root.exists():raise FileExistsError('Preserve previous experiment')
    a.root.mkdir(parents=True);pin=file_sha(a.config)
    env=os.environ.copy();env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']=str(a.data_audit.resolve());env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None)
    trainer=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py'
    def phase(name,command):
        if file_sha(a.config)!=pin:raise ValueError('Frozen config changed')
        atomic_json(a.root/'pipeline_status.json',dict(status='RUNNING_'+name.upper(),pid=os.getpid(),command=command,config_sha256=pin))
        with (a.root/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(a.root/'pipeline_status.json',dict(status='RUNNING_'+name.upper(),pid=os.getpid(),child_pid=child.pid,command=command,config_sha256=pin))
            if child.wait()!=0:raise RuntimeError(name+' failed; inspect log; no automatic fallback')
    try:
        smoke=a.root/'smoke';base=[sys.executable,'-u',str(trainer),'--config',str(a.config.resolve()),'--output',str(smoke.resolve()),'--smoke']
        phase('smoke_first2',base+['--max-updates','2'])
        phase('smoke_resume32',base+['--max-updates','32','--resume',str((smoke/'checkpoints/resume_latest.pt').resolve())])
        result=json.loads((smoke/'summary.json').read_text())
        if result['status']!='COMPLETE_REQUESTED_BUDGET' or result['successful_updates']!=32 or result['successful_frame_exposures']!=64:raise ValueError('Independent resumed32 smoke incomplete')
        formal=a.root/'formal';phase('formal',[sys.executable,'-u',str(trainer),'--config',str(a.config.resolve()),'--output',str(formal.resolve())])
        result=json.loads((formal/'summary.json').read_text())
        if result['status']!='COMPLETE_REQUESTED_BUDGET' or result['successful_updates']!=10000 or result['successful_frame_exposures']!=20000:raise ValueError('Fresh20K formal incomplete')
        phase('seq08_41',[sys.executable,'-u',str(ROOT/'scripts/research_v2/hier_occ_flow/eval_triplane_seq08.py'),'--config',str(a.config.resolve()),'--checkpoint',str((formal/'checkpoints/resume_latest.pt').resolve()),'--output',str((a.root/'evaluation/seq08_41').resolve())])
        atomic_json(a.root/'pipeline_status.json',dict(status='COMPLETE_20K_AND_SEQ08_41',pid=os.getpid(),config_sha256=pin))
    except Exception as e:
        atomic_json(a.root/'pipeline_status.json',dict(status='BLOCKED_REVIEW_REQUIRED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc(),pid=os.getpid()));raise

if __name__=='__main__':main()
