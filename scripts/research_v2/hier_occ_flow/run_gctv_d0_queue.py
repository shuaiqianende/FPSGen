"""Run read-only 900-frame D0 models sequentially on one admitted GPU."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--models',nargs='+',choices=('r0','v1','v2'),required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('GPU mismatch')
    if a.root.exists():raise FileExistsError('Preserve old audit')
    a.root.mkdir(parents=True);script=ROOT/'scripts/research_v2/hier_occ_flow/audit_gctv_baseline_900.py';pin=file_sha(script)
    try:
        for name in a.models:
            if file_sha(script)!=pin:raise ValueError('Diagnostic changed while queued')
            command=[sys.executable,'-u',str(script),'--model',name,'--gpu',str(a.gpu),'--output',str((a.root/name).resolve())]
            with (a.root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(a.root/'queue_status.json',dict(status='RUNNING',model=name,gpu=a.gpu,pid=os.getpid(),child_pid=child.pid,source_sha256=pin,command=command))
                code=child.wait()
            if code:raise RuntimeError(f'{name} failed, exit {code}; no automatic retry')
            result=json.loads((a.root/name/'results.json').read_text())
            if result['status']!='COMPLETE' or result['frames']!=900:raise ValueError('Full900 diagnostic incomplete')
            time.sleep(2)
        atomic_json(a.root/'queue_status.json',dict(status='COMPLETE',models=a.models,gpu=a.gpu,source_sha256=pin))
    except Exception as e:
        atomic_json(a.root/'queue_status.json',dict(status='BLOCKED',model=name,gpu=a.gpu,reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
