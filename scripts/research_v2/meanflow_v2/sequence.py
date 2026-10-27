"""Only the two authorized arms: A gate/smoke/train/eval before B; stop on failure."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,subprocess,time,traceback
from fpsgen.hier_occ_flow.meanflow_v2_training import own_sha,load_config
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve previous sequence')
    a.output.mkdir(parents=True)
    pin=own_sha()
    atomic_json(a.output/'launch.json',dict(pid=os.getpid(),source_sha256=pin,epoch=time.time(),
        order=['MFV2-A','MFV2-B'],physical_gpu=2,batch=2,seed=20261032,maximum_formal_exposures_per_arm=20000,
        no_extra_arms=True,no_automatic_retry=True))
    completed=[]
    try:
        for arm in ('a','b'):
            if own_sha()!=pin:raise RuntimeError('Pinned source changed; do not continue')
            config=ROOT/f'configs/research_v2/hier_occ_flow/meanflow_v2_{arm}_gpu2.yaml'
            cfg,_=load_config(config)
            command=[sys.executable,'-u',str(ROOT/'scripts/research_v2/meanflow_v2/worker.py'),'--config',str(config),'--mode','pipeline']
            with (a.output/f'{arm}_pipeline.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(a.output/'status.json',dict(status='RUNNING_'+cfg['arm'],pid=os.getpid(),child_pid=child.pid,
                    completed=completed,source_sha256=pin,command=command))
                code=child.wait()
            if code:raise RuntimeError(f'{cfg["arm"]} exited{code}; later arms BLOCKED')
            state=json.loads((Path(cfg['output_root'])/'supervisor.json').read_text())
            if state['status']!='COMPLETE20K_AND_SEQ08_41':raise ValueError('A/B dependency completion missing')
            completed.append(cfg['arm'])
        atomic_json(a.output/'status.json',dict(status='COMPLETE_TWO_AUTHORIZED_ARMS',completed=completed,pid=os.getpid(),no_more_training=True))
    except Exception as exc:
        atomic_json(a.output/'status.json',dict(status='BLOCKED_REVIEW_REQUIRED',reason=str(exc),traceback=traceback.format_exc(),completed=completed,pid=os.getpid()))
        raise


if __name__=='__main__':main()
