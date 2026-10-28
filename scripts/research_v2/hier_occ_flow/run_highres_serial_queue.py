"""Per-GPU independent H experiments with staged21st-century exposure protocol."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import argparse,json,os,statistics,subprocess,time,traceback
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha

DATA_AUDIT='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
OLD=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-gctv-encoder-v1/experiments/desescene_gctv_encoder_v1')

def metrics(path):
    obj=json.loads(path.read_text())
    if obj['status']!='COMPLETE_DEV41':raise ValueError('Screen requires completed independent dev41')
    q=obj['result']['nfe_results']['2'];rows=q['rows']
    if len(rows)!=41:raise ValueError('Wrong development screen length')
    return q['summary']['macro']['iou'],statistics.median(r['encoder_seconds'] for r in rows[5:]),[(r['sequence'],r['frame']) for r in rows]

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,3),required=True);p.add_argument('--dependency',type=Path,required=True);p.add_argument('--peer',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--h1-gate',type=Path);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('Physical GPU mismatch')
    if a.root.exists():raise FileExistsError('Prior results preserved')
    a.root.mkdir(parents=True);env=os.environ.copy();env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']=DATA_AUDIT;env.pop('HS_OCCFLOW_ALLOW_SHARED_GPU',None)
    sources=[ROOT/'fpsgen/hier_occ_flow'/x for x in ('stage1_training.py','stage1_occupancy.py','triplane_encoder.py','gctv_encoder.py','gctv_highres_encoder.py')]
    sources += [ROOT/'configs/research_v2/hier_occ_flow'/f'gctv_{name}_20k.yaml' for name in ('g1','h1','h2','h3','h4','h5','h6')]
    sources += [ROOT/'scripts/research_v2/hier_occ_flow'/name for name in ('verify_triplane_gpu.py','train_dense_fidelity_single_gpu.py','eval_highres_dev41.py','eval_triplane_seq08.py','run_highres_serial_queue.py')]
    pin={str(p):file_sha(p) for p in sources}
    def phase(label,cmd):
        if any(file_sha(Path(k))!=v for k,v in pin.items()):raise RuntimeError('Pinned model source changed')
        with (a.root/(label+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
            atomic_json(a.root/'status.json',dict(status='RUNNING_'+label,gpu=a.gpu,pid=os.getpid(),child_pid=child.pid,command=cmd,source_sha256=pin))
            code=child.wait()
        if code:raise RuntimeError(label+' exited '+str(code))
    def wait_prior():
        atomic_json(a.root/'status.json',dict(status='WAITING_EXISTING_JOB',gpu=a.gpu,pid=os.getpid(),dependency=str(a.dependency)))
        while True:
            if a.dependency.exists():
                s=json.loads(a.dependency.read_text()).get('status','')
                if s in ('COMPLETE_20K_AND_41','COMPLETE_20K_AND_SEQ08_41'):return
                if s in ('BLOCKED','BLOCKED_REVIEW_REQUIRED'):raise RuntimeError('Existing task blocked; inspect before starting')
            time.sleep(20)
    def runner(name,root):
        cfg=ROOT/f'configs/research_v2/hier_occ_flow/gctv_{name}_20k.yaml';gate=root/'gate';smoke=root/'smoke';formal=root/'formal'
        verify=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';dev=ROOT/'scripts/research_v2/hier_occ_flow/eval_highres_dev41.py';final=ROOT/'scripts/research_v2/hier_occ_flow/eval_triplane_seq08.py'
        if name=='h1' and a.h1_gate:
            g=json.loads((a.h1_gate/'result.json').read_text())
            if g['status']!='GPU_PASS' or g['config_sha256']!=file_sha(cfg) or any(file_sha(ROOT/'fpsgen/hier_occ_flow'/k)!=v for k,v in g['source_sha256'].items()):raise ValueError('H1 gate identity changed')
            atomic_json(root/'preexisting_gate.json',g)
        else:phase(name+'_gate',[sys.executable,'-u',str(verify),'--config',str(cfg),'--output',str(gate)])
        base=[sys.executable,'-u',str(train),'--config',str(cfg),'--output',str(smoke),'--smoke']
        phase(name+'_smoke2',base+['--max-updates','2']);phase(name+'_smoke32',base+['--max-updates','32','--resume',str(smoke/'checkpoints/resume_latest.pt')])
        s=json.loads((smoke/'summary.json').read_text())
        if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34:raise ValueError('Smoke failed successful-update/AMP health')
        phase(name+'_train2k',[sys.executable,'-u',str(train),'--config',str(cfg),'--output',str(formal),'--max-updates','1000'])
        ck=formal/'checkpoints/resume_latest.pt';phase(name+'_dev2k',[sys.executable,'-u',str(dev),'--config',str(cfg),'--checkpoint',str(ck),'--output',str(root/'dev_2k'),'--gpu',str(a.gpu)])
        phase(name+'_train5k',[sys.executable,'-u',str(train),'--config',str(cfg),'--output',str(formal),'--max-updates','2500','--resume',str(ck)])
        phase(name+'_dev5k',[sys.executable,'-u',str(dev),'--config',str(cfg),'--checkpoint',str(ck),'--output',str(root/'dev_5k'),'--gpu',str(a.gpu)])
        # Always compare to same-card G1 5K as a stable matched baseline. This
        # prespecified screen does not tune using seq08. CUDA event/kernel
        # benchmarking remains a separate final measurement.
        basefile=a.root/'baseline_g1_5k/results.json'
        if not basefile.exists():
            bcfg=ROOT/'configs/research_v2/hier_occ_flow/gctv_g1_20k.yaml';bck=OLD/'2k_queue_gpu2/g1/formal/checkpoints/step002500.pt'
            phase('baseline_g1_5k',[sys.executable,'-u',str(dev),'--config',str(bcfg),'--checkpoint',str(bck),'--output',str(basefile.parent),'--gpu',str(a.gpu)])
        now,nt,ids=metrics(root/'dev_5k/results.json');ref,rt,rids=metrics(basefile)
        if ids!=rids:raise ValueError('Development identities differ')
        stop=now-ref<-.02 and nt>=.95*rt
        atomic_json(root/'screen_decision.json',dict(status='STOP_5K' if stop else 'CONTINUE_20K',screen='predefined: IoU delta below -2 percentage points AND no >5% encoder-wall improvement versus same-device G1 5K',dev_frames=41,iou=now,baseline_iou=ref,delta_iou=now-ref,encoder_wall_median=nt,baseline_encoder_wall_median=rt,screen_uses_seq08=False))
        if stop:return 'STOP_5K'
        phase(name+'_train20k',[sys.executable,'-u',str(train),'--config',str(cfg),'--output',str(formal),'--resume',str(ck)])
        s=json.loads((formal/'summary.json').read_text())
        if s['successful_updates']!=10000 or s['successful_frame_exposures']!=20000:raise ValueError('20K exposure target not reached')
        phase(name+'_seq08_41',[sys.executable,'-u',str(final),'--config',str(cfg),'--checkpoint',str(ck),'--output',str(root/'seq08_41')])
        return 'COMPLETE_20K_AND_41'
    def h6_supported():
        # Test independent 5K dev evidence of feedback H4-H1 and DCN H3-H2.
        # No checkpoint or threshold selection on seq08 is used.
        files=[a.root/'h4/dev_5k/results.json',a.peer/'h1/dev_5k/results.json',a.peer/'h3/dev_5k/results.json',a.peer/'h2/dev_5k/results.json']
        while not all(p.exists() for p in files):
            status=a.peer/'status.json'
            if status.exists() and json.loads(status.read_text()).get('status') in ('COMPLETE','BLOCKED'):return False,'peer completed without required 5K dev results'
            time.sleep(30)
        values=[metrics(p) for p in files]
        if any(x[2]!=values[0][2] for x in values):raise ValueError('H6 decision dev frames differ')
        feedback=values[0][0]-values[1][0];dcn=values[2][0]-values[3][0]
        return feedback>=.003 and dcn>=.003,f'dev41 5K Parent IoU H4-H1={feedback:+.6f}, H3-H2={dcn:+.6f}; require each >=+0.003'
    try:
        wait_prior();results={}
        for name in (('h1','h2','h3') if a.gpu==2 else ('h4','h5','h6')):
            root=a.root/name;root.mkdir()
            if name=='h6':
                atomic_json(a.root/'status.json',dict(status='WAITING_H6_EVIDENCE',gpu=a.gpu,pid=os.getpid()))
                supported,reason=h6_supported();atomic_json(root/'h6_decision.json',dict(status='GO' if supported else 'SKIPPED',reason=reason,uses_seq08=False))
                if not supported:results[name]='SKIPPED';continue
            try:results[name]=runner(name,root)
            except Exception as e:
                results[name]='BLOCKED';atomic_json(root/'failure.json',dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()))
                # Continue only after child process has exited; the next arm
                # independently checks admission and never changes its budget.
            atomic_json(a.root/'progress.json',dict(gpu=a.gpu,results=results))
        atomic_json(a.root/'status.json',dict(status='COMPLETE',gpu=a.gpu,pid=os.getpid(),results=results))
    except Exception as e:
        atomic_json(a.root/'status.json',dict(status='BLOCKED',gpu=a.gpu,pid=os.getpid(),reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
