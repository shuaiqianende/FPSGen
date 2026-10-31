"""Independent 100K D1/D2 queues, with strict source-pinned resume."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.resources import gpu_preflight
PRIOR=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-u1-scratch100k-v1/experiments/desescene_u1_scratch100k_v1')
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(2,),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();arm='d1' if a.gpu==2 else 'd2';prior=PRIOR/f'gpu{a.gpu}_recovery2/status.json'
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.gpu):raise ValueError('wrong GPU')
 if a.resume:
  old=json.loads((a.output/'status.json').read_text())
  if old.get('child_pid') and subprocess.run(['ps','-p',str(old['child_pid'])],stdout=subprocess.DEVNULL).returncode==0:raise RuntimeError('old child still active')
  if any(file_sha(Path(k))!=v for k,v in old['source_sha256'].items()):raise RuntimeError('pinned source differs')
  ck=a.output/'formal/checkpoints/resume_latest.pt'
  if not ck.exists() or (a.output/'seq08_41').exists():raise ValueError('Need formal checkpoint and no partial evaluation')
 else:
  if a.output.exists():raise FileExistsError('Preserve prior output')
  a.output.mkdir(parents=True)
 env=os.environ.copy();env['HS_OCCFLOW_ALLOW_SHARED_GPU']='1';env['HS_OCCFLOW_VERIFIED_DATA_AUDIT']='/data-12/M2024-HWZ/FPSGen/worktrees/hs-occflow-dense-fidelity-v2-longrun/experiments/dense_fidelity_v2_single_gpu/verified_data_audit.json'
 cfg=ROOT/'configs/research_v2/hier_occ_flow/scratch100k_depth.yaml';gate=ROOT/'scripts/research_v2/hier_occ_flow/verify_triplane_gpu.py';train=ROOT/'scripts/research_v2/hier_occ_flow/train_dense_fidelity_single_gpu.py';ev=ROOT/'scripts/research_v2/hier_occ_flow/eval_u1_100k_seq08.py';bench=ROOT/'scripts/research_v2/hier_occ_flow/benchmark_d1_d2.py'
 files=[cfg,gate,train,ev,bench,Path(__file__),*[ROOT/'fpsgen/hier_occ_flow'/n for n in ('stage1_training.py','stage1_occupancy.py','stage1_network.py','patch_unet_condition.py','patch_fm_backbone.py','gctv_encoder.py','depth_moment_encoder.py')]];pins={str(x):file_sha(x) for x in files}
 def status(s,**kw):atomic_json(a.output/'status.json',dict(status=s,gpu=a.gpu,arm=arm,pid=os.getpid(),source_sha256=pins,**kw))
 def run(label,args):
  if any(file_sha(Path(k))!=v for k,v in pins.items()):raise RuntimeError('source changed')
  n=0
  while (a.output/f'{label}_{n}.log').exists():n+=1
  with (a.output/f'{label}_{n}.log').open('x') as f:
   c=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);status('RUNNING_'+label,child_pid=c.pid)
   if c.wait():raise RuntimeError(label+' failed; inspect log')
 try:
  status('WAITING_GPU')
  while gpu_preflight((a.gpu,),minimum_free_gib=14.5,allow_shared_compute=True)['status']!='AVAILABLE':time.sleep(30)
  if not a.resume:
   run('gate',[gate,'--config',cfg,'--output',a.output/'gate'])
   smoke=a.output/'smoke';base=[train,'--config',cfg,'--output',smoke,'--smoke'];run('smoke2',base+['--max-updates','2']);run('smoke32',base+['--max-updates','32','--resume',smoke/'checkpoints/resume_latest.pt'])
   s=json.loads((smoke/'summary.json').read_text());init=json.loads((smoke/'initialization_audit.json').read_text())
   if s['successful_updates']!=32 or s['successful_frame_exposures']!=64 or s['attempts']>34 or init['pretrained_loaded']:raise ValueError('Smoke or scratch check failed')
   if arm=='d1':expected='6efa2c64da812878dd70b883153bd6e04d897a864c19637f71b5b79a37875ddd'
   else:
    import torch
    from fpsgen.hier_occ_flow.patch_fm_backbone import PatchFlowUNet
    with torch.random.fork_rng(devices=[]):torch.manual_seed(20261032+104729);model=PatchFlowUNet(activation_checkpointing=False)
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.contiguous().numpy().tobytes())
    expected=h.hexdigest()
   if init['shared_flow_initialization_sha256']!=expected:raise ValueError('Shared FM initialization mismatch')
  formal=a.output/'formal';args=[train,'--config',cfg,'--output',formal]
  if a.resume:args+=['--resume',formal/'checkpoints/resume_latest.pt']
  run('train100k',args);s=json.loads((formal/'summary.json').read_text())
  if s['successful_updates']!=50000 or s['successful_frame_exposures']!=100000:raise ValueError('100K incomplete')
  run('seq08_41',[ev,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',a.output/'seq08_41'])
  run('benchmark',[bench,'--config',cfg,'--checkpoint',formal/'checkpoints/resume_latest.pt','--output',a.output/'timing.json']);status('COMPLETE_100K_AND_41')
 except Exception as e:status('BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc());raise
if __name__=='__main__':main()
