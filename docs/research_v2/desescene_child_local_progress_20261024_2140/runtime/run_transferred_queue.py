
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-child-local-state-v1')
SCRIPTS=ROOT/'scripts/research_v2/hier_occ_flow'
sys.path[:0]=[str(ROOT),str(SCRIPTS)]
from fpsgen.hier_occ_flow.resources import gpu_preflight
from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json
from run_desescene_m1_matched_queue import health_report
Q=ROOT/'experiments/desescene_child_local_state_v1/pilot_pair_v1'
OUT=Path(__file__).resolve().parent
PYTHON='/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39/bin/python'
CONFIGS=ROOT/'configs/research_v2/hier_occ_flow'
contract=json.loads((Q/'contract.json').read_text());FROZEN=contract['source_sha256']
lock=(Q/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
transfer=json.loads((OUT/'transfer.json').read_text())
if source_sha()!=FROZEN:raise RuntimeError('Training source changed')

def run(name,gpu,script,args,completion,expected=None):
 if source_sha()!=FROZEN:raise RuntimeError('Training source changed before task')
 if completion.exists():
  value=json.loads(completion.read_text())
  if value.get('status') in ('FAIL','FAILED','BLOCKED'):raise RuntimeError('Retained failed artifact '+str(completion))
  if expected is None or value.get('successful_updates',0)>=expected:return
 while gpu_preflight((gpu,))['status']!='AVAILABLE':
  atomic_json(Q/(name+'.status.json'),{'status':'WAITING_GPU_IDLE','physical_gpu':gpu,'epoch':time.time()})
  time.sleep(20)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2',HS_STAGE2_ALLOW_SHARED_COMPUTE='0',TMPDIR='/data-12/M2024-HWZ/FPSGen/env/tmp',TORCH_EXTENSIONS_DIR='/data-12/M2024-HWZ/FPSGen/env/voxel_point_cache/torch_extensions')
 cmd=[('bash' if script.suffix=='.sh' else PYTHON),str(script),*map(str,args)]
 with (Q/(name+'.log')).open('a',buffering=1) as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  atomic_json(Q/(name+'.status.json'),{'status':'RUNNING','pid':child.pid,'physical_gpu':gpu,'command':cmd,'epoch':time.time()})
  code=child.wait()
 if code or not completion.exists():
  atomic_json(Q/(name+'.status.json'),{'status':'FAILED','exit_code':code,'epoch':time.time()})
  raise RuntimeError(name+' failed; retained log')
 if json.loads(completion.read_text()).get('status') in ('FAIL','FAILED','BLOCKED'):raise RuntimeError(name+' did not pass')
 atomic_json(Q/(name+'.status.json'),{'status':'COMPLETE','exit_code':0,'epoch':time.time()})

def evaluate(arm,gpu):
 folder=Q/(arm+'_20k');output=folder/'evaluation/primary900_nfe4.json'
 run(arm+'_eval900',gpu,SCRIPTS/'eval_stage2_384.py',['--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt','--output',output,'--ema','--max-frames',900,'--nfes',4,'--sampling-seeds',20261031,'--continuous-metrics'],output)
 output=folder/'evaluation/mechanisms64'
 run(arm+'_mechanisms64',gpu,SCRIPTS/'audit_desescene_mechanisms.py',['--config',folder/'config.json','--checkpoint',folder/'checkpoints/resume_latest.pt','--output',output,'--ema','--max-frames',64,'--mode','both'],output/'summary.json')

def position():
 folder=Q/'position_20k'
 for stop in (100,500,10000):
  args=['--config',CONFIGS/'stage2_384_child_local_position_20k.yaml','--output',folder,'--max-updates',stop]
  if (folder/'checkpoints/resume_latest.pt').exists():args+=['--resume',folder/'checkpoints/resume_latest.pt']
  run('position_train_'+str(stop),2,SCRIPTS/'launch_stage2_384.sh',args,folder/'summary.json',stop)
  health_report(folder,stop)
 evaluate('position',2)

def attention():
 folder=Q/'attention_20k';pid=transfer['adopted_attention_pid']
 while True:
  p=folder/'summary.json'
  if p.exists():
   value=json.loads(p.read_text())
   if value.get('successful_updates')==10000 and value.get('successful_frame_exposures')==20000:break
  proc=Path('/proc')/str(pid)
  try:
   cmd=(proc/'cmdline').read_bytes().decode().replace('\0',' ')
   stat=(proc/'stat').read_text().split(') ',1)[1].split()[0]
  except FileNotFoundError:raise RuntimeError('Adopted attention stopped before budget')
  if str(folder) not in cmd or 'train_stage2_384.py' not in cmd or stat=='Z':raise RuntimeError('Adopted attention PID identity changed or exited')
  atomic_json(OUT/'adopted_attention.status.json',{'status':'ADOPTED_RUNNING','pid':pid,'physical_gpu':3,'epoch':time.time()})
  time.sleep(20)
 # The original trainer writes the final summary only after atomic checkpointing.
 atomic_json(Q/'attention_train_10000.status.json',{'status':'COMPLETE','adopted_pid':pid,'epoch':time.time()})
 health_report(folder,10000)
 evaluate('attention',3)

atomic_json(OUT/'scheduler.json',{'status':'RUNNING','supervisor_pid':os.getpid(),'physical_gpus':{'position':2,'attention':3},'source_sha256':FROZEN,'epoch':time.time()})
with ThreadPoolExecutor(max_workers=2) as pool:
 futures={pool.submit(position):'position',pool.submit(attention):'attention'}
 from concurrent.futures import as_completed
 errors=[]
 for future in as_completed(futures):
  arm=futures[future]
  try:future.result();atomic_json(OUT/(arm+'.result.json'),{'status':'COMPLETE','epoch':time.time()})
  except Exception as exc:
   errors.append(arm+': '+str(exc));atomic_json(OUT/(arm+'.result.json'),{'status':'FAILED','reason':str(exc),'epoch':time.time()})
atomic_json(OUT/'summary.json',{'status':'FAILED' if errors else 'COMPLETE','errors':errors,'epoch':time.time()})
