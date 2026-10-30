"""Wait for both 100K arms and compute per-frame paired 2x2 contrasts."""
from pathlib import Path
import csv,json,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.storage import atomic_json
BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-u1-scratch100k-v1/experiments/desescene_u1_scratch100k_v1')
NEW=ROOT/'experiments/desescene_u1_d1_d2_100k_v1';OUT=NEW/'paired_recovery'
def main():
 if OUT.exists():raise FileExistsError('Prior summary retained')
 OUT.mkdir(parents=True)
 try:
  atomic_json(OUT/'status.json',dict(status='WAITING_D1_D2'))
  while True:
   files=[NEW/f'gpu{g}_requeued/status.json' for g in (2,3)]
   if all(f.exists() for f in files):
    statuses=[json.loads(f.read_text())['status'] for f in files]
    if 'BLOCKED' in statuses:raise RuntimeError('D1 or D2 blocked')
    if all(x=='COMPLETE_100K_AND_41' for x in statuses):break
   time.sleep(60)
  paths={'A_original':BASE/'gpu2_recovery2/u1a_seq08_41','AB_original':NEW/'gpu2_requeued/seq08_41','A_patch':NEW/'gpu3_requeued/seq08_41','AB_patch':BASE/'gpu3_recovery2/patch_seq08_41','equal_original':BASE/'gpu2_recovery/equal_seq08_41'}
  data={}
  for name,p in paths.items():
   rows=[json.loads(x) for x in (p/'per_frame.jsonl').read_text().splitlines()];selected={r['frame']:r for r in rows if r['nfe']==2}
   if set(selected)!=set(range(0,4001,100)):raise ValueError('frame identities differ: '+name)
   data[name]=selected
  pairs=[('AB_original','A_original'),('AB_patch','A_patch'),('A_patch','A_original'),('AB_patch','AB_original'),('AB_original','equal_original')]
  with (OUT/'per_frame_delta.csv').open('x',newline='') as f:
   w=csv.writer(f);w.writerow(['variant','reference','frame','parent_iou_delta','parent_recall_delta','fine_ceiling_delta','cd_m_delta','iou05_delta','iou02_delta','iou01_delta'])
   for variant,reference in pairs:
    for frame in range(0,4001,100):
     a,b=data[variant][frame],data[reference][frame];w.writerow([variant,reference,frame,*[a['parent'][k]-b['parent'][k] for k in ('iou','recall','fine_recall_ceiling')],*[a['continuous'][k]-b['continuous'][k] for k in ('cd_m','iou05','iou02','iou01')]])
  atomic_json(OUT/'status.json',dict(status='COMPLETE',frames=41,pairs=pairs,limitations='single training seed; shared GPU timings may not be matched'))
 except Exception as e:atomic_json(OUT/'status.json',dict(status='BLOCKED',reason=f'{type(e).__name__}: {e}',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
