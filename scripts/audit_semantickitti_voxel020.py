#!/usr/bin/env python3
"""Audit fixed 0.20 m GT candidate cardinality without writing dataset arrays."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from generate_semantickitti_poisson_gt import _full_candidate, voxel_downsample_indices
from prepare_semantickitti import load_map, load_poses

def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-root',type=Path,default=Path('/data-12/M2024-HWZ/KITTI_Odometry'));p.add_argument('--sequences',required=True);p.add_argument('--voxel-size',type=float,default=.20);p.add_argument('--target-points',type=int,default=180000);p.add_argument('--log-dir',type=Path,default=Path('outputs/research_v2/gt_poisson_generation'));a=p.parse_args();a.max_range=50.;a.min_z=-4.;a.max_z=None;a.viewpoint_voxel_size=10.;a.log_dir.mkdir(parents=True,exist_ok=True)
 rows=[]
 for seq in [x.strip() for x in a.sequences.split(',') if x.strip()]:
  d=a.data_root/seq; poses=load_poses(d); mx,ml=load_map(d/'map_clean.npy')
  for scan in sorted((d/'velodyne').glob('*.bin'),key=lambda x:int(x.stem)):
   frame=int(scan.stem)
   try:
    raw=_full_candidate(scan,poses[frame],mx,ml,a); count=len(voxel_downsample_indices(raw[:,:3],a.voxel_size)); rows.append({'sequence':seq,'frame':scan.stem,'raw_count':len(raw),'voxel020_count':count,'enough_for_180k':count>=a.target_points})
   except Exception as e: rows.append({'sequence':seq,'frame':scan.stem,'raw_count':'','voxel020_count':'','enough_for_180k':False,'error':f'{type(e).__name__}: {e}'})
 with (a.log_dir/'voxel020_counts.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['sequence','frame','raw_count','voxel020_count','enough_for_180k','error'],extrasaction='ignore');w.writeheader();w.writerows(rows)
 vals=np.array([r['voxel020_count'] for r in rows if isinstance(r['voxel020_count'],int)])
 summary={'total_frames':len(rows),'valid_count_frames':len(vals),'frames_ge_180k':int((vals>=a.target_points).sum()),'frames_lt_180k':int((vals<a.target_points).sum()),'voxel_size':a.voxel_size,'target_points':a.target_points}
 if len(vals): summary['voxel020_count']={k:float(v) for k,v in {'min':vals.min(),'p01':np.quantile(vals,.01),'p05':np.quantile(vals,.05),'p10':np.quantile(vals,.1),'median':np.median(vals),'mean':vals.mean(),'p90':np.quantile(vals,.9),'p95':np.quantile(vals,.95),'max':vals.max()}.items()}
 (a.log_dir/'voxel020_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
