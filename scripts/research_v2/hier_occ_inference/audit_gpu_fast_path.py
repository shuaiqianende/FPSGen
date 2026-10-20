#!/usr/bin/env python3
"""Single-frame exact-parity audit for GPU selection/mask/decode and Stage2 cache."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE)]
import numpy as np
import torch
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.stage1_occ_adapter import FrozenStage1OccupancyAdapter
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.stage2_model import Stage2OccupancyModel
from fpsgen.hier_occ_flow.stage2_condition import scene_seed
from fpsgen.hier_occ_flow.stage2_candidates import select_occupancy_candidates
from fpsgen.hier_occ_flow.condition import sample_pyramid, observed_children
from fpsgen.hier_occ_flow.inference_fast import gpu_select_occupancy_candidates,gpu_observed_children,gpu_occupied_fine_coords
from fpsgen.hier_occ_flow.voxelization import occupied_fine_coords,pack_children

def elapsed_ms(fn,device):
    torch.cuda.synchronize(device); start=time.perf_counter(); out=fn(); torch.cuda.synchronize(device)
    return out,(time.perf_counter()-start)*1000

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physical-gpu',type=int,choices=(2,3),default=2)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage2-checkpoint',type=Path,required=True);p.add_argument('--frame',type=int,default=100)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.physical_gpu):raise RuntimeError('Expose only selected physical GPU')
    dev=torch.device('cuda:0');cfg,g=load_stage2_config(a.config)
    adapter=FrozenStage1OccupancyAdapter(**cfg['stage1']['adapter']).to(dev).eval()
    model=Stage2OccupancyModel(g,**cfg['model']).to(dev).eval()
    saved=torch.load(a.stage2_checkpoint,map_location='cpu');model.load_state_dict(saved['model'],strict=True)
    xyz_np=input_xyz(a.data_root/'08'/'input_'/f'{a.frame:06d}.npy').astype(np.float32,copy=False)
    xyz=torch.from_numpy(xyz_np).to(dev)
    with torch.inference_mode():
      features=adapter.encode_features(xyz)
      with torch.autocast('cuda',dtype=torch.float16):
        score,_=adapter.model.sample(xyz,nfe=2,condition=features,
            generator=torch.Generator(device=dev).manual_seed(scene_seed('08',a.frame,cfg['candidates']['seed'])))
      from export_stage2_snapshot import predicted_policy
      policy=predicted_policy(cfg);policy['stage1_nfe']=2
      ids_np,scores_np,prov=select_occupancy_candidates(score[0,0].float().cpu().numpy(),g,**policy['selection'])
      ids_gpu,score_gpu,gpu_prov=gpu_select_occupancy_candidates(score,g,**policy['selection'])
      torch.testing.assert_close(ids_gpu.cpu(),torch.from_numpy(ids_np),rtol=0,atol=0)
      torch.testing.assert_close(score_gpu.cpu(),torch.from_numpy(scores_np),rtol=0,atol=0)
      torch.testing.assert_close(ids_gpu,torch.sort(ids_gpu).values,rtol=0,atol=0)
      centers=g.centers(g.unflatten(ids_gpu)).float()
      lidar=sample_pyramid(features,centers,g.origin,g.maximum).detach().half()
      observed_cpu=torch.as_tensor(observed_children(xyz,ids_np,g),device=dev)
      observed_gpu=gpu_observed_children(xyz,ids_gpu,g)
      torch.testing.assert_close(observed_gpu,observed_cpu,rtol=0,atol=0)
      valid=g.valid_children(ids_gpu)
      cond={'lidar_features':lidar,'parent_score':score_gpu,'observed_child':observed_gpu,'valid_child_mask':valid}
      source=torch.randn((len(ids_gpu),g.k),device=dev,generator=torch.Generator(device=dev).manual_seed(cfg['validation']['seed']+a.frame))*valid
      def baseline_rollout():
        state=source.clone()
        for step in range(4):
          with torch.autocast('cuda',dtype=torch.float16):
            velocity=model(ids_gpu,state=state,time=state.new_tensor(step/4),**cond)
          state=state+velocity.float()/4
        return state
      def cached_rollout():
        state=source.clone()
        with torch.autocast('cuda',dtype=torch.float16):
          context=model.prepare_inference_context(ids_gpu,**cond)
        for step in range(4):
          with torch.autocast('cuda',dtype=torch.float16):
            velocity=model.forward_with_inference_context(context,state,state.new_tensor(step/4))
          state=state+velocity.float()/4
        return state
      # One untimed warm-up for each graph path; then three alternating timings.
      _=baseline_rollout();_=cached_rollout()
      btimes=[];ctimes=[]
      for _ in range(3):
        b,bms=elapsed_ms(baseline_rollout,dev);c,cms=elapsed_ms(cached_rollout,dev)
        btimes.append(bms);ctimes.append(cms)
      maxdiff=float((b-c).abs().max().item());mean_abs=float((b-c).abs().mean().item())
      occ_b=(b>0)&valid;occ_c=(c>0)&valid
      occ_mismatch=int((occ_b!=occ_c).sum().item())
      old_fine=occupied_fine_coords(ids_gpu.cpu().numpy(),pack_children(occ_b.cpu().numpy(),g),g)
      fast_fine=gpu_occupied_fine_coords(ids_gpu,occ_c,valid,g).cpu().numpy()
      same_coords=bool(np.array_equal(old_fine,fast_fine))
      report={'status':'PASS' if len(ids_gpu)==len(ids_np) and maxdiff<1e-4 and occ_mismatch==0 and same_coords else 'FAIL',
        'physical_gpu':a.physical_gpu,'frame':a.frame,'candidate_count':len(ids_np),
        'candidate_ids_exact':True,'candidate_scores_exact':True,'observed_child_exact':True,
        'candidate_provenance_cpu':prov,'candidate_provenance_gpu':gpu_prov,
        'stage2_endpoint_max_abs_diff':maxdiff,'stage2_endpoint_mean_abs_diff':mean_abs,
        'occupancy_bit_mismatch_count':occ_mismatch,'fine_coords_exact_and_ordered':same_coords,
        'stage2_baseline_4step_wall_ms_median':float(np.median(btimes)),
        'stage2_cached_4step_wall_ms_median':float(np.median(ctimes)),
        'stage2_speedup':float(np.median(btimes)/np.median(ctimes)),
        'valid_child_count':int(valid.sum().item()),'occupied_child_count':int(occ_b.sum().item())}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
