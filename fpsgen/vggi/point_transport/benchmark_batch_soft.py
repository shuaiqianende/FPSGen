"""Interleaved B=1/2/4 real-scene soft target benchmarking, no training."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from .soft_flow import reference_match as single
from .batch_soft import match as batch
from .quality import set_metrics,collapse_metrics
from .export_ply import write_ply,rainbow_rgb

CONFIG=dict(k=32,epsilons=(.02,.005,.002),penalty=50.,iterations=30,step_size=.9,debias=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--batches',nargs='+',type=int,default=[1,2,4]);p.add_argument('--coordinate-probe',action='store_true');args=p.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.65)
    scenes=['00_000500','00_001000','01_000500'];arrays=[];hashes={}
    for scene in scenes:
        path=Path('fpsgen/vggi/runs/point_transport_poisson_gt_v1')/scene/'inputs.npz'
        with np.load(path) as d:arrays.append((d['source'],d['target']))
        hashes[scene]=hashlib.sha256(path.read_bytes()).hexdigest()
    allrows=[]
    for b in args.batches:
        ids=[i%3 for i in range(b)]
        x,y=[torch.tensor(np.stack([arrays[i][j] for i in ids]),device='cuda') for j in (0,1)]
        names=['serial_fp32','serial_fused_fp32','batch_wide_fp32','batch_fp32','batch_fp16','batch_bf16']
        if args.coordinate_probe and b==1:names+=['coords_fp16','coords_bf16']
        def call(name):
            if name in ('serial_fp32','serial_fused_fp32'):
                outputs=[];records=[]
                for i in range(b):
                    if name=='serial_fp32':z,info=single(x[i],y[i],**CONFIG)
                    else:
                        z,info=batch(x[i:i+1],y[i:i+1],**CONFIG);z=z[0]
                    outputs.append(z);records.append(info)
                return torch.stack(outputs),dict(samples=records)
            if name.startswith('coords_'):
                dtype=torch.float16 if name.endswith('fp16') else torch.bfloat16
                return batch(x.to(dtype).float(),y.to(dtype).float(),**CONFIG)
            if name=='batch_wide_fp32':return batch(x,y,compact_indices=False,**CONFIG)
            return batch(x,y,cost_precision=name.split('_')[1],**CONFIG)
        records={name:[] for name in names};outputs={};peaks={}
        for name in names:call(name)
        for repeat in range(args.repeats):
            order=names[repeat%len(names):]+names[:repeat%len(names)]
            for name in order:
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
                z,info=call(name);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                records[name].append(dict(seconds=elapsed,diagnostics=info));peaks[name]=max(peaks.get(name,0),torch.cuda.max_memory_allocated()/2**30)
                outputs[name]=z.cpu().numpy();del z
        for name in names:
            folder=root/f'b{b}_{name}';folder.mkdir(exist_ok=True)
            quality=[]
            for i,sceneidx in enumerate(ids):
                z=outputs[name][i];base=outputs['serial_fp32'][i];source,target=arrays[sceneidx]
                delta=np.linalg.norm(z.astype(float)-base.astype(float),axis=1);move=np.linalg.norm(z.astype(float)-source.astype(float),axis=1)
                q=dict(scene=scenes[sceneidx],slot=i,set=set_metrics(z,target),collapse=collapse_metrics(z),
                    displacement=dict(mean_m=float(move.mean()),rms_m=float(np.sqrt(np.mean(move*move)))),
                    delta_vs_serial=dict(rms_m=float(np.sqrt(np.mean(delta*delta))),max_m=float(delta.max()),p99_m=float(np.quantile(delta,.99))))
                quality.append(q);np.save(folder/f'{i}_{scenes[sceneidx]}.endpoints.npy',z)
                write_ply(folder/f'{i}_{scenes[sceneidx]}.ply',z,rainbow_rgb(source))
            med=float(np.median([r['seconds'] for r in records[name]]));maximum=max(r['seconds'] for r in records[name])
            result=dict(batch=b,backend=name,median_seconds=med,max_seconds=maximum,per_scene_seconds=med/b,max_allocated_gib=peaks[name],config=CONFIG,
                scene_order=[scenes[i] for i in ids],note='B4 repeats first real scene in slot 3; same physical coordinates are intentional isolation stress, not four distinct scenes.',
                timings='Includes full matching and returned diagnostics, fresh trees every call; excludes JIT/H2D/IO/CPU quality/probability-field sampling. Interleaved backend order.',
                input_sha256=hashes,runs=records[name],quality=quality,hard_assignment=False,no_training=True)
            (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            row=dict(batch=b,backend=name,median_seconds=med,max_seconds=maximum,per_scene_seconds=med/b,max_allocated_gib=peaks[name],
                cd_cm_mean=float(np.mean([q['set']['cd_l1_mean_m']*100 for q in quality])),dcd_mean=float(np.mean([q['set']['dcd_alpha_100'] for q in quality])),
                f1_mean=float(np.mean([q['set']['prf_0.1m']['f1'] for q in quality])),
                delta_rms_max_m=max(q['delta_vs_serial']['rms_m'] for q in quality))
            print(json.dumps(row),flush=True);allrows.append(row)
            with (root/'comparison.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(allrows)
        del x,y
    print('Completed all requested batches.',flush=True)
if __name__=='__main__':main()
