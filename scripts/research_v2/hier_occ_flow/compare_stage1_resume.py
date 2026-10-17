"""Independent CUDA run comparison; exact cursor/RNG and registered AMP tolerances.

DeformConv and CUDA reductions are nondeterministic. Floating tolerance is
atol=1e-3, rtol=1e-2, plus exact attempts, optimizer step, RNG and frame order;
this is not a claim of bit-identical CUDA training.
"""
from common import ROOT
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.storage import atomic_json


def compare(a,b,*,atol=1e-3,rtol=1e-2):
    diffs=[];fail=[]
    def walk(x,y,path):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:fail.append(path);return
            if x.is_floating_point() and not path.endswith('/step'):
                delta=float((x-y).abs().max()) if x.numel() else 0.
                limit=atol+rtol*(float(x.abs().max()) if x.numel() else 0.)
                diffs.append(delta)
                if not torch.isfinite(y).all() or delta>limit:fail.append(path)
            elif not torch.equal(x,y):fail.append(path)
        elif isinstance(x,np.ndarray):
            if not np.array_equal(x,y):fail.append(path)
        elif isinstance(x,dict):
            if set(x)!=set(y):fail.append(path);return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):fail.append(path);return
            for i,(u,v) in enumerate(zip(x,y)):walk(u,v,path+'/'+str(i))
        elif x!=y:fail.append(path)
    walk(a,b,'')
    return {'pass':not fail,'maximum_float_abs_difference':max(diffs,default=0.),'failed_paths':fail[:30]}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--continuous',type=Path,required=True);p.add_argument('--resumed',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--same-origin',action='store_true');p.add_argument('--atol',type=float,default=1e-3);p.add_argument('--rtol',type=float,default=1e-2)
    args=p.parse_args();a=torch.load(args.continuous/'checkpoints/resume_latest.pt',map_location='cpu');b=torch.load(args.resumed/'checkpoints/resume_latest.pt',map_location='cpu')
    summary=args.resumed/'summary.json';logfile=args.resumed/'train_rank0.jsonl'
    if not summary.exists() or summary.stat().st_mtime_ns<logfile.stat().st_mtime_ns:
        raise RuntimeError('Wait for resumed run to finish and commit its checkpoint before comparison')
    rows={k:compare(a[k],b[k],atol=0.,rtol=0.) if k in ('sampler','rank_rng','scaler','successful_updates','attempts','initialization') else compare(a[k],b[k])
        for k in ('model','ema','optimizer','scheduler','scaler','sampler','rank_rng','successful_updates','attempts','initialization')}
    for k in ('model','ema','optimizer','scheduler'):rows[k]=compare(a[k],b[k],atol=args.atol,rtol=args.rtol)
    logs=[]
    for root in (args.continuous,args.resumed):
        logs.append([json.loads(line) for line in (root/'train_rank0.jsonl').read_text().splitlines()])
    order=lambda data:[(r['attempts'],tuple((x['sequence'],x['frame']) for x in r.get('batch_records',[r]))) for r in data]
    if args.same_origin:
        if len(logs[1])<=len(logs[0]):raise RuntimeError('No newly resumed update has been recorded')
        logs[1]=list({r['attempts']:r for r in logs[1]}.values())
    rows['frame_order']={'pass':order(logs[0])==order(logs[1])}
    atomic_json(args.output,{'status':'PASS_CUDA_TOLERANCE' if all(v['pass'] for v in rows.values()) else 'FAIL',
        'atol':args.atol,'rtol':args.rtol,'same_origin':args.same_origin,'bit_identical_claim':False,'rows':rows,'continuous_updates':a['successful_updates'],'resumed_updates':b['successful_updates']})

if __name__=='__main__':main()
