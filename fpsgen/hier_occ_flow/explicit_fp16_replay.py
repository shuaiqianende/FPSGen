"""Preregistered half-storage numerical replay; legacy FP32 bounds are unchanged.

Every RNG/integer/step/shape remains exact. Half weights use8 storage ULP plus
1e-6 around zero; FP32 masters and EMA use1e-6+1e-5|x|. Global gradient relative
L2 must be <=0.005. This is numerical replay, never bitwise certification.
"""
import hashlib,json,math
import numpy as np
import torch

POLICY={'version':'explicit-fp16-master-replay-v1','half_storage_ulps':8,'half_storage_near_zero_atol':1e-6,
 'master_parameters':{'atol':1e-6,'rtol':1e-5},'ema':{'atol':1e-6,'rtol':1e-5},
 'model_fp32_buffers':{'atol':1e-6,'rtol':1e-5},'optimizer_m1':{'atol':5e-5,'rtol':5e-3},'optimizer_m2':{'atol':5e-7,'rtol':1e-2},
 'prediction':{'atol':1e-3,'rtol':5e-3},'target':{'atol':1e-3,'rtol':5e-3},'derivative':{'atol':2e-3,'rtol':5e-3},
 'gradient':{'atol':5e-4,'rtol':5e-3},'loss':{'atol':1e-5,'rtol':1e-4},'gradient_norm':{'atol':1e-3,'rtol':1e-3},
 'gradient_global_relative_l2_limit':5e-3}
POLICY_SHA=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest()


def compare(expected,actual):
    differences=[];violations=[];totals=[0.,0.]
    def visit(a,b,path):
        category=path.split('/')[0]
        if isinstance(a,torch.Tensor):
            if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape:violations.append(dict(path=path,reason='shape_dtype'));return
            if torch.equal(a,b):
                if category=='gradient':totals[1]+=float(a.double().square().sum())
                return
            if not a.is_floating_point() or category in ('rng','interval_rng') or path.endswith('/step'):violations.append(dict(path=path,reason='required_exact'));return
            x=a.double();y=b.double();d=(x-y).abs();finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            if a.dtype==torch.float16 and category=='model':
                _,exp=torch.frexp(x.abs());ulp=torch.pow(torch.tensor(2.,dtype=torch.double),exp-11).clamp_min(2**-24)
                limit=POLICY['half_storage_ulps']*ulp+POLICY['half_storage_near_zero_atol'];ok=finite and bool((d<=limit).all())
            else:
                key='model_fp32_buffers' if category=='model' else category
                if category=='optimizer':key='optimizer_m1' if path.endswith('/exp_avg') else 'optimizer_m2' if path.endswith('/exp_avg_sq') else 'exact'
                bound=POLICY.get(key)
                ok=finite and isinstance(bound,dict) and bool((d<=bound['atol']+bound['rtol']*x.abs()).all())
            item=dict(path=path,max_abs=float(d.max()),mean_abs=float(d.mean()),relative_l2=float(d.norm()/(x.norm()+1e-30)),within_bounds=ok)
            differences.append(item)
            if not ok:violations.append(item)
            if category=='gradient':totals[0]+=float(d.square().sum());totals[1]+=float(x.square().sum())
        elif isinstance(a,np.ndarray):
            if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):violations.append(dict(path=path,reason='numpy_rng'))
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():violations.append(dict(path=path,reason='keys'));return
            for k in a:visit(a[k],b[k],str(k) if not path else path+'/'+str(k))
        elif isinstance(a,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):violations.append(dict(path=path,reason='sequence'));return
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif type(a)!=type(b) or a!=b:
            bound=POLICY.get(path);ok=isinstance(bound,dict) and isinstance(a,float) and isinstance(b,float) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=bound['atol']+bound['rtol']*abs(a)
            if not ok:violations.append(dict(path=path,reason='scalar'))
    visit(expected,actual,'');rel=math.sqrt(totals[0]/(totals[1]+1e-60))
    if rel>POLICY['gradient_global_relative_l2_limit']:violations.append(dict(path='gradient',reason='global_relative_l2',value=rel))
    return dict(status='PASS' if not violations else 'FAIL',policy=POLICY,policy_sha256=POLICY_SHA,exact=not differences and not violations,
        differences=differences,violations=violations,gradient_global_relative_l2=rel,scope='same-state one native update only; dtype-aware NUMERICAL_REPLAY, not geometry-quality certification')
