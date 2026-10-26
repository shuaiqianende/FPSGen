"""Versioned numerical-replay bounds, registered before the v2 GPU gate.

Every RNG/nonfloating/control value remains exact. Native CUDA float reductions
may differ within the fixed limits below, only after the same source FM control
passes the identical check on the same real maximum Batch2 and physical GPU.
These bounds certify local replay, not cross-seed or long-trajectory identity.
"""
import hashlib,json,math
import numpy as np
import torch

POLICY={
    'version':'native-replay-v2-20261026',
    'model':{'atol':1e-7,'rtol':1e-6},
    'ema':{'atol':1e-7,'rtol':1e-6},
    'optimizer_m1':{'atol':5e-6,'rtol':5e-4},
    'optimizer_m2':{'atol':5e-8,'rtol':1e-3},
    'gradient':{'atol':5e-5,'rtol':5e-4},
    'prediction':{'atol':5e-4,'rtol':1e-6},
    'target':{'atol':5e-4,'rtol':1e-6},
    'derivative':{'atol':1e-3,'rtol':1e-6},
    'loss':{'atol':1e-7,'rtol':1e-6},
    'gradient_norm':{'atol':3e-5,'rtol':5e-6},
    'gradient_global_relative_l2_limit':5e-4,
}
POLICY_SHA=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()).hexdigest()


def compare_native_replay(expected,actual):
    differences=[];violations=[];totals={'gradient_d2':0.,'gradient_ref2':0.}
    def visit(a,b,path):
        if isinstance(a,torch.Tensor):
            if not isinstance(b,torch.Tensor) or a.dtype!=b.dtype or a.shape!=b.shape:
                violations.append({'path':path,'reason':'shape_or_dtype'});return
            if torch.equal(a,b):
                if path.startswith('gradient/') and a.is_floating_point():totals['gradient_ref2']+=float(a.double().square().sum())
                return
            category=path.split('/')[0]
            if not a.is_floating_point() or path.endswith('/step'):
                violations.append({'path':path,'reason':'required_exact_state'});return
            if category=='optimizer':
                category='optimizer_m2' if path.endswith('/exp_avg_sq') else 'optimizer_m1' if path.endswith('/exp_avg') else 'exact'
            bound=POLICY.get(category)
            if not isinstance(bound,dict):
                violations.append({'path':path,'reason':'unregistered_floating_state'});return
            x=a.double();y=b.double();d=(x-y).abs()
            finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            ok=finite and bool((d<=bound['atol']+bound['rtol']*x.abs()).all())
            item={'path':path,'max_abs':float(d.max()),'mean_abs':float(d.mean()),
                  'relative_l2':float(torch.linalg.vector_norm(d)/(torch.linalg.vector_norm(x)+1e-30)),
                  'different_elements':int((a!=b).sum()),'within_bounds':ok}
            differences.append(item)
            if not ok:violations.append(item)
            if path.startswith('gradient/'):
                totals['gradient_d2']+=float(d.square().sum());totals['gradient_ref2']+=float(x.square().sum())
        elif isinstance(a,np.ndarray):
            if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or not np.array_equal(a,b):violations.append({'path':path,'reason':'numpy_rng_or_state_changed'})
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():violations.append({'path':path,'reason':'keys'});return
            for k in a:visit(a[k],b[k],str(k) if not path else path+'/'+str(k))
        elif isinstance(a,(tuple,list)):
            if type(a)!=type(b) or len(a)!=len(b):violations.append({'path':path,'reason':'sequence'});return
            for k,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(k))
        elif type(a)!=type(b):violations.append({'path':path,'reason':'scalar_type'})
        elif a!=b:
            bound=POLICY.get(path)
            ok=isinstance(bound,dict) and isinstance(a,float) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=bound['atol']+bound['rtol']*abs(a)
            if ok:differences.append({'path':path,'max_abs':abs(a-b),'within_bounds':True})
            if not ok:violations.append({'path':path,'reason':'scalar_value','expected':repr(a),'actual':repr(b)})
    visit(expected,actual,'')
    relative=math.sqrt(totals['gradient_d2']/(totals['gradient_ref2']+1e-60))
    if relative>POLICY['gradient_global_relative_l2_limit']:violations.append({'path':'gradient','reason':'global_relative_l2','value':relative})
    return dict(status='PASS' if not violations else 'FAIL',policy=POLICY,policy_sha256=POLICY_SHA,
        exact=not differences and not violations,gradient_global_relative_l2=relative,
        float_tensor_differences=differences,violations=violations,
        scope='same maximum real Batch2, same source/RNG/state, one native update; numerical replay only')
