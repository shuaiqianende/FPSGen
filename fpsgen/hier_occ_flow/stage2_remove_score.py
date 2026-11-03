"""Physically remove the score branch and fold neutral score into fuse bias."""
import copy,torch
from torch import nn
from torch.nn import functional as F


def convert_state(state):
    result={k:v.clone() for k,v in state.items() if not k.startswith('score.')}
    w=state['fuse.0.weight'];score=F.linear(F.silu(F.linear(w.new_ones((1,1)),state['score.0.weight'],state['score.0.bias'])),state['score.2.weight'],state['score.2.bias']).reshape(-1)
    if w.shape[1]<48 or len(score)!=16:raise ValueError('Unexpected score layout')
    result['fuse.0.bias']=state['fuse.0.bias']+w[:,32:48]@score
    result['fuse.0.weight']=torch.cat((w[:,:32],w[:,48:]),dim=1)
    return result


def remove_score(model,optimizer,ema):
    """Keep common Adam moments; slice fuse moments; bias moments unchanged."""
    old_names={id(p):n for n,p in model.named_parameters()}; old_groups=optimizer.param_groups;old_state=optimizer.state
    state=convert_state(model.state_dict());ema=convert_state(ema)
    old=model.fuse[0];new=nn.Linear(old.in_features-16,old.out_features,bias=old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
    model.fuse[0]=new;del model.score;model.load_state_dict(state,strict=True)
    new_names=dict(model.named_parameters());groups=[]
    for group in old_groups:
        params=[new_names[old_names[id(p)]] for p in group['params'] if not old_names[id(p)].startswith('score.')]
        groups.append({**{k:v for k,v in group.items() if k!='params'},'params':params})
    newopt=type(optimizer)(groups,**optimizer.defaults)
    for group in old_groups:
        for p in group['params']:
            name=old_names[id(p)]
            if name.startswith('score.') or p not in old_state:continue
            values=copy.deepcopy(old_state[p])
            if name=='fuse.0.weight':
                for k,v in list(values.items()):
                    if torch.is_tensor(v) and v.shape==p.shape:values[k]=torch.cat((v[:,:32],v[:,48:]),1)
            newopt.state[new_names[name]]=values
    return newopt,ema
