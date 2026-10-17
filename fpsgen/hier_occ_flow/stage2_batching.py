"""Ragged full-scene packing; no padding, truncation or cross-scene FM time."""
import torch


def pack_scenes(scenes):
    if not scenes:raise ValueError('Nonempty scene batch required')
    ids=[];targets=[];batch=[];conditions={};lengths=[]
    for index,(parent_ids,condition,target,provenance) in enumerate(scenes):
        if not len(parent_ids):raise ValueError('Empty scene cannot silently disappear')
        ids.append(parent_ids);targets.append(target);lengths.append(len(parent_ids))
        batch.append(torch.full_like(parent_ids,index))
        for name,value in condition.items():conditions.setdefault(name,[]).append(value)
    result={key:torch.cat(values) for key,values in conditions.items()}
    result['batch']=torch.cat(batch)
    return torch.cat(ids),result,torch.cat(targets),lengths


def packed_training_pair(occupancy,valid,batch,generator):
    if generator is None or bool(((occupancy!=0)&(occupancy!=1)).any()):raise ValueError('Binary labels and explicit source generator required')
    scenes=int(batch.max())+1
    source=torch.randn(occupancy.shape,device=occupancy.device,generator=generator)*valid
    target=(2*occupancy.float()-1)*valid
    time=torch.rand(scenes,device=occupancy.device,generator=generator)
    t=time[batch,None]
    return {'state':(1-t)*source+t*target,'source':source,'target':target,'velocity':target-source,'time':time}


def scene_mean_mse(prediction,target,valid,batch):
    """Each scene has equal weight, independent of its number of parent tokens."""
    loss=(prediction.float()-target.float()).square();values=[]
    for index in range(int(batch.max())+1):
        mask=valid[batch==index]
        if not bool(mask.any()):raise ValueError('Scene has no valid child voxels')
        values.append((loss[batch==index]*mask).sum()/mask.sum())
    return torch.stack(values).mean()
