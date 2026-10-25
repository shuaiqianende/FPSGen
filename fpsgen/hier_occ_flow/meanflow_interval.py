"""Continuous interval embedding, preserving original time-module tensor names."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from torch import nn

_ACTIVE_INTERVAL = ContextVar('meanflow_interval', default=None)


class IntervalEmbedding(nn.Module):
    def __init__(self, output_channels, hidden_channels):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(32,hidden_channels),nn.SiLU(),nn.Linear(hidden_channels,output_channels))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,6.9,16)),persistent=False)

    def _apply(self,fn):
        original=self.frequencies.detach().float().clone()
        result=super()._apply(fn)
        self.frequencies=original.to(device=self.frequencies.device,dtype=torch.float32)
        return result

    def forward(self,duration):
        math_duration=duration if duration.dtype==torch.float64 else duration.float()
        angle=math_duration[:,None]*self.frequencies.to(math_duration)[None]
        features=torch.cat((angle.sin(),angle.cos()),-1).to(self.net[0].weight.dtype)
        return self.net(features)


class IntervalTimeEmbedding(nn.Sequential):
    """Original numeric layer names remain0,1,2; only interval.* is new.

    The context is local to a forward, never stored in a model/EMA/checkpoint.
    PTv3 block checkpoints receive the already-computed embedding tensor.
    """
    def __init__(self, original, output_channels, hidden_channels):
        layers=list(original.children())
        super().__init__(*layers)
        self.original_length=len(layers)
        self.interval=IntervalEmbedding(output_channels,hidden_channels)

    @contextmanager
    def condition(self,duration):
        token=_ACTIVE_INTERVAL.set((id(self),duration))
        try:yield
        finally:_ACTIVE_INTERVAL.reset(token)

    def forward(self,features):
        value=features
        for index in range(self.original_length):value=self[index](value)
        active=_ACTIVE_INTERVAL.get()
        if active is None or active[0]!=id(self):return value
        duration=active[1].reshape(-1)
        if len(duration)!=len(value):raise ValueError('One interval per scene required')
        residual=self.interval(duration).to(value.dtype)
        return value+duration.to(value.dtype)[:,None]*residual


def install_interval(base, name, output_channels, hidden_channels):
    original=getattr(base,name)
    if isinstance(original,IntervalTimeEmbedding):raise ValueError('Interval already installed')
    shared={k:v.detach().clone() for k,v in base.state_dict().items()}
    adapted=IntervalTimeEmbedding(original,output_channels,hidden_channels)
    parameter=next(original.parameters())
    adapted.interval.to(device=parameter.device,dtype=parameter.dtype)
    setattr(base,name,adapted)
    current=base.state_dict()
    if any(not torch.equal(v,current[k]) for k,v in shared.items()):raise RuntimeError('Shared source tensor changed')
    new=sorted(set(current)-set(shared))
    if not new or any(not k.startswith(name+'.interval.') for k in new):raise RuntimeError('Unexpected new parameters')
    return {'shared_tensor_count':len(shared),'shared_tensors_exact':True,'allowed_new_keys':new}


def scene_times(time,time_end,scenes,*,validate=True):
    if time is None or time_end is None or time.numel()!=scenes or time_end.numel()!=scenes:
        raise ValueError('Explicit current/end time[B] required')
    t=time.reshape(scenes);s=time_end.reshape(scenes)
    if validate and not bool((torch.isfinite(t)&torch.isfinite(s)&(t>=0)&(s>=t)&(s<=1)).all()):
        raise ValueError('Require0<=current_time<=endpoint<=1')
    return t,s
