"""Independent endpoint-conditioned adapter around the unchanged occupancy UNet."""
import torch
from torch import nn
from .meanflow_interval import install_interval,scene_times


class MeanFlowStage1(nn.Module):
    def __init__(self,flow):
        super().__init__()
        if flow.objective!='fm':raise ValueError('Original FM UNet required')
        self.base=flow
        self.initialization_audit=install_interval(flow,'embedding',128,128)

    def _apply(self,fn):
        frequency=self.base.frequencies.detach().float().clone()
        result=super()._apply(fn)
        self.base.frequencies=frequency.to(self.base.frequencies.device)
        return result

    def forward(self,state,observed,condition,time,*,time_end,validate_times=True,inference_context=None):
        t,s=scene_times(time,time_end,len(state),validate=validate_times)
        with self.base.embedding.condition(s-t):
            if inference_context is not None:
                if inference_context['observed'] is not observed:raise ValueError('Observed-mask context changed')
                return self.base._forward(state,observed,None,t,
                    condition_projected=inference_context['condition_projected'],validate_time_range=False)
            return self.base._forward(state,observed,condition,t,validate_time_range=False)

    def prepare_inference_context(self,observed,condition):
        return self.base.prepare_inference_context(observed,condition)

    def forward_with_inference_context(self,state,context,time,*,time_end):
        return self(state,context['observed'],None,time,time_end=time_end,inference_context=context)
