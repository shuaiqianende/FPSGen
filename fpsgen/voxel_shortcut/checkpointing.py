"""PyTorch2.0 checkpoint recomputation without a second BatchNorm update."""

from contextlib import contextmanager,nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def isolated_batchnorm_buffers(module):
    originals=[]
    try:
        for child in module.modules():
            if isinstance(child,nn.modules.batchnorm._BatchNorm) and child.training and child.track_running_stats:
                values={name:getattr(child,name) for name in ('running_mean','running_var','num_batches_tracked')}
                originals.append((child,values))
                # Preserve original buffer identities/versions referenced by the
                # forward graph; recomputation uses disposable clones.
                for name,value in values.items():setattr(child,name,value.clone())
        yield
    finally:
        for child,values in originals:
            for name,value in values.items():setattr(child,name,value)


def checkpoint_once_batchnorm(module,*args):
    first=True
    def run(*inputs):
        nonlocal first
        initial=first;first=False
        with nullcontext() if initial else isolated_batchnorm_buffers(module):
            return module(*inputs)
    return checkpoint(run,*args,use_reentrant=False)
