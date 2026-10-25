"""Explicitly authorized shared admission for bounded adaptation, never timing."""
import os
from .resources import gpu_preflight

ALLOCATOR_GIB = 12.
RESERVE_GIB = 2.


def adaptation_admission(physical, *, configure_allocator=False):
    shared = os.environ.get('DESESCENE_SHARED_ADAPTATION') == '1'
    if not shared:
        return gpu_preflight((physical,))
    if physical not in (2, 3) and not (physical==0 and os.environ.get('DESESCENE_GPU0_FLOW_CONTROL')=='1'):
        raise ValueError('Shared adaptation is authorized only on GPU2/3')
    report = gpu_preflight((physical,), minimum_free_gib=ALLOCATOR_GIB + RESERVE_GIB,
                           allow_shared_compute=True)
    report.update(scope='Batch2 adaptation / quality only; never formal latency',
                  allocator_limit_gib=ALLOCATOR_GIB, reserve_gib=RESERVE_GIB,
                  no_token_truncation=True)
    if configure_allocator and report['status'] == 'AVAILABLE':
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical):
            raise ValueError('One explicitly selected physical GPU required')
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_GIB * 2**30 / total, 0)
        print(f'Shared adaptation GPU{physical}: allocator cap12GiB, true Batch2; no scene/token reduction', flush=True)
    return report
