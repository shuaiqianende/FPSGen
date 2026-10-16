"""Read-only GPU admission; never compete with an existing compute process."""
import os
import subprocess


def gpu_preflight(physical_ids=(2,3),minimum_free_gib=16.):
    if any(p not in (2,3) for p in physical_ids): raise ValueError('Only physical GPU2/3 allowed')
    rows=[]; blocked=False
    for physical in physical_ids:
        r=subprocess.run(['nvidia-smi',f'--id={physical}','--query-gpu=memory.free,memory.total,utilization.gpu,driver_version','--format=csv,noheader,nounits'],text=True,capture_output=True)
        if r.returncode: return {'status':'BLOCKED','reason':'nvidia-smi unavailable','detail':r.stderr+r.stdout}
        free,total,util,driver=[x.strip() for x in r.stdout.strip().split(',')]
        processes=subprocess.run(['nvidia-smi',f'--id={physical}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,capture_output=True)
        pids=[int(x.strip()) for x in processes.stdout.splitlines() if x.strip().isdigit()]
        unavailable=bool(pids) or float(free)<minimum_free_gib*1024 or float(util)>10
        blocked |= unavailable
        rows.append({'physical_gpu':physical,'free_mib':float(free),'total_mib':float(total),'utilization_percent':float(util),
                     'driver':driver,'existing_compute_pids':pids,'available':not unavailable})
    return {'status':'BLOCKED' if blocked else 'AVAILABLE','gpus':rows,'minimum_free_gib':minimum_free_gib,
            'policy':'No existing compute process; no stopping/reclaiming memory from existing jobs'}
