"""Explicit inference GPU admission, independent of the GPU2/3 training policy."""
import os
import subprocess


def gpu0_processes():
    output = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    return {int(pid.strip()): float(memory.strip())
        for line in output.splitlines() if line.strip()
        for pid, memory in [line.split(',')]}


def check_gpu0_background(allowed_service_pid):
    processes = gpu0_processes()
    processes.pop(os.getpid(), None)
    for pid, memory in processes.items():
        if pid != allowed_service_pid or memory > 512:
            raise RuntimeError('GPU0 background workload changed; preserve existing jobs')
        command = subprocess.check_output(['ps', '-p', str(pid), '-o', 'comm='], text=True).strip()
        if command != 'uvicorn':
            raise RuntimeError('Only the explicitly identified idle service may share GPU0')
    return processes


def admit_gpu0(allowed_service_pid=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Explicit user-authorized physical GPU0 only')
    processes = check_gpu0_background(allowed_service_pid)
    output = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, util = map(float, output.strip().split(','))
    if free < 16384 or util > 10:
        raise RuntimeError('GPU0 is not sufficiently idle for bounded evaluation')
    return {'physical_gpu': 0, 'free_mib': free, 'utilization_percent': util,
        'existing_service_memory_mib': processes, 'allocator_memory_fraction': .25,
        'authorization': 'user: 0号卡测试最新ckpt，stage1+stage2联合推理',
        'training_gpu_policy_modified': False}
