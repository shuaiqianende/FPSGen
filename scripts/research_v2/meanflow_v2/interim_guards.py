"""Explicit interim evaluation proof and opt-in sharing with our own A process."""
import os
import subprocess
from pathlib import Path
from fpsgen.hier_occ_flow.storage import file_sha
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import config_sha


def validate_interim_checkpoint(saved,cfg,expected_updates,training_source_sha,geometry_sha):
    i,c=saved['identity'],saved['counts']
    if not 0<expected_updates<10000 or c['updates']!=expected_updates or c['exposures']!=2*expected_updates:
        raise ValueError('Explicit intermediate successful budget required; never label as final20K')
    if (i['initialization_mode']!='scratch' or i['arm']!='MFV2-A' or i['config_sha256']!=config_sha(cfg)
        or i['source_sha256']!=training_source_sha or i['geometry_sha256']!=geometry_sha
        or c['attempts']!=c['updates']+c['amp_skips'] or c['cursor']!=2*c['attempts']
        or saved['scheduler']['last_epoch']!=c['updates']):
        raise ValueError('Interim checkpoint identity/RNG/counter/schedule mismatch')


def validate_own_sharing(report,training_pid,self_pid=None):
    allowed={training_pid}
    if self_pid is not None:allowed.add(self_pid)
    row=report['gpus'][0]
    if row['free_mib']<8*1024 or not set(row['existing_compute_pids']).issubset(allowed):
        raise RuntimeError('Shared interim evaluation admits only our own A training and >=8GiB free')


def admission(cfg,training_pid,training_root):
    import torch
    from fpsgen.hier_occ_flow.resources import gpu_preflight
    if cfg['physical_gpu']!=2 or os.environ.get('CUDA_VISIBLE_DEVICES')!='2':
        raise ValueError('Explicit shared evaluation is confined to our A GPU2')
    proc=Path(f'/proc/{training_pid}')
    if proc.stat().st_uid!=os.getuid():raise RuntimeError('Training process is not ours')
    command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if str(training_root/'scripts/research_v2/meanflow_v2/worker.py') not in command or '--mode formal' not in command:
        raise RuntimeError('Expected live original A formal process')
    report=gpu_preflight((2,),minimum_free_gib=8.,allow_shared_compute=True)
    validate_own_sharing(report,training_pid)
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    report.update(authorized_scope='User-requested interim evaluation sharing only with own A; no foreign task sharing',
                  evaluator_torch_allocator_cap_gib=6,training_pid=training_pid)
    return report


def prove_training_source(root,training_root,expected_sha):
    # Run the ORIGINAL source hash implementation in its original worktree.
    actual=subprocess.check_output([__import__('sys').executable,'-c',
        'from fpsgen.hier_occ_flow.meanflow_v2_training import own_sha; print(own_sha())'],
        cwd=training_root,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True).strip()
    if actual!=expected_sha:raise ValueError('Original A training source no longer matches checkpoint')
    # This evaluator adds scripts only; all original model/vendor/math code and
    # original evaluation source must remain byte-identical.
    folders=['fpsgen/hier_occ_flow','fpsgen/voxel_point/vendor/voxdet','fpsgen/voxel_point/vendor/pwm/ptv3_reference','fpsgen/voxel_shortcut']
    paths=[p.relative_to(training_root) for folder in folders for p in (training_root/folder).rglob('*.py')]
    paths.append(Path('scripts/research_v2/meanflow_v2/evaluate.py'))
    for rel in paths:
        if file_sha(root/rel)!=file_sha(training_root/rel):raise ValueError(f'Changed inference/model implementation: {rel}')
    return dict(original_source_sha256=actual,original_root=str(training_root),unchanged_model_vendor_files=len(paths))
