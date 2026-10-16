"""Publish measured milestones only; pending formal/30min states stay explicit."""
from common import ROOT
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,default=Path('experiments/hier_occ_flow'));args=p.parse_args()
    run=args.run_root;docs=ROOT/'docs/research_v2';reports=docs/'hs_occ384_stage1_reports';reports.mkdir(parents=True,exist_ok=True)
    for name in ('stage1_cpu_model_audit','stage1_gpu2_f64_gate','stage1_gpu3_f64_gate','stage1_f64_same_origin_resume_audit','stage1_ddp_resume_audit','viewpoint_lookup_audit'):
        if (run/f'{name}.json').exists():shutil.copyfile(run/f'{name}.json',reports/f'{name}.json')
    suites=ET.parse(run/'stage1_cpu_tests.xml').getroot()
    tests={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) for k in ('tests','failures','errors','skipped')}
    tests['passed']=tests['tests']-tests['failures']-tests['errors']-tests['skipped'];atomic_json(reports/'cpu_tests.json',tests)
    rows=[read(p) for p in (run/'occ384_full_parallel/parts').glob('*/progress.json')]
    data={'requested_train':17870,'requested_development':900,'guard':360,'seq08_excluded':True,
        'voxelized_frames':sum(r['done_frames'] for r in rows),'committed_frames':sum(r['committed_frames'] for r in rows),
        'full_dataset_ready':(run/'occ384_full_parallel/full_data_audit.json').exists(),
        'output':str((ROOT/run/'occ384_full_parallel').resolve()),'preprocessing_tmux':'fpsgen_occ384_rawmap_parallel',
        'workers':8,'fine_grid':[768,768,64],'split_manifest_sha256':file_sha(run/'occ384_full/split_manifest.json')}
    atomic_json(reports/'data_progress.json',data)
    dataset=read(run/'occ384_full_parallel/full_data_audit.json')
    if dataset:
        slim={k:v for k,v in dataset.items() if k!='rows'};atomic_json(reports/'full_data_audit_summary.json',slim)
    table='| Model | Scope | Updates | NFE | Macro F1 | Macro IoU | Recall |\n|---|---|---:|---:|---:|---:|---:|\n'
    status=[]
    for objective in ('fm','bce'):
        for scope,dirname,step in (('overfit8_pre_f64',f'stage1_{objective}_overfit200',200),('pilot_f64_dev8',f'stage1_{objective}_pilot1000_f64',1000)):
            root=run/dirname;summary=read(root/'summary.json');metrics=read(root/'validation'/f'update_{step:06d}.json')
            log=root/'train_rank0.jsonl';last=json.loads(log.read_text().splitlines()[-1]) if log.exists() else None
            status.append({'model':objective,'scope':scope,'last_update':last['successful_updates'] if last else 0,'summary':summary})
            if metrics:
                for nfe,v in metrics['nfe_results'].items():
                    m=v['summary']['macro'];table+=f"| {objective} | {scope} | {step} | {nfe} | {m['f1']:.4f} | {m['iou']:.4f} | {m['recall']:.4f} |\n"
                shutil.copyfile(root/'validation'/f'update_{step:06d}.json',reports/f'{dirname}_metrics.json')
    atomic_json(reports/'experiment_progress.json',status)
    docs.joinpath('HS_OCCFLOW_384_STAGE1_DATA.md').write_text(f'''#384 Stage1 raw-map occupancy data

Main geometry is locked by explicit user selection:384x384x32,2x2x2 refinement,
fine768x768x64, open50m sphere and Z(-4,4.4). Geometry SHA:
`c07b659712f6260ed07c77941568748389826ff3523c3603484969cacb35dd9b`.
Parent size0.2604167x0.2604167x0.2625m; fine0.1302083x0.1302083x0.13125m.

Train17,870, development900, excluded temporal guard360. Frozen manifest bytes
are preserved; no seq08 is accessed. GT comes from `_full_candidate()` map pose,
local crop, height and viewpoint support before Poisson/fixed-point processing.
This is surface-hit occupancy, not verified solid occupancy or free space.

Current snapshot: **{data['voxelized_frames']} voxelized frames,
{data['committed_frames']} transactionally committed frames** out of18,770.
Full data audit ready: **{data['full_dataset_ready']}**. Progress is a snapshot,
not an assertion that the full train split is usable. Full Stage1 requires the
assembled train/development manifests and successful full audit.

Output: `{data['output']}`. CPU tmux: `{data['preprocessing_tmux']}`.
Eight low-priority workers process at most1,024 frames per sequence chunk.
Each validates raw-source checksums and independently verifies direct parent
rasterization equals fine-child OR. Labels are uint32 IDs plus uint8 masks.
Shards are atomically committed; the final dataset uses hardlinks to worker
shards, avoiding a second copy. Check unique inode size when measuring storage.
The prior sequential109-frame partial dataset remains intact.

An exact small integer lookup replaces structured-array viewpoint membership,
without changing target support. Four real candidate arrays, including1,322,038
rows of00/000000, match the original exactly in coordinates and order. Raw
candidate time fell from4.45s to1.48s on that frame, excluding map-index setup.
Initial slow-worker committed shards were preserved on restart; implementation
history and source hashes are recorded. See[lookup proof](hs_occ384_stage1_reports/viewpoint_lookup_audit.json).
'''.replace('#384','# 384'))
    docs.joinpath('HS_OCCFLOW_384_STAGE1_GPU_PREFLIGHT.md').write_text(f'''#384 Stage1 engineering preflight

CPU suite: {tests['passed']} passed,{tests['skipped']} CUDA test skipped in that
CPU invocation. The skipped CPU/CUDA geometry parity test was separately run
on admitted physicalGPU2:1 passed. No system environment was changed.
Environment: `/data-12/M2024-HWZ/FPSGen/env/voxdet_density_py39` for full model,
`env/pwm_ptv3_pointflow` for CPU/data tools. PyTorch2.0.1+cu117.

Encoder19,056,568 parameters. FM U-Net524,073; BCE U-Net503,465. Both reuse the
same VoxDet trunk and convolution widths8/16/32/64. BCE uses a learned constant
FiLM embedding; FM uses time embedding. No density/detection/regression neck head
is optimized. All parameters start randomly; constructor checkpoint reads are
forbidden by the construction audit. EMA initially equals the online model.

The prototype imported a missing E2E module and passed unsupported uint32 NumPy
arrays to Torch2.0. Both are fixed in independent Stage1 code. Tokenizer coarse
coordinates now use configured FP64 sizes, matching hierarchy discretization;
the old FP32-size helper remains untouched. The feature version is explicitly
`coarse_f64_v1` in new configurations. CPU/CUDA indices match exactly, with
feature absolute differences<=1.2e-7 on three real scenes.

Actual GPU2FM/GPU3BCE forwards and backwards passed at384 resolution:
condition ZYX16x192x192,8x96x96,4x48x48,128 channels each;
output[1,1,32,384,384]. Peak allocation~5GiB; sample-scene backward~1.05s after
warmup. These are **cohort maxima**, not the unprocessed population maximum.
Initial unscaled FP16 diagnostics underflowed; GradScaler65536 overflowed the
FM output layer initially. Scale16384 is registered; occasional recoverable
startup overflow occurred during warmup and remains in logs.

Final coordinate version completed32 successful DDP updates in33 attempts.
A true same-origin31-save/32-continuous vs restart31/32 test passes with
atol1e-5,rtol1e-3; maximum model difference1.2e-7. RNG, scaler, sampler,
optimizer steps and frame order match exactly. The earlier independent-run32-step
comparison had divergent BN buffers and remains archived asFAIL; it is not
re-labeled as a successful exact-resume result. The same-origin test isolates
checkpoint correctness from independent-run CUDA variability.

Original Student PIDs1294615/1294616 exited externally before GPU admission.
This task sent them no signals. Their last observed update was26,469; exit
reason was not established and they are not labeled complete. All GPU compute
for this work is restricted to physical2/3. CPU preprocessor interruptions
affected only this task's own verified processes and retained committed shards.
'''.replace('#384','# 384'))
    docs.joinpath('HS_OCCFLOW_384_STAGE1_PILOT.md').write_text('''#384 occupancy learning evidence

No density mapping/normalization, occupancy auxiliary, shortcut, teacher target,
Chamfer or mass loss is used. FM loss is masked velocity MSE only. BCE is plain
masked BCEWithLogits. Input XYZ and derived features/observed mask are the only
conditions. Seeds,8-frame training order, model widths and successful-update
budgets match between baselines. Development is the separate frozen8-frame
initial cohort; it does not replace full900-frame formal evaluation.

'''+table+'''
The200-update overfit runs used the pre-FP64-size helper and evaluated their
8 training frames with online weights. They show learning capacity, not held-out
quality. The fresh1K pilots use the final precision-audited geometry helper and
evaluate online weights on8 held-out frames, plus train-set diagnostics at1K.
Pilot parameters/optimizers are never reused by formal training. Short-pilot
EMA still lags; online and EMA evaluation choices are explicitly labeled.

BCE is currently substantially stronger in the200-step comparison. FM loss
reduction alone is not proof of good occupancy or an advantage over BCE.
No claim that FM wins is made. Formal gates require real learning/no collapse,
finite gradients, successful resume and complete audited full data.
'''.replace('#384','# 384'))
    formal=run/'stage1_occ384_fm_100k';first30=read(formal/'first30min.json')
    if not (formal/'train_rank0.jsonl').exists():
        docs.joinpath('HS_OCCFLOW_384_STAGE1_FORMAL_LAUNCH.md').write_text('''# Formal100K launch: NOT STARTED

Required before launch: complete audited17,870/900 data, full-data maximum scene
GPU check, matched overfit and1K pilots, frozen config/engineering gate. New
formal output is `experiments/hier_occ_flow/stage1_occ384_fm_100k`.
Planned tmux is `fpsgen_hs_occ384_stage1_fm100k_gpu23`. No formal process/PID or
checkpoint is claimed. Existing pilot and smoke checkpoints are not formal ones.
''')
    if not first30 or not first30.get('first30min_complete'):
        docs.joinpath('HS_OCCFLOW_384_STAGE1_FIRST30MIN.md').write_text('''# Formal first30-minute observation: NOT COMPLETED

Actual formal monitored duration:0 seconds. Formal training has not yet reached
this gate. CPU preparation, GPU smoke, overfit and pilot time do not count.
The monitoring script alone is not evidence. Actual live logs/GPU/process checks
must continue for at least30 minutes after formal successful updates begin.
''')
    print('Updated measured reports:',reports)

if __name__=='__main__':main()
