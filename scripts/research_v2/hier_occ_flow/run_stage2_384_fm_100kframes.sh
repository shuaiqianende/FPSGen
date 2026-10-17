#!/usr/bin/env bash
# User-authorized100K successful scene exposures; unchanged engineering gates.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
task_main=$(dirname "$(git rev-parse --git-common-dir)")
task_python=${HS_STAGE2_PYTHON:-$task_main/env/voxdet_density_py39/bin/python}
export CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
task_cfg=configs/research_v2/hier_occ_flow/stage2_384_fm_gb4_100kframes.yaml
task_output=experiments/hier_occ_flow/stage2_384_fm_gb4_100kframes
task_log=experiments/hier_occ_flow/stage2_384_preflight/fm_gb4_100kframes.log
exec > >(tee -a "$task_log") 2>&1
task_phase=full_split_first20
trap 'task_exit=$?; printf "FULL100KFRAMES phase=%s exit=%s. Session remains open; inspect summary.json and checkpoints.\n" "$task_phase" "$task_exit"' EXIT
echo 'User-authorized full17870 training split, FM only, frozen Stage1, Oracle support.'
echo 'Total budget100000 successful frame exposures includes1808 earlier frames. Remaining98192 frames =24548 GB4 updates.'
echo 'Prior Oracle quality gate FAILED and remains recorded. Quality is monitored; numerical, memory and synchronization gates remain enforced.'
if [[ ! -f "$task_output/checkpoints/resume_latest.pt" ]]; then
  bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_cfg" \
    --output "$task_output" --transition-from experiments/hier_occ_flow/stage2_384_preflight/fm_before_full100kframes.pt \
    --max-updates 592
  "$task_python" - "$task_output" <<'PY'
import json,sys,torch,time
from pathlib import Path
from fpsgen.hier_occ_flow.storage import atomic_json
root=Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text())
a=torch.load(root/'checkpoints/step000572_transition.pt',map_location='cpu')
b=torch.load(root/'checkpoints/resume_latest.pt',map_location='cpu')
assert summary['status']=='COMPLETE_REQUESTED_BUDGET' and b['successful_updates']==592
assert b['sampler']['successful_frame_exposures']==1888 and b['sampler']['cohort_indices'] is None
assert b['parameter_sync_pass'] and b['rank_hashes'][0]==b['rank_hashes'][1]
assert b['amp_skips']==a['amp_skips']==0 and b['frame_budget']['target_successful_updates']==25120
assert b['frame_budget']['target_successful_frame_exposures']==100000
changed=sum(not torch.equal(v,b['model'][k]) for k,v in a['model'].items());assert changed>0
for rank in (0,1):
 rows=[json.loads(line) for line in (root/f'train_rank{rank}.jsonl').read_text().splitlines()]
 assert len(rows)==20 and all(r['finite'] and r['gradient_norm']>0 and r['global_batch']==4 for r in rows)
 assert len({r['sampler_index'] for row in rows for r in row['batch_records']})>12
report={'status':'PASS_FULL_SPLIT20_FRAME_BUDGET','recorded_at_epoch':time.time(),'successful_updates':592,
        'successful_frame_exposures':1888,'new_full_split_frames':80,'global_batch':4,
        'changed_model_tensors':changed,'parameter_sync_pass':True,'amp_skips':0,
        'formal_training_complete':False,'quality_gate_pass_claimed':False}
atomic_json(root/'full_split20_audit.json',report);print(report,flush=True)
PY
fi
task_phase=resume_full100kframes
bash scripts/research_v2/hier_occ_flow/launch_stage2_384.sh --config "$task_cfg" \
  --output "$task_output" --resume "$task_output/checkpoints/resume_latest.pt"
task_phase=budget_complete_or_safe_pause
echo 'Inspect summary for actual completion versus a numerical/signal safe pause. No Stage2 PointFlow/Teacher/BCE job queued.'
