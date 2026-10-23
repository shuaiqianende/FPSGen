"""Register a narrow research-pilot gate from real packed-batch GPU smoke evidence."""
import argparse
import json
from pathlib import Path

from fpsgen.hier_occ_flow.storage import atomic_json, file_sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--base-gate', type=Path, required=True)
    p.add_argument('--smoke-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    gate = json.loads(a.base_gate.read_text())
    summary_path = a.smoke_output / 'summary.json'
    launch_path = a.smoke_output / 'launch.json'
    cfg_path = a.smoke_output / 'config.json'
    log_path = a.smoke_output / 'train_rank0.jsonl'
    summary, launch, cfg = (json.loads(x.read_text()) for x in (summary_path, launch_path, cfg_path))
    if not gate.get('packed_batch2_gpu') or gate.get('status') not in (
            'PACKED_GB4_DDP_SMOKE_READY', 'PACKED_GB4_FM_PILOT_READY', 'PACKED_GB4_OVERFIT_READY',
            'RESEARCH_PILOT_ENGINEERING_READY'):
        raise ValueError('Base gate lacks verified two-scene GPU evidence')
    protocol=cfg.get('research_protocol')
    child_smoke=(protocol in ('desescene_child_bridge_geometry_v1','desescene_child_bridge_position_control_v1')
        and cfg.get('model', {}).get('child_bridge') == 'child_conditioned'
        and cfg.get('model', {}).get('child_input') in ('geometry_stats', 'position_only'))
    static_objective={'desescene_static_context_fm_v1':'fm','desescene_static_context_bce_v1':'bce'}.get(protocol)
    static_smoke=(static_objective is not None and cfg.get('model', {}).get('context_strategy')=='static'
        and cfg.get('model', {}).get('objective')==static_objective)
    if (summary.get('status') != 'COMPLETE_REQUESTED_BUDGET'
        or summary.get('successful_updates') != 32 or summary.get('successful_frame_exposures') != 64
        or launch.get('batch_per_gpu') != 2 or launch.get('global_batch') != 2
        or launch.get('physical_gpus') not in ([2], [3])
        or not (child_smoke or static_smoke)):
        raise ValueError('Smoke output is not a completed B2 registered DeSeScene research run')
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if len(rows) != 32 or any(not r.get('finite') or r.get('loss') is None for r in rows):
        raise ValueError('Smoke training contains missing or non-finite updates')
    gate = dict(gate)
    gate['research_model_smoke'] = True
    gate['research_model_smoke_protocol'] = 'desescene_child_bridge_b2_64exposure'
    gate.setdefault('evidence', []).append({
        'path': str(a.smoke_output.resolve()),
        'sha256_summary': file_sha(summary_path),
        'sha256_log': file_sha(log_path),
        'scope': cfg['training']['scope'],
        'physical_gpus': launch['physical_gpus'],
        'protocol': protocol,
        'status': 'PASS_DESESCENE_B2_SMOKE',
    })
    gate['status'] = 'RESEARCH_PILOT_ENGINEERING_READY'
    atomic_json(a.output, gate)
    print(json.dumps({'status': gate['status'], 'output': str(a.output),
                      'smoke_updates': len(rows), 'physical_gpus': launch['physical_gpus']}, indent=2))


if __name__ == '__main__':
    main()
