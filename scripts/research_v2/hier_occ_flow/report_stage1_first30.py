"""Summarize actual formal observations; refuses incomplete duration evidence."""
from common import ROOT
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.storage import atomic_json, file_sha
from monitor_stage1_occ import read_rows, snapshot


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--monitor-report',type=Path)
    a=p.parse_args();out=a.output
    monitor_path=a.monitor_report or out/'first30min.json'
    monitor=json.loads(monitor_path.read_text())
    checks=read_rows(out/'agent_checks.jsonl')
    # A long gap between interactions is not personal observation. Require the
    # latest uninterrupted segment; preserve all historical checks separately.
    for i in range(len(checks)-1,0,-1):
        if checks[i]['epoch_time']-checks[i-1]['epoch_time']>120:
            checks=checks[i:];break
    if not monitor.get('first30min_complete') or monitor['actual_observed_seconds']<1800:
        raise ValueError('Real formal 30-minute monitoring is not complete')
    if len(checks)<20 or checks[-1]['epoch_time']-checks[0]['epoch_time']<1800:
        raise ValueError('Need separate live agent checks spanning at least30 minutes')
    current=snapshot(out)
    if len(current['processes'])<current['expected_world_size'] or current['latest_update_age_seconds']>180:
        raise ValueError('Formal training is not currently healthy')
    start=monitor['events'][0]['epoch_time']
    observed=[r for r in monitor['events'] if r.get('actual_training_observed')]
    start=observed[0]['epoch_time'];end=observed[-1]['epoch_time']
    rows=read_rows(out/'train_rank0.jsonl')
    all_success=[r for r in rows if r.get('finite') and r['epoch_time']<=end]
    success=[r for r in all_success if r['epoch_time']>=start]
    last=success[-1];recent=success[-100:]
    audits=[r for r in success if r.get('gradient_details')]
    initial_path=out/'checkpoints/step000000.pt'
    if not initial_path.exists():
        cfg=json.loads((out/'config.json').read_text())
        initial_path=Path(cfg['batch_transition']['source_config']).parent/'checkpoints/step000000.pt'
    initial=torch.load(initial_path,map_location='cpu')
    latest_path=out/'checkpoints/resume_latest.pt'
    # Hold one inode across hashing/readback while the trainer atomically
    # replaces resume_latest.pt; do not pair a previous payload with a newSHA.
    with latest_path.open('rb') as stream:
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        checkpoint_sha=digest.hexdigest();stream.seek(0)
        latest=torch.load(stream,map_location='cpu')
    changes={}
    for name,prefix in [('encoder','encoder.'),('flow','flow.'),('velocity_head','flow.output.'),('time_embedding','flow.embedding.')]:
        keys=[k for k,v in initial['model'].items() if k.startswith(prefix) and v.is_floating_point()]
        squared=sum(float((latest['model'][k].double()-initial['model'][k].double()).square().sum()) for k in keys)
        changes[name]={'state_difference_l2':math.sqrt(squared),'changed_tensors':sum(not torch.equal(initial['model'][k],latest['model'][k]) for k in keys),'floating_tensors':len(keys)}
    changes['ema']={'changed_tensors':sum(not torch.equal(v,latest['ema'][k]) for k,v in initial['ema'].items())}
    checkpoint={'path':str(latest_path.resolve()),'sha256':checkpoint_sha,'successful_updates':latest['successful_updates'],
        'parameter_sync_pass':latest['parameter_sync_pass'],'rank_model_hashes':latest['rank_model_hashes'],
        'rank_rng_states':len(latest['rank_rng']),'optimizer_entries':len(latest['optimizer']['state']),
        'scheduler_last_epoch':latest['scheduler'].get('last_epoch'),'sampler':latest['sampler'],
        'required_fields_present':all(k in latest for k in ('model','ema','optimizer','scheduler','scaler','rank_rng','sampler','identity'))}
    peak={}
    for rank in range(2):
        rr=read_rows(out/f'train_rank{rank}.jsonl')
        peak[str(rank)]={k:max(r[k] for r in rr) for k in ('peak_allocated_bytes','peak_reserved_bytes')}
    validation=[]
    for path in sorted((out/'validation').glob('update_*.json')):
        v=json.loads(path.read_text())
        metrics={}
        for k,x in v['nfe_results'].items():
            metrics[k]={**x['summary']['macro'],
                **{name:float(np.mean([r[name] for r in x['rows']])) for name in ('predicted_occupied_ratio','gt_occupied_ratio')},
                'thresholds':{threshold:{name:float(np.mean([r['thresholds'][threshold][name] for r in x['rows']]))
                    for name in ('precision','recall','f1','iou','active_parents','fine_recall_ceiling')}
                    for threshold in x['rows'][0]['thresholds']}}
        validation.append({'file':str(path.resolve()),'results':metrics})
    launch=json.loads((out/'launch_parameters.json').read_text())
    result={'formal_training_started':True,'process_alive':True,'still_running':True,
        'monitor_report':str(monitor_path.resolve()),'interactive_observation':'latest continuous segment; gaps >120 seconds excluded',
        'monitor_status':monitor['status'],'actual_observed_seconds':monitor['actual_observed_seconds'],
        'agent_checks':len(checks),'agent_check_span_seconds':checks[-1]['epoch_time']-checks[0]['epoch_time'],
        'starting_update':observed[0]['successful_updates'],'ending_update':last['successful_updates'],
        'current_successful_updates':current['successful_updates'],'attempts':current['attempts'],
        'initial_formal_loss':all_success[0]['loss'],'initial_loss':success[0]['loss'],'latest_loss':last['loss'],
        'last100_mean_loss':float(np.mean([r['loss'] for r in recent])),
        'last100_p50_loss':float(np.percentile([r['loss'] for r in recent],50)),
        'last100_p95_loss':float(np.percentile([r['loss'] for r in recent],95)),
        'loss_ema_alpha005':current['loss_ema_alpha005'],
        'latest_global_gradient_norm':last['global_gradient_before_clip'],
        'latest_module_gradients':audits[-1]['gradient_details'],
        'gradient_clipping_events':last['gradient_clipping_events'],
        'amp_overflow_count':last['amp_overflow_count'],'skipped_attempts':last['skipped_attempts'],
        'nonfinite_loss_incidents':sum(r.get('loss') is None for r in rows),
        'nonfinite_gradient_attempts':sum(not r.get('finite') for r in rows),
        'amp_scale':last['amp_scale'],
        'mean_train_step_seconds':float(np.mean([r['seconds'] for r in success])),
        'last100_data_seconds':float(np.mean([r['data_seconds'] for r in recent])),
        'wallclock_successful_updates_per_second':(last['successful_updates']-success[0]['successful_updates'])/(last['epoch_time']-success[0]['epoch_time']),
        'peak_memory_by_local_rank':peak,'checkpoint':checkpoint,'parameter_changes_since_step0':changes,
        'launch':launch,'current_gpu':current['gpu'],'current_processes':current['processes'],
        'initial_checkpoint_origin':str(initial_path.resolve()),
        'disk_free_bytes':current['disk_free_bytes'],'validation':validation}
    reports=ROOT/'docs/research_v2/hs_occ384_stage1_reports';reports.mkdir(exist_ok=True)
    atomic_json(reports/'formal_first30_summary.json',result)
    fields=['epoch_time','successful_updates','attempts','last100_loss_mean','loss_ema_alpha005','mean_step_seconds','latest_update_age_seconds','disk_free_bytes','gpu']
    with (reports/'formal_monitor_timeline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in observed:writer.writerow({k:r.get(k) for k in fields})
    text=f'''# Stage1 384 Occupancy: measured30-minute observation

Formal training remains **RUNNING**. Observed {monitor['actual_observed_seconds']:.1f}s after actual successful updates began; separate interactive agent observations: {len(checks)}, spanning {result['agent_check_span_seconds']:.1f}s. Preparation/Pilot time is excluded.

Observation source: `{monitor_path.resolve()}`. The original `first30min.json` is preserved independently. When this is a follow-up observation, it is not described as personal observation of the initial30 minutes. Long gaps between agent interactions are excluded rather than counted toward the continuous observation gate.

Updates observed: {result['starting_update']} → {result['ending_update']}; current snapshot: {current['successful_updates']} /100,000. Attempts: {current['attempts']}. This is an ongoing100K budget, not completed training.

Loss {result['initial_loss']:.6f} → {result['latest_loss']:.6f}; last100 mean {result['last100_mean_loss']:.6f}, P50 {result['last100_p50_loss']:.6f}, P95 {result['last100_p95_loss']:.6f}. Loss uses global mean over both ranks; random frame/time variation remains.

Gradient before clipping {result['latest_global_gradient_norm']:.6f}; clipping events {result['gradient_clipping_events']}; AMP overflows {result['amp_overflow_count']}, skipped attempts {result['skipped_attempts']}, nonfinite loss incidents {result['nonfinite_loss_incidents']}. Latest AMP scale {result['amp_scale']}.

Mean compute/update time {result['mean_train_step_seconds']:.3f}s; last100 host data preparation {result['last100_data_seconds']:.4f}s. Wall-clock speed including checkpoint/validation pauses {result['wallclock_successful_updates_per_second']:.3f} successful updates/s. No batch size or architecture change was made during formal training.

Encoder, Flow, output head and time embedding all have measured gradients and changed state tensors. Exact module gradient statistics, parameter differences from step0, rank peak allocated/reserved memory, GPU minute snapshots, checkpointSHA and development metrics are in [measured JSON](hs_occ384_stage1_reports/formal_first30_summary.json) and [timeline](hs_occ384_stage1_reports/formal_monitor_timeline.csv).

Checkpoint readback passed with model/EMA/optimizer/scheduler/scaler/rankRNG/sampler/identity fields; rank hashes synchronized: {checkpoint['parameter_sync_pass']}. Same-origin resume was tested before formal launch; the running formal process was not interrupted merely to repeat that test.

Physical GPUs2/3 only. tmux `fpsgen_hs_occ384_stage1_fm100k_gpu23`; launch commit `{launch['git_commit']}`. Latest checkpoint `{checkpoint['path']}` (snapshot update {checkpoint['successful_updates']}). Output `{out.resolve()}`.

Primary validation uses online weights. EMA0.999 is retained and evaluated separately; its cold-start lag is not conflated with online performance. Fixed development thresholds0.25/0.5/0.75 are diagnostics; no seq08 tuning. Stage2 is not trained. Pilot BCE is slightly stronger than FM and no FM superiority is claimed.
'''
    (ROOT/'docs/research_v2/HS_OCCFLOW_384_STAGE1_FIRST30MIN.md').write_text(text)
    print(json.dumps({k:result[k] for k in ('still_running','actual_observed_seconds','current_successful_updates','latest_loss','last100_mean_loss','amp_overflow_count')},indent=2))


if __name__=='__main__':main()
