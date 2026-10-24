"""Read-only progress export; this report tool is outside the pinned training sources."""
import argparse
import csv
from datetime import datetime,timedelta
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from fpsgen.hier_occ_flow.stage2_runtime import source_sha


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def logs(path):
    # A concurrently appended, incomplete last line is excluded from the snapshot.
    rows=[json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n') and line.strip()]
    good={r['successful_updates']:r for r in rows if r.get('finite') and r.get('loss') is not None}
    return rows,good


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();before=source_sha();meta=read(a.experiment_root/'latest_queue.json')
    queue=Path(meta['output']);now=datetime.now().astimezone()
    a.output.mkdir(parents=True,exist_ok=False)
    artifacts=[];summary={'captured_at':now.isoformat(),'source_sha256':before,
        'training_seed':20261032,'batch_per_gpu':2,'budget_per_arm':20000,
        'status':'IN_PROGRESS_SNAPSHOT','runs':{},'sampling_seeds_are_not_training_repetitions':True}
    def retain(path,name):
        dest=a.output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        artifacts.append({'source':str(path.resolve()),'artifact':name,'sha256':sha(dest)})
    retain(a.experiment_root/'latest_queue.json','runtime/latest_queue.json')
    retain(queue/'contract.json','runtime/original_contract.json')
    for name in ('transfer.json','scheduler.json','run_transferred_queue.py'):
        retain(queue/'gpu2_transfer'/name,'runtime/'+name)
    smoke={}
    source={};loss_rows=[]
    for arm in ('position','attention'):
        folder=queue/(arm+'_20k');path=folder/'train_rank0.jsonl';rows,good=logs(path)
        source[arm]=good;last=rows[-1];recent=[good[k] for k in sorted(good)[-500:]]
        step_seconds=statistics.mean(r['seconds'] for r in recent)
        remaining=max(0,10000-last['successful_updates'])*step_seconds
        launch=read(folder/'launch.json');cfg=read(folder/'config.json')
        if cfg['training']['seed']!=20261032 or cfg['training']['successful_frame_exposures']!=20000:
            raise ValueError('Training protocol differs from the registered pair')
        if launch['identity']['source_sha256']!=before:raise ValueError('Training source identity changed')
        age=time.time()-path.stat().st_mtime
        status=('COMPLETE' if last['successful_updates']==10000 else 'RUNNING' if age<300 else 'NEEDS_LIVE_PROCESS_CHECK')
        primary=folder/'evaluation/primary900_nfe4.json'
        item={'status':status,'successful_updates':last['successful_updates'],
            'successful_frame_exposures':last['successful_frame_exposures'],'attempts':last['attempts'],
            'amp_skips':last['amp_skips'],'physical_gpu':meta['training_gpu_map'][arm],
            'recent500_mean_update_seconds':step_seconds,'remaining_training_hours_estimate':remaining/3600,
            'estimated_training_finish':(now+timedelta(seconds=remaining)).isoformat(),
            'estimate_excludes_evaluation':True,'log_age_seconds':age,
            'last_log_loss':last['loss'],'quality_status':'COMPLETE' if primary.exists() else 'NOT TESTED',
            'checkpoint_identity':launch['identity'],'initialization':launch.get('initialization_audit')}
        if primary.exists():
            result=read(primary)['nfe_results']['4'];item['primary900']={k:result[k] for k in ('summary','continuous_macro')}
        immutable=folder/'checkpoints/step002500.pt'
        if immutable.exists():item['saved_2500_update_checkpoint']={'path':str(immutable),'sha256':sha(immutable),'exposures':5000}
        summary['runs'][arm]=item
        for name in ('config.json','launch.json','health_000100.json','health_000500.json'):
            retain(folder/name,arm+'/'+name)
        smoke[arm]=read(queue/('engineering_'+arm)/'summary.json')
    (a.output/'engineering_smokes.json').write_text(json.dumps(smoke,indent=2)+'\n')
    _,baseline=logs(a.baseline_root/'train_rank0.jsonl');source['b0_reference']=baseline
    limit=max(max(source[k]) for k in ('position','attention'))
    for arm,values in source.items():
        end=min(max(values),limit)
        for start in range(1,end+1,200):
            stop=min(start+199,end);window=[values[k] for k in range(start,stop+1)]
            loss_rows.append({'arm':arm,'first_update':start,'last_update':stop,'updates':len(window),
                'window_status':'COMPLETE_200' if len(window)==200 else 'PARTIAL',
                'mean_fm_loss':statistics.mean(x['loss'] for x in window),
                'mean_update_seconds':statistics.mean(x['seconds'] for x in window)})
    write_csv(a.output/'loss_200updates.csv',loss_rows)
    def batch_ids(row):return [(x['sequence'],x['frame'],x['sampler_index']) for x in row['batch_records']]
    alignment={}
    for left,right in (('position','attention'),('b0_reference','position'),('b0_reference','attention')):
        count=min(max(source[left]),max(source[right]));diff=[k for k in range(1,count+1) if batch_ids(source[left][k])!=batch_ids(source[right][k])]
        alignment[left+'_vs_'+right]={'successful_updates_compared':count,'mismatched_batch_count':len(diff),'first_mismatch':diff[0] if diff else None}
    summary['step_window_frame_alignment']=alignment
    summary['after_training_automatic']=['900-frame primary NFE4 native+continuous metrics','64-frame NFE1/2/4/8 and condition diagnostics']
    summary['encoder_followups']={'status':'REGISTERED_NOT_ENGINEERING_READY','automatic_training_queued':False,
        'candidates':['encoder-only adaptation with frozen Flow','matched Encoder+Flow adaptation','conditional scratch architecture retraining']}
    summary['artifact_manifest']=artifacts
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Local state pilot: current progress and next experiments','',f"Captured: {now.isoformat()}",'',
        '**Training is still in progress. Final L0/L1 quality is NOT TESTED.** This dated snapshot supersedes the earlier GPU1 assignment; the actual pair is GPU2 position and GPU3 position+attention.','',
        '| Arm | GPU | Successful updates | Successful exposures | Recent500 seconds/update | Remaining training hours, estimate |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,v in summary['runs'].items():
        lines.append(f"| {arm} | {v['physical_gpu']} | {v['successful_updates']} /10000 | {v['successful_frame_exposures']} /20000 | {v['recent500_mean_update_seconds']:.3f} | {v['remaining_training_hours_estimate']:.2f} |")
    lines+=['','Single training seed20261032, real packedBatch2, same original raw-GT source and Frozen Stage1 NFE2 support. AMP skipped attempts are excluded from exposure counts. ETA follows the last500 successful updates and excludes evaluation; changing scene costs can move it.','',
        '## Loss in200-successful-update windows','',
        '| Updates | B0 original reference | Position | Position+Attention |','|---|---:|---:|---:|']
    for end in range(200,limit+1,200):
        values=[]
        for arm in ('b0_reference','position','attention'):
            values.append(f"{statistics.mean(source[arm][k]['loss'] for k in range(end-199,end+1)):.6f}" if end<=max(source[arm]) else 'NOT COMPLETE')
        lines.append(f"| {end-199}–{end} | {' | '.join(values)} |")
    lines+=['','Full and partial windows are distinguished in `loss_200updates.csv`. Batch-order checks are saved in summary.json: different AMP skip histories can misalign successful-step windows. Training loss does not establish a geometric quality gain.','',
        '## Automatic continuation and candidates','',
        'The transferred supervisor adopts the original running GPU3 process and launches the formerly waiting position arm on idleGPU2. Source hash, configurations, seeds, optimizer horizon and all existing artifacts are preserved. The transfer audit and exact server-local supervisor script are retained under runtime/.','',
        'After each20K run, the active queue automatically evaluates900 Development frames (NFE4, native Fine IoU/F1, exact KD-tree CD, corrected physical IoU and JSD), followed by64-frame NFE/conditioning diagnostics. Compare against completed B0; report matched Parent support, worst-frame deltas and actual runtime before selecting a candidate.','',
        'Next candidates are Encoder-only adaptation with frozen Stage1 Flow, matched Encoder+Flow adaptation, and hypothesis-driven scratch retraining. Each is limited to20K successful exposures, Batch2 and the same training seed. **Their engineering runner/checks and automatic training continuation are not yet implemented. They are registered candidates, not active or auto-queued training.** Existing verified scratch20K controls can be reused when identities match. No automatic100K or additional training seeds.','',
        'Completed C1/B0/M1, Stage1 quality and three-round Encoder timings remain in `../desescene_completed_20261024/`. The current snapshot adds progress,200-step losses, GPU2 scheduling and follow-up readiness. Checkpoints and prediction clouds remain server-local.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    if source_sha()!=before:raise RuntimeError('Report generation changed pinned training sources')
    print(json.dumps({'output':str(a.output),'captured_at':now.isoformat(),'runs':{k:{f:v[f] for f in ('status','successful_updates','successful_frame_exposures')} for k,v in summary['runs'].items()},'source_hash_unchanged':True}))


if __name__=='__main__':main()
