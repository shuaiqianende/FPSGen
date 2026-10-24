"""Export compact completed research evidence; never copy weights or point clouds."""
from common import ROOT
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

from fpsgen.hier_occ_flow.storage import file_sha


def read(path):return json.loads(path.read_text())


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--matched-root',type=Path,required=True)
    p.add_argument('--encoder-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=[];summary={'stage2':{},'encoder':{}}
    def retain(source,destination):
        destination=a.output/destination;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        manifest.append({'source':str(source.resolve()),'artifact':str(destination.relative_to(a.output)),
                         'sha256':file_sha(source)})
    metric_rows=[];loss_rows=[];encoder_rows=[]
    for arm in ('b0','m1'):
        folder=a.matched_root/(arm+'_20k')
        train=read(folder/'summary.json')
        if train['successful_updates']!=10000 or train['successful_frame_exposures']!=20000:
            raise ValueError('Cannot export an incomplete training run as complete')
        evaluation=read(folder/'evaluation/primary900_nfe4.json');primary=evaluation['nfe_results']['4']
        if len(primary['rows'])!=900:raise ValueError('Require all900 primary frames')
        summary['stage2'][arm]={'native':primary['summary'],'continuous':primary['continuous_macro'],
            'checkpoint_sha256':evaluation['checkpoint_sha256'],'training_identity':train['identity'],
            'sampling':'EMA, NFE4, native voxel centers; fixed predicted NFE2 Stage1 support',
            'successful_frame_exposures':20000,'successful_updates':10000}
        for name in ('config.json','summary.json','launch.json'):
            retain(folder/name,arm+'/'+name)
        for name in ('summary.json','contract.json','per_frame.csv','sensitivity.csv'):
            retain(folder/'evaluation/mechanisms64'/name,arm+'/mechanisms64/'+name)
        retain(folder/'evaluation/primary900_nfe4.protocol.json',arm+'/evaluation_protocol.json')
        for row in primary['rows']:
            metric_rows.append({'arm':arm,'sequence':row['sequence'],'frame':row['frame'],
                'input_sha256':row['input_sha256'],'parent_ids_sha256':row['parent_ids_sha256'],
                'parent_count':row['candidate']['active_parent_count'],
                'fine_gt_ceiling':row['candidate']['fine_occupancy_recall_ceiling'],
                **{'native_'+k:row['fine'][k] for k in ('iou','f1','precision','recall','tp','fp','fn')},
                'support_only_iou':row['support_only_fine']['iou'],
                **{k:row['continuous'][k] for k in ('cd_m','iou05','iou02','iou01','jsd_3d','jsd_bev')},
                'predicted_points':row['continuous']['predicted_points'],
                'raw_gt_xyz_sha256':row['continuous']['raw_gt_xyz_sha256']})
        logs=[json.loads(x) for x in (folder/'train_rank0.jsonl').read_text().splitlines() if x.strip()]
        successes={r['successful_updates']:r for r in logs if r['finite']}
        for end in range(500,10001,500):
            window=[successes[k] for k in range(end-499,end+1)]
            loss_rows.append({'arm':arm,'first_update':end-499,'last_update':end,'successful_exposures':end*2,
                'mean_loss':statistics.mean(r['loss'] for r in window),
                'mean_update_seconds':statistics.mean(r['seconds'] for r in window),
                'cumulative_amp_skips':window[-1]['amp_skips']})
    retain(a.matched_root/'matched900_comparison.json','matched900_comparison.json')
    for path in sorted(a.encoder_root.glob('*_round*.json')):
        value=read(path)
        if value.get('status')!='COMPLETE':raise ValueError('Incomplete Encoder timing')
        name=path.name.split('_round')[0]
        summary['encoder'].setdefault(name,[]).append(value['summary'])
        retain(path,'encoder/'+path.name)
        for row in value['per_frame']:
            encoder_rows.append({'variant':name,'paired_round':int(path.stem.rsplit('round',1)[1]),
                'sequence':row['sequence'],'frame':row['frame'],
                'input_sha256':row['input_sha256'],'encoder_cuda_ms':row['network_cuda_event_ms'],
                'encoder_wall_including_h2d_ms':row['generation_encoder_wall_ms'],
                'peak_allocated_gib':value['peak_allocated_gib']})
    for name,rounds in summary['encoder'].items():
        if len(rounds)!=3:raise ValueError('Require three paired rounds for '+name)
    retain(a.encoder_root/'summary.json','encoder/summary.json')
    write_csv(a.output/'per_frame_quality900.csv',metric_rows)
    write_csv(a.output/'loss_500updates.csv',loss_rows)
    write_csv(a.output/'encoder_per_frame_timing.csv',encoder_rows)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Completed DeSeScene evidence, 2026-10-24','','B0/M1 each completed20K additional matched adaptation; same raw-GT source, seed20261032, batch2, predicted Stage1 NFE2 support. C1 scratch/NFE4 metrics are a separate cohort.','',
           '| Stage2 | Native Fine IoU | CD m | IoU0.1m | JSD3D |','|---|---:|---:|---:|---:|']
    for name,r in summary['stage2'].items():
        c=r['continuous'];lines.append(f"| {name} | {r['native']['macro']['iou']:.8f} | {c['cd_m']:.8f} | {c['iou01']:.8f} | {c['jsd_3d']:.8f} |")
    lines+=['','M1 has no demonstrated benefit and fails the registered+1pp native-IoU gate. Its paired95% interval includes zero. Preserve this negative result; do not expand M1 training.','',
        '| Trained Encoder | CUDA ms, three-round mean | Encoder wall including H2D ms |','|---|---:|---:|']
    for name,rounds in sorted(summary['encoder'].items()):
        lines.append(f"| {name} | {statistics.mean(r['network_cuda_event']['mean_ms'] for r in rounds):.3f} | {statistics.mean(r['encoder_wall_including_h2d']['mean_ms'] for r in rounds):.3f} |")
    lines+=['','All Encoder variants use physicalGPU3, explicitFP16, the same5+20 Development inputs and three alternating rounds. These are complete Encoder timings, not full-generation latency. E1 has the highest observed20K ParentIoU but little timing benefit; E2 has the largest PyTorch timing benefit. TensorRT, fixed-K and downstream checks remain necessary before deployment selection.','',
        'The original Encoder timing queue failed while loading inline geometry; the minimal loader fix and regression test are included in the same research branch. Original failed artifacts remain untouched. No weights, point clouds or compiled caches are exported. The sensitivity CSV and NFE tables are diagnostics, not final architecture acceptance.']
    (a.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(a.output),'quality_rows':len(metric_rows),'timing_rows':len(encoder_rows)}))


if __name__=='__main__':main()
