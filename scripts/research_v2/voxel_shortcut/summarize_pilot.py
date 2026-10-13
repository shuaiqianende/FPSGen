"""Generate small, auditable development-only SS-VSF pilot reports."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics
from common import save_json,sha_file


EXPERIMENTS={
    'Gaussian CFM':('eval_gaussian_cfm_final',64),
    'S2 CFM':('eval_structured_cfm_final',64),
    'S3 CFM':('eval_observation_cfm',64),
    'Residual':('eval_residual',64),
    'S2 Shortcut':('eval_structured_shortcut',128),
    'S3 Shortcut':('eval_observation_shortcut',128),
    'Shortcut continuation':('eval_shortcut_continuation',160),
    'Surface occupancy':('eval_surface_occupancy',160),
    'Surface multiscale':('eval_surface_multiscale',160),
    'Surface both':('eval_surface_both',160),
}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.experiment_root;args.output.mkdir(parents=True,exist_ok=True)
    curves={};table=[]
    for label,(folder,updates) in EXPERIMENTS.items():
        path=root/folder/'nfe_curve.json';d=json.loads(path.read_text())
        if d['status']!='COMPLETE' or d['frames']!=12 or d['split']!='development' or not d['flow_pilot_only']:
            raise ValueError('Expected complete12-frame development pilot only')
        curves[label]=d
        shutil.copyfile(path,args.output/f'{folder}.json')
        for n,mean in d['means'].items():
            if n=='0' and label!='Gaussian CFM':continue
            metrics=mean['density_metrics']
            table.append({'method':'Base' if n=='0' else label,'velocity_nfe':int(n),'correction_updates_total':0 if n=='0' else updates,
                'mass_tv':metrics['mass_tv'],'occupancy_iou':metrics['occupancy_iou'],'occupancy_f1':metrics['occupancy_f1'],
                'occupied_count_mae':metrics['occupied_count_mae'],'log_count_mae':metrics['log_count_mae'],
                'raw_mass_calibration_error':metrics['raw_mass_calibration_error'],
                'base_ms_warm':statistics.mean(r['base_ms'] for r in d['rows'] if r['requested_nfe']==int(n) and r['frame']!=d['rows'][0]['frame']),
                'flow_decode_ms_warm':statistics.mean(r['flow_and_decode_ms'] for r in d['rows'] if r['requested_nfe']==int(n) and r['frame']!=d['rows'][0]['frame'])})
    base_shas={d['base_sha256'] for d in curves.values()}
    frames=[[(r['frame']['sequence'],r['frame']['frame']) for r in d['rows'] if r['requested_nfe']==0] for d in curves.values()]
    if len(base_shas)!=1 or any(f!=frames[0] for f in frames):raise ValueError('Different base or development frames')
    with (args.output/'nfe_table.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=table[0].keys(),lineterminator='\n');writer.writeheader();writer.writerows(table)
    resources={}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'summary_rank0.json').exists():continue
        if not folder.name.startswith(('base_hurdle','cfm_','gaussian_cfm_pilot','structured_cfm_pilot','observation_cfm_pilot','residual_pilot','shortcut_','structured_shortcut','observation_shortcut','surface_')):continue
        summaries=[]
        for path in sorted(folder.glob('summary_rank*.json')):
            d=json.loads(path.read_text());log=folder/path.name.replace('summary_','train_').replace('.json','.jsonl')
            records=[json.loads(line) for line in log.read_text().splitlines()]
            d['median_step_seconds_excluding_first']=statistics.median(r['seconds'] for r in records[1:]) if len(records)>1 else None
            d['off_path_updates']=sum(r.get('off_path',False) for r in records)
            d['t0_h1_endpoint_updates']=sum(r.get('endpoint_t')==0. for r in records)
            d['bootstrap_half_steps_visited']=sorted({r['half_step'] for r in records if r.get('half_step') is not None})
            summaries.append(d)
        resources[folder.name]=summaries
    save_json(args.output/'resource_summary.json',resources)
    for name in ('inference_validation.json','background_mass_diagnostic/nfe_curve.json','bn_buffer_update_validation.json'):
        shutil.copyfile(root/name,args.output/Path(name).name.replace('nfe_curve','background_mass'))
    anchors={}
    for name in ('observation_cfm','observation_shortcut','residual','surface_both'):
        d=json.loads((root/f'anchor_{name}'/'nfe_curve.json').read_text())
        if d['frames']!=3 or d['status']!='COMPLETE':raise ValueError('Anchor diagnostic incomplete')
        anchors[name]={'means':d['means'],'source_rows':d['source_rows'],'point_count':180000,'frames':3,'config_sha256':d['config_sha256']}
        shutil.copyfile(root/f'anchor_{name}'/'nfe_curve.json',args.output/f'anchor_{name}.json')
    save_json(args.output/'anchor_summary.json',anchors)
    save_json(args.output/'config_sha256.json',{str(path):sha_file(path) for folder in (root/'configs',Path('configs/research_v2/voxel_shortcut')) for path in sorted(folder.glob('*.yaml'))})
    cpu=json.loads((root/'cpu_test_report.json').read_text())
    save_json(args.output/'final_gate.json',{'mechanics':'PASS','cpu_tests':cpu,'gt_free_inference_5_frames':'PASS',
        'structured_source_helps':'INCONCLUSIVE','shortcut_low_nfe_improves':'INCONCLUSIVE','surface_helps':'INCONCLUSIVE',
        'formal_training_ready':False,'quality_gate':'FAIL',
        'reason':'All correction pilots worsen density Mass-TV and180k anchor CD vs frozen base; small F1 gain is insufficient',
        'base_sha256':next(iter(base_shas)),'formal_long_training_started':False,'stage2_modified':False,
        'pilot_comparability':'Equal64-update CFM/residual pilots; Shortcut64 additional; surface/control32 additional; not fully converged matched-budget research evidence'})
    print(json.dumps({'rows':len(table),'methods':len(curves),'formal_training_ready':False}))


if __name__=='__main__':main()
