"""Development-only support calibration and frozen-source/noise diagnostics."""

import copy
import json
from pathlib import Path
import statistics
import time
import torch
import yaml
from common import parser,read_config,frames_for,select_spaced,runtime,save_json,load_base
from fpsgen.voxel_shortcut.dataset import ShortcutDataset
from fpsgen.voxel_shortcut.source_builder import build_source
from fpsgen.voxel_shortcut.count_mapping import CountMapping
from fpsgen.voxel_shortcut.evaluator import field_metrics,endpoint_density


def main():
    p=parser(__doc__);p.add_argument('--calibrate-support',action='store_true');args=p.parse_args()
    cfg,spec,sha=read_config(args.config);device,rank,world=runtime()
    if world!=1:raise ValueError('Calibration is single-process')
    base,base_sha,checkpoint=load_base(cfg,spec,device)
    dataset=ShortcutDataset(select_spaced(frames_for(cfg,'development'),args.max_frames),spec,**cfg['data'].get('gt_validation',{}))
    # Keep the geometry from config for all targets and masks.
    dataset.geometry=spec
    mapping=CountMapping(**cfg['mapping']);valid=spec.field_mask(device=device)
    candidates=cfg['source_calibration']['support_threshold_candidates']
    rows=[];noise_rows=[];raw_base_rows=[]
    with torch.inference_mode():
        for i in range(len(dataset)):
            sample=dataset[i];gt=sample['gt_counts'][None].to(device);observed=sample['observed'][None].to(device)
            with torch.autocast('cuda',dtype=torch.float16,enabled=cfg['runtime']['precision']=='fp16'):
                output=base(sample['input_xyz'].to(device))
            raw_base_rows.append({'frame':sample['frame'],'gt_validation':sample['gt_validation'],**field_metrics(output['density'],gt,spec,observed=observed,threshold=cfg['evaluation']['occupancy_threshold'])})
            for threshold in candidates:
                source_config={**cfg['source'],'mode':'deterministic','support_threshold':threshold}
                x,source=build_source(output['density'],output['occupancy_probability'],observed,valid,mapping,
                                     generator=torch.Generator(device=device).manual_seed(cfg['seed']+i),**source_config)
                metrics=field_metrics(source['base_density'],gt,spec,observed=observed,threshold=cfg['evaluation']['occupancy_threshold'])
                rows.append({'frame':sample['frame'],'support_threshold':threshold,**metrics})
            for mode in ('gaussian','deterministic','fixed_gaussian','observed_unknown_gaussian'):
                sigmas=cfg['source_calibration']['noise_unobserved_candidates'] if mode in ('fixed_gaussian','observed_unknown_gaussian') else [None]
                for sigma in sigmas:
                    options={**cfg['source'],'mode':mode}
                    if sigma is not None:options.update(sigma_fixed=sigma,sigma_unobserved=sigma)
                    x,_=build_source(output['density'],output['occupancy_probability'],observed,valid,mapping,
                        generator=torch.Generator(device=device).manual_seed(cfg['seed']+i),**options)
                    density,raw,diag=endpoint_density(x,mapping,valid)
                    noise_rows.append({'frame':sample['frame'],'mode':mode,'sigma':sigma,
                        'mapped_source_target_mse':float(((x-mapping.encode(gt)).square()*valid).sum()/valid.sum()),
                        **field_metrics(density,gt,spec,observed=observed,threshold=cfg['evaluation']['occupancy_threshold']),**diag})
            print(f'development{i+1}/{len(dataset)}',flush=True)
    grouped={str(v):{key:statistics.mean(r[key] for r in rows if r['support_threshold']==v)
        for key in ('occupancy_iou','occupancy_f1','mass_tv','mass_calibration_error')} for v in candidates}
    best=min(candidates,key=lambda v:grouped[str(v)]['mass_tv'])
    save_json(args.output/'base_development_evaluation.json',{'config_sha256':sha,'base_sha256':base_sha,
        'frames':len(dataset),'support_candidates':grouped,'selected_support_threshold':best,
        'selection_criterion':'minimum gated normalized density Mass-TV on development only','rows':rows,'raw_base_rows':raw_base_rows,'noise_diagnostics':noise_rows,
        'pilot_only':checkpoint['pilot_only'],'formal_quality_claim':False})
    if args.calibrate_support:
        calibrated=copy.deepcopy(cfg);calibrated.pop('voxel');calibrated['source']['support_threshold']=best
        calibrated['source_calibration']['frozen']=True
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'calibrated.yaml').write_text(yaml.safe_dump(calibrated,sort_keys=False))
        (args.output/cfg['geometry']).write_text(yaml.safe_dump({'voxel':spec.to_dict()},sort_keys=False))
    print(json.dumps(grouped,indent=2))


if __name__=='__main__':main()
