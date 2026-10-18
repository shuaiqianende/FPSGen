"""Bind actual GT-only morphology GPU/restore evidence to new source and data."""
from common import ROOT
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,source_sha,config_sha
from fpsgen.hier_occ_flow.morphology import MORPH_DATA_VERSION
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gpu-reports',type=Path,nargs=2,required=True)
    p.add_argument('--resume-report',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg,g=load_stage2_config(a.config);train,dev,identity=dataset_identity(cfg,g,verify=True)
    if cfg.get('target_data_version')!=MORPH_DATA_VERSION:raise ValueError('Explicit new morphology target required')
    maximum=int(train.index['length'].max());physical=set();evidence=[]
    for path in a.gpu_reports:
        v=json.loads(path.read_text())
        fields={'geometry_sha256':g.sha256,'source_sha256':source_sha(),
                'data_identity':identity,'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],
                **{k:cfg[k] for k in ('target_data_version','target_transform_sha256','target_processor_sha256')}}
        if (v.get('status')!='PASS_PACKED_BATCH2' or any(v.get(k)!=value for k,value in fields.items())
            or v.get('candidate_policy')!=cfg['candidates'] or v.get('parameters')!=2677128
            or v.get('stage1_unchanged') is not True or v.get('checkpointing_parity',{}).get('pass') is not True):
            raise ValueError('Actual new-target/source GPU evidence is missing or failed')
        rows=v['rows'];mixed=next(r for r in rows if r['kind']=='unequal_real_scenes')
        largest=next(r for r in rows if r['kind']=='two_complete_maximum_scenes')
        if (largest['tokens_per_scene']!=[maximum,maximum] or largest['tokens_total']!=2*maximum
            or any(not mixed.get(k) for k in ('permutation_pass','scene_isolation_pass','packed_vs_separate_pass','fp32_packed_vs_separate_pass'))):
            raise ValueError('No complete maximum/permutation/packing GPU proof')
        for row in rows:
            checks=row.get('actual_condition_checks',[])
            if (not row['finite'] or not row['optimizer_update_executed'] or row['gradient_norm']<=0
                or row.get('flash_calls',0)<=0 or row.get('spconv_calls',0)<=0 or len(checks)!=2
                or any(not x['actual_score_matches_stage1'] or not x['gt_parents_only'] or x['extra_empty_parents']!=0 for x in checks)):
                raise ValueError('No actual Flash/SpConv update or real-score/GT-only evidence')
        physical.add(v['physical_gpu']);evidence.append({'path':str(path.resolve()),'sha256':file_sha(path)})
    if physical!={2,3}:raise ValueError('Both physical GPUs2/3 required')
    restored=False
    if a.resume_report:
        v=json.loads(a.resume_report.read_text())
        if v['status']!='PASS_DDP_RESUME' or v.get('global_batch')!=4 or v.get('batch_per_gpu')!=2:
            raise ValueError('Actual GB4 DDP restore comparison required')
        for entry in v['checkpoints'].values():
            path=Path(entry['path'])
            if file_sha(path)!=entry['sha256']:raise ValueError('Restore checkpoint bytes changed')
            saved=torch.load(path,map_location='cpu');ident=saved['identity']
            previous=json.loads((Path(ident['output'])/'config.json').read_text())
            if (ident.get('source_sha256')!=source_sha() or any(ident.get(k)!=value for k,value in identity.items())
                or ident.get('config_sha256')!=config_sha(previous) or previous['candidates']!=cfg['candidates']
                or previous['model']!=cfg['model'] or saved.get('parameter_sync_pass') is not True
                or any(previous.get(k)!=cfg.get(k) for k in ('target_data_version','target_transform_sha256','target_processor_sha256'))):
                raise ValueError('DDP restore belongs to another source/target/protocol')
        restored=True;evidence.append({'path':str(a.resume_report.resolve()),'sha256':file_sha(a.resume_report)})
    gate={'schema':'hs-occflow-stage2-morph-engineering-v1','geometry_sha256':g.sha256,
        'source_sha256':source_sha(),'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],
        **{k:cfg[k] for k in ('target_data_version','target_transform_sha256','target_processor_sha256')},
        **identity,'full_scene_gpu':True,'maximum_scene_gpu':True,'packed_batch2_gpu':True,
        'ptv3_permutation':True,'checkpointing_parity':True,'ddp_resume':restored,'packed_batch2_ddp_resume':restored,
        'oracle_bce_overfit':False,'oracle_fm_overfit':False,'matched_bce_fm_1k':False,'pilot5000':False,
        'maximum_gt_parents':maximum,'maximum_b2_tokens':2*maximum,'physical_gpus':[2,3],
        'oracle_quality_pass_claimed':False,'evidence':evidence,
        'status':'PASS_MORPH_ENGINEERING' if restored else 'READY_FOR_MORPH_DDP_SMOKE'}
    atomic_json(a.output,gate);print(gate['status'],flush=True)


if __name__=='__main__':main()
