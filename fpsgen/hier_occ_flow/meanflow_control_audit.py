"""Evidence-based control reuse; model names or loss values are not identities."""
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .encoder_adaptation import STAGE1_SHA,STAGE2_SHA
from .encoder_adaptation_runtime import config_sha
from .storage import file_sha
from .stage1_batch_transition import batch_indices


def audit_control(cfg,*,require_complete=True):
    root=Path(cfg['control']['run_root']);result={'arm':cfg['control']['arm'],'run_root':str(root),'checks':{}}
    checks=result['checks']
    def check(name,value):checks[name]=bool(value)
    if cfg['arm']=='S1-MF20K':
        p=Path(cfg['control']['source_config']);c=yaml.safe_load(p.read_text())
        shared=('source_stage1','source_stage2','reference_stage2_config_sha256','train_dataset','development_dataset','data_root','geometry')
        check('source_data_geometry_support',all(c[k]==cfg[k] for k in shared))
        check('freeze_policy',c['arm']=='S1-FM20K')
        for key in ('seed','batch_per_gpu','successful_updates','successful_frame_exposures','precision','optimizer','weight_decay','warmup_updates','grad_clip','ema_decay','amp_initial_scale'):
            check(key,c['training'][key]==cfg['training'][key])
        check('flow_learning_rate',c['training']['lr_flow']==cfg['training']['learning_rate'])
        summary=root/'run/summary.json';checkpoint=root/'run/checkpoints/resume_latest.pt';journal=root/'run/train.jsonl'
        if require_complete:
            if not summary.exists():return dict(result,status='WAITING_CONTROL_EVALUATION',reason='Control training/evaluation not complete')
            value=json.loads(summary.read_text());check('completed20k',value.get('successful_updates')==10000 and value.get('successful_frame_exposures')==20000)
            proof=json.loads((root/'supervisor.json').read_text()) if (root/'supervisor.json').exists() else {}
            check('completed900',proof.get('status')=='COMPLETE20K_AND_900')
        base_config_sha=config_sha(c);expected_source=STAGE1_SHA
    else:
        c=json.loads(Path(cfg['reference_stage2_config']).read_text());t=c['training']
        launch=json.loads((root/'launch.json').read_text());init=launch['initialization_audit']
        check('source_online_original',init.get('checkpoint_sha256')==STAGE2_SHA and init.get('weights')=='model' and init.get('source_successful_frame_exposures')==99520)
        check('fresh_optimizer_ema_scheduler',init.get('optimizer_ema_scheduler_restarted') is True)
        check('no_new_architecture',not init['loaded']['new_keys'])
        check('frozen_stage1',launch['stage1']['checkpoint_sha256']==STAGE1_SHA and launch['stage1']['weights']=='ema')
        check('predicted_support',c['candidates']['mode']=='predicted' and c['candidates']['stage1_nfe']==2 and c['candidates']['selection']['threshold']==.5 and c['candidates']['selection']['expansion']==0)
        for key in ('seed','batch_per_gpu','successful_updates','successful_frame_exposures','precision','weight_decay','warmup_updates','grad_clip','ema_decay'):
            check(key,t[key]==cfg['training'][key])
        check('learning_rate',t['lr']==cfg['training']['learning_rate'])
        check('packed_complete_scenes',t['batching']=='packed_variable_length' and not c.get('cohort_indices') and not c.get('cache',{}).get('enabled',False))
        check('source_uniform_time',c.get('flow')=={'source':'gaussian','sigma':1.0,'target_mapping':'binary_symmetric'})
        summary=root/'summary.json';checkpoint=root/'checkpoints/resume_latest.pt';journal=root/'train_rank0.jsonl'
        if require_complete:
            value=json.loads(summary.read_text());check('completed20k',value['successful_updates']==10000 and value['successful_frame_exposures']==20000)
        base_config_sha=config_sha(c);expected_source=STAGE2_SHA
    # Check EVERY logged attempt's actual record IDs, including skipped AMP
    # attempts. This compares sampler policy, not a subset of favorable scenes.
    if journal.exists():
        order_ok=True;attempts=0;seed=cfg['training']['seed'];orders={}
        with journal.open() as stream:
            for line in stream:
                row=json.loads(line);a=int(row['attempts']);cursor=2*(a-1)
                expected=[]
                for pos in (cursor,cursor+1):
                    epoch,index=divmod(pos,17870)
                    if epoch not in orders:orders[epoch]=np.random.default_rng(seed+epoch).permutation(17870)
                    expected.append(int(orders[epoch][index]))
                records=row.get('batch_records',row.get('records',[]))
                actual=[r.get('sampler_index',r.get('index')) for r in records]
                order_ok &= actual==expected;attempts+=1
        check('all_attempt_data_order',order_ok and attempts>0);result['audited_attempts']=attempts
    elif require_complete:check('all_attempt_data_order',False)
    if require_complete and checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu');identity=saved['identity']
        check('final_config_identity',identity['config_sha256']==base_config_sha)
        for split in ('train','development'):
            check(split+'_manifest',identity[split+'_manifest_sha256']==file_sha(Path(cfg[split+'_dataset'])/'manifest.json'))
        if cfg['arm']=='S2-MF20K':
            check('full_source_rng_scaler_scheduler',len(saved['rank_rng'])==1 and 'source' in saved['rank_rng'][0] and saved['scaler'] and saved['scheduler']['last_epoch']==10000)
        else:check('full_source_rng_scaler_scheduler','source' in saved['rng'] and saved['scaler'] and saved['scheduler']['last_epoch']==10000)
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=file_sha(checkpoint),config_sha256=base_config_sha)
    elif require_complete:check('final_checkpoint_exists',False)
    result.update(status='MATCHED_CONTROL' if checks and all(checks.values()) and require_complete else 'UNMATCHED_REFERENCE_ONLY',
        source_rng_policy='seed+1000; original source then uniform per-scene time; independent new interval RNG does not consume source stream',
        precision_forward_gate='Actual source/prepared zero-init parity is an additional mandatory GPU gate',
        shared_parameter_optimizer='AdamW, original per-scene masked MSE; new interval parameters use same LR')
    return result
