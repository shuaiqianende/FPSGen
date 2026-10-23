"""Register real M1 engineering evidence; GO is separate from smoke success."""
from common import ROOT
import argparse
import json
from pathlib import Path

from fpsgen.hier_occ_flow.stage2_runtime import source_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--gpu-verification',type=Path,required=True)
    p.add_argument('--smoke-outputs',type=Path,nargs=2,required=True)
    p.add_argument('--review-decision',type=Path)
    a=p.parse_args()
    gate=json.loads(a.gate.read_text());gpu=json.loads(a.gpu_verification.read_text())
    if gpu.get('status')!='PASS' or gpu['source_sha256']!=source_sha():
        raise ValueError('GPU correctness/source identity not verified')
    protocols=set()
    for folder in a.smoke_outputs:
        summary=json.loads((folder/'summary.json').read_text())
        launch=json.loads((folder/'launch.json').read_text())
        cfg=json.loads((folder/'config.json').read_text())
        rows=[json.loads(x) for x in (folder/'train_rank0.jsonl').read_text().splitlines() if x.strip()]
        if (summary['successful_updates']!=32 or summary['successful_frame_exposures']!=64
            or launch['global_batch']!=2 or launch['identity']['source_sha256']!=source_sha()
            or len(rows)!=32 or any(not r.get('finite') for r in rows)
            or launch['initialization']!='paired_finetune_stage2'):
            raise ValueError('Need two finite, matched-source B2 32-update smokes')
        protocols.add(cfg['research_protocol'])
    if protocols!={'desescene_sci_dynamic_b0_v1','desescene_sci_dynamic_m1_v1'}:
        raise ValueError('Both B0 and M1 engineering smokes required')
    gate.update(research_model_smoke=True,state_condition_gpu_verified=True,source_sha256=source_sha(),
                initialization_checkpoint_sha256=gpu['source_initialization']['checkpoint_sha256'],
                c1_review_complete=False,status='M1_ENGINEERING_READY_RESEARCH_REVIEW_PENDING')
    gate.setdefault('evidence',[]).append({'path':str(a.gpu_verification.resolve()),
                                         'sha256':file_sha(a.gpu_verification),'status':'PASS_M1_GPU'})
    if a.review_decision:
        review=json.loads(a.review_decision.read_text())
        if review.get('decision')!='PROCEED_M1_MATCHED20K' or not review.get('rationale'):
            raise ValueError('Explicit evidence-based research GO decision required')
        for item in review.get('evidence',[]):
            if file_sha(item['path'])!=item['sha256']:raise ValueError('Review evidence changed')
        if not review.get('evidence'):raise ValueError('Review needs actual result evidence')
        gate.update(c1_review_complete=True,status='M1_MATCHED20K_READY',
                    review_decision_sha256=file_sha(a.review_decision))
    atomic_json(a.gate,gate)
    print(json.dumps({'status':gate['status'],'new_formal_training_started':False},indent=2))


if __name__=='__main__':
    main()
