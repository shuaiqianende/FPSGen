"""Summarize actual primary C1 metrics and paired Encoder timings; no automatic training approval."""
import argparse
import json
from pathlib import Path

import numpy as np
from audit_desescene_evidence import paired_delta


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--phase0-root',type=Path,required=True)
    a=p.parse_args();root=a.phase0_root
    fm=json.loads((root/'c1/static_context_fm_20k_900.json').read_text())['nfe_results']['4']
    bce=json.loads((root/'c1/static_context_bce_20k_900.json').read_text())['nfe_results']['1']
    if len(fm['rows'])!=900 or len(bce['rows'])!=900:raise ValueError('Complete primary900 required')
    for x,y in zip(fm['rows'],bce['rows']):
        if x['input_sha256']!=y['input_sha256'] or x['parent_ids_sha256']!=y['parent_ids_sha256']:
            raise ValueError('C1 paired input/support identity differs')
    comparisons={'fine_iou':paired_delta(fm['rows'],bce['rows'],nested=True)}
    for metric in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d'):
        x=[{**r,**r['continuous']} for r in fm['rows']]
        y=[{**r,**r['continuous']} for r in bce['rows']]
        comparisons[metric]=paired_delta(x,y,metric=metric)
    timings={}
    manifest=None;gpu=None
    for name in ('r0_gpu2_20k','e1_gpu3_20k','e2_gpu2_20k','e3_gpu3_20k'):
        rounds=[]
        for run in range(1,4):
            path=root/'d4_encoder'/(name+'_round'+str(run)+'.json')
            if not path.exists():continue
            x=json.loads(path.read_text())
            if manifest is None:manifest=x['input_manifest_sha256'];gpu=x['physical_gpu']
            if x['input_manifest_sha256']!=manifest or x['physical_gpu']!=gpu:raise ValueError('Encoder timing protocol differs')
            rounds.append(x['summary'])
        timings[name]={'rounds':rounds,'status':'COMPLETE' if len(rounds)==3 else 'NOT TESTED'}
    result={'status':'RESEARCH_REVIEW_REQUIRED','comparison':'BCE minus Static FM; no matched dynamic20K control yet',
            'native_and_continuous_pairs':comparisons,'encoder_timings':timings,
            'm1_training_approved':False,'reason':'Review C1 together with frozen dynamic D1/D2, then explicitly register GO/HOLD. Static vs BCE alone cannot establish the value of dynamic PTv3.'}
    (root/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# DeSeScene Phase0 results for research review','','Primary C1 uses the original NFE4 Parent protocol. B0/M1 is a separate NFE2 fine-tuning cohort.','',
           '| Metric | Static FM (4 NFE) | BCE (1 pass) |','|---|---:|---:|',
           f"| Native Fine IoU | {fm['summary']['macro']['iou']:.6f} | {bce['summary']['macro']['iou']:.6f} |"]
    for metric in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d'):
        lines.append(f"| {metric} | {fm['continuous_macro'][metric]:.6f} | {bce['continuous_macro'][metric]:.6f} |")
    lines+=['','Both use the same 900 frames, seed, input SHA and exact Parent-ID SHA. Full-GT native metrics include missed Parents. Paired 100-frame-group bootstrap intervals and worst-frame deltas are in review.json.','','No new M1/B0 formal training was started by the Phase0 queue.']
    (root/'RESEARCH_REVIEW.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
