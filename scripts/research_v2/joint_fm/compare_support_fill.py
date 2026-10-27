"""Publish the all-valid-Children control after verifying identical41-frame support."""
from pathlib import Path
import argparse,hashlib,json


def compare(docs,control):
    result=json.loads((docs/'results.json').read_text())
    rows=[json.loads(x) for x in __import__('gzip').open(docs/'per_frame.jsonl.gz','rt')]
    fill=json.loads(control.read_text())
    if fill['status']!='EVALUATED_NATIVE_SUPPORT_FILL41' or len(fill['rows'])!=41:
        raise ValueError('Complete41-frame support-fill control required')
    hashes={r['frame']:r['parent_ids_sha256'] for r in fill['rows']}
    if set(hashes)!=set(range(0,4001,100)) or any(hashes[r['frame']]!=r['parent_ids_sha256'] for r in rows):
        raise ValueError('Stage1 support differs between control and joint inference')
    if any(r['extra_parent_count'] for r in rows+fill['rows']):raise ValueError('Unexpected extra Parents')
    (docs/'support_fill_control.json').write_bytes(control.read_bytes())
    s=fill['summary']
    entries=[dict(variant='Stage1 support filled with all valid Children',fine_iou=s['iou'],
                  fine_precision=s['precision'],fine_recall=s['recall'],**{k:s[k] for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')})]
    entries.extend(r for r in result['summary'] if r['stage2_nfe']==4)
    text='''# Same-support Stage1 fill versus Stage2 refinement

41 frames, seq08/skip100, native Fine voxel centers, Stage1 ordinary FM NFE2/threshold0.5.
All41 Parent-ID hashes match both joint-inference arms at every NFE. No extra or GT-selected Parents.
The fill control occupies every valid Child inside the predicted Parents; it is not a learned Fine decoder.
Stage2 rows use NFE4 and the historical25%-augmentation-trained weights, which remain noncompliant training results.

|Output|Native Fine IoU %|P %|R %|CD m|IoU0.5 %|IoU0.2 %|IoU0.1 %|JSD-BEV|JSD-3D|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in entries:
        text+=f"|{r['variant']}|{r['fine_iou']*100:.4f}|{r['fine_precision']*100:.4f}|{r['fine_recall']*100:.4f}|{r['cd_m']:.6f}|{r['iou05']*100:.4f}|{r['iou02']*100:.4f}|{r['iou01']*100:.4f}|{r['jsd_bev']:.6f}|{r['jsd_3d']:.6f}|\n"
    text+='\nMacro averages over41 fixed frames. Different occupancy densities can trade off native IoU, CD and physical IoU; report them together. No latency claim or compliant retraining is implied.\n'
    (docs.parent/'JOINT_FM_SUPPORT_FILL_CONTROL.md').write_text(text)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in docs.iterdir() if p.is_file() and p.name!='SHA256SUMS.json'}
    (docs/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(text)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--docs',type=Path,required=True);p.add_argument('--control',type=Path,required=True)
    a=p.parse_args();compare(a.docs,a.control)
