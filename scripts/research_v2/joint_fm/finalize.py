"""Compact complete41 joint metrics with explicit legacy-training warning."""
from pathlib import Path
import argparse,csv,gzip,json,os,time,statistics,hashlib


def validate(rows,parents):
    expected={(f,name,n) for f in range(0,4001,100) for name in ('no_encoder','with_encoder') for n in (1,2,4,8)}
    keys={(r['frame'],r['variant'],r['stage2_nfe']) for r in rows}
    if len(rows)!=328 or keys!=expected or len(parents)!=41:raise ValueError('Complete41x2x4 records required')
    for f in range(0,4001,100):
        q=[r for r in rows if r['frame']==f]
        if len({r['parent_ids_sha256'] for r in q})!=1:raise ValueError('Encoder arms did not share exact predicted support')
        if any(r['extra_parent_count']!=0 or r['gt_support_used'] or r['parent_expansion']!=0 for r in q):
            raise ValueError('Forbidden extra/GT Parents in actual inference')


def finalize(run,docs):
    result=json.loads((run/'results.json').read_text());rows=[json.loads(x) for x in (run/'per_frame.jsonl').open()]
    parents=json.loads((run/'stage1_per_frame.json').read_text());validate(rows,parents)
    if result['status']!='EVALUATED_JOINT_PREDICTED41':raise ValueError('Incomplete evaluation')
    docs.mkdir(parents=True,exist_ok=False)
    for src in ('protocol.json','results.json','summary.csv','stage1_per_frame.json'):
        (docs/src).write_bytes((run/src).read_bytes())
    with gzip.open(docs/'per_frame.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    old=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-explicit-fp16-training-v1/experiments/desescene_explicit_fp16_v1/S1-FM-Scratch20K/evaluation/seq08_41/per_frame.jsonl')
    reference={r['frame']:r['parent'] for r in map(json.loads,old.open()) if r['nfe']==2}
    replay=all(all(r['stage1'][k]==reference[r['frame']][k] for k in ('tp','fp','fn','active_parents')) for r in parents)
    if not replay:raise ValueError('Stage1 count replay differs; review rather than label exact reference')
    paired=[]
    for n in (1,2,4,8):
        by={(r['frame'],r['variant']):r for r in rows if r['stage2_nfe']==n}
        v=[by[f,'with_encoder']['native']['fine']['iou']-by[f,'no_encoder']['native']['fine']['iou'] for f in range(0,4001,100)]
        paired.append(dict(nfe=n,encoder_minus_noencoder_iou=statistics.mean(v),wins=sum(d>0 for d in v),frames=41,
                           min_delta=min(v),max_delta=max(v)))
    (docs/'paired_deltas.json').write_text(json.dumps(paired,indent=2))
    text='''# Stage1->Stage2真实联合推理：零额外Parent /seq08 skip10041帧

**训练协议违规保留说明：本表Stage2是此前错误加入25%邻近空Parent训练的旧20K权重，不是按用户要求重训的GT-only/no-extra-Parent模型。此次仅纠正联合推理候选，不会抹去训练历史。**

普通FM Stage1 Scratch20K EMA，NFE2，threshold0.5；两组Stage2 Scratch20K EMA，主NFE4，另测1/2/8。原LiDAR/ROI/Source seed，完整41帧。没有GT候选、邻域扩增、额外空Parent、Observed Parent union、TopK截断或GT过滤。Stage1自然产生的假阳性保留。GT在两个网络完成所有占用预测后才用于评价。

两组Parent-ID SHA逐帧完全一致；额外Parent为0；Stage1 TP/FP/FN/count在全41帧与原普通FM NFE2参考完全一致。所有GT计入Fine FN，包括Stage1漏掉的Parent；另列Support-only IoU。点输出为原生Fine Voxel Centers，无Jitter。全部使用当前20K研究权重，未混入成熟100K模型。

## Stage1候选支撑

'''
    s=result['stage1_macro'];text+=f"Parent P={s['precision']*100:.4f}%, R={s['recall']*100:.4f}%, F1={s['f1']*100:.4f}%, IoU={s['iou']*100:.4f}%;平均Parents={s['active_parents']:.2f};Fine-GT Recall Ceiling={s['fine_recall_ceiling']*100:.4f}%。\n"
    text+='\n## 全GT Native Fine指标\n\n|Stage2|NFE|P %|R %|F1 %|Fine IoU %|Support-only IoU %|Occupied Children|Missed-parent Fine FN|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in result['summary']:
        text+=f"|{r['variant']}|{r['stage2_nfe']}|{r['fine_precision']*100:.4f}|{r['fine_recall']*100:.4f}|{r['fine_f1']*100:.4f}|{r['fine_iou']*100:.4f}|{r['support_only_fine_iou']*100:.4f}|{r['occupied_child_count']:.1f}|{r['missed_parent_fine_fn']:.1f}|\n"
    text+='\n## 连续几何\n\n|Stage2|NFE|CD m|IoU0.5 %|IoU0.2 %|IoU0.1 %|JSD-BEV|JSD-3D|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in result['summary']:
        text+=f"|{r['variant']}|{r['stage2_nfe']}|{r['cd_m']:.6f}|{r['iou05']*100:.4f}|{r['iou02']*100:.4f}|{r['iou01']*100:.4f}|{r['jsd_bev']:.6f}|{r['jsd_3d']:.6f}|\n"
    text+='''\n## 结论边界\n\n此前GT粗支撑Oracle表中的约50%/58% Fine IoU不能当作实际联合质量。当前结果包含真实Support漏检与Stage2细化误差。旧模型还存在训练GT支持/恒定Score1与推理预测支持/真实Score的分布差异；本次不通过使用GT或改阈值掩盖它们。\n\n不将质量评价运行时间当作网络推理时延。未训练任何新模型；无额外Seed/预算。当前分支的GT Parent helper已默认0且拒绝非零扩增；历史训练器若仍传入.25会明确报错。需要符合用户要求的训练结论，必须使用另行注册的纯GT/no-extra-Parent训练，而不是给旧权重改标签。\n'''
    (docs.parent/'JOINT_FM_PREDICTED_NO_EXTRA_PARENT_RESULTS.md').write_text(text)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in docs.iterdir() if p.is_file()}
    (docs/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(result['summary'],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--docs',type=Path,required=True)
    p.add_argument('--wait-pid',type=int);a=p.parse_args()
    while not (a.run/'results.json').exists():
        if not a.wait_pid:raise FileNotFoundError('Incomplete result')
        try:os.kill(a.wait_pid,0)
        except ProcessLookupError:raise RuntimeError('Evaluator exited before complete41 result')
        time.sleep(15)
    finalize(a.run,a.docs)


if __name__=='__main__':main()
