"""Validate a complete interim41 run and retain descriptive budget-labelled tables."""
from pathlib import Path
import argparse,csv,json,statistics,time,os,gzip

BASE=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-explicit-fp16-training-v1/experiments/desescene_explicit_fp16_v1')
NATIVE=('precision','recall','f1','iou','active_parents','fine_recall_ceiling')
CONTINUOUS=('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')


def read_rows(path):return [json.loads(s) for s in path.open()]


def validate_rows(rows):
    expected={(f,n) for f in range(0,4001,100) for n in (1,2,4,8)}
    if len(rows)!=164 or {(r['frame'],r['nfe']) for r in rows}!=expected:
        raise ValueError('Must have all164 unique frame/NFE results; never summarize a partial set as41')


def table(rows,name,exposures):
    validate_rows(rows);out=[]
    for nfe in (1,2,4,8):
        a=[r for r in rows if r['nfe']==nfe]
        row=dict(model=name,successful_exposures=exposures,frames=41,nfe=nfe,
            **{k:statistics.mean(r['parent'][k] for r in a) for k in NATIVE},
            **{k:statistics.mean(r['continuous'][k] for r in a) for k in CONTINUOUS})
        for ix,k in enumerate((84963,120000)):
            for field in ('actual_k','precision','recall','f1','iou','fine_recall_ceiling'):
                row[f'fixedK{k}_{field}']=statistics.mean(r['fixed_k'][ix][field] for r in a)
        out.append(row)
    return out


def csv_write(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def finalize(run,docs):
    result=json.loads((run/'quality41/results.json').read_text())
    if result['status']!='EVALUATED_INTERIM' or result['protocol']['checkpoint_counts']['exposures']!=15000:
        raise ValueError('Own pinned15K interim evaluation required')
    new=read_rows(run/'quality41/per_frame.jsonl');validate_rows(new)
    fm=read_rows(BASE/'S1-FM-Scratch20K/evaluation/seq08_41/per_frame.jsonl')
    mf=read_rows(BASE/'S1-MeanFlow-Scratch20K/evaluation/seq08_41/per_frame.jsonl')
    values=table(fm,'Original FM',20000)+table(mf,'Original MeanFlow',20000)+table(new,'MFV2-A INTERIM',15000)
    output=run/'comparison';output.mkdir(exist_ok=False)
    csv_write(output/'macro_comparison.csv',values)
    docs.mkdir(parents=True,exist_ok=True)
    csv_write(docs/'macro_comparison.csv',values)
    (docs/'interim_protocol.json').write_text(json.dumps(result['protocol'],indent=2))
    # Compact original per-frame records; no weights, GT point clouds or inputs.
    with gzip.open(docs/'interim_per_frame.jsonl.gz','wt') as f:
        for r in new:f.write(json.dumps(r,separators=(',',':'))+'\n')
    text='''# MFV2-A 15K中期评价：seq08/skip10041帧

这是用户指定的最新完整保存EMA快照：7500成功更新、15000成功曝光、2次AMP Skip。不是最终20K结果；对照FM/原MeanFlow均训练20000曝光，因此下表是不同预算的描述性比较，不是同预算提升结论。

相同41帧、相同Source seed、原始GT/ROI、threshold0.5、NFE1/2/4/8。输出为Stage1粗Parent中心，无Stage2、无Jitter、无GT引导推理候选。未选择或调整阈值。中期评价独立脚本保留最终20K严格检查，原训练源码不变。

|模型|曝光|NFE|P %|R %|F1 %|Parent IoU %|Fine-GT Ceiling %|Parents|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in values:
        text+=f"|{r['model']}|{r['successful_exposures']}|{r['nfe']}|{r['precision']*100:.4f}|{r['recall']*100:.4f}|{r['f1']*100:.4f}|{r['iou']*100:.4f}|{r['fine_recall_ceiling']*100:.4f}|{r['active_parents']:.1f}|\n"
    text+='\n|模型|曝光|NFE|CD m|IoU0.5 %|IoU0.2 %|IoU0.1 %|JSD-BEV|JSD-3D|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in values:
        text+=f"|{r['model']}|{r['successful_exposures']}|{r['nfe']}|{r['cd_m']:.6f}|{r['iou05']*100:.4f}|{r['iou02']*100:.4f}|{r['iou01']*100:.4f}|{r['jsd_bev']:.6f}|{r['jsd_3d']:.6f}|\n"
    text+='\n完整两档Fixed-K P/R/F1/IoU/Ceiling均在`macro_comparison.csv`。逐帧原始指标保存在压缩JSONL；不以部分帧代替全41帧。A/B正式训练继续，最终20K结果和B结果仍为NOT TESTED；本次不增加训练预算或更改筛选门槛。\n'
    text+='\nGPU2评价显存受6GiB PyTorch allocator上限约束，只与本项目A共享；GPU0/1的其他用户任务不参与。共卡期间不作正式时延/加速宣称。\n'
    (docs.parent/'MEANFLOW_V2_A_INTERIM_15K_RESULTS.md').write_text(text)
    (output/'completion.json').write_text(json.dumps(dict(status='COMPLETE_INTERIM41_COMPARISON',interim=True,frames=41,rows=164,
        note='15K versus20K descriptive comparisons; no final GO conclusion'),indent=2))
    return values


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--docs',type=Path,required=True)
    p.add_argument('--wait-pid',type=int);a=p.parse_args()
    while not (a.run/'quality41/results.json').exists():
        if a.wait_pid:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:raise RuntimeError('Evaluator exited without complete41 results')
        else:raise FileNotFoundError('Evaluation incomplete')
        time.sleep(15)
    values=finalize(a.run,a.docs)
    print(json.dumps(values,indent=2),flush=True)


if __name__=='__main__':main()
