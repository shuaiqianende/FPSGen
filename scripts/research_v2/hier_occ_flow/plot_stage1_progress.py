"""CPU-only plots from actual optimizer and held-out evaluation records."""
from common import ROOT
import argparse
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'env/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monitor_stage1_occ import read_rows


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[r for r in read_rows(a.output/'train_rank0.jsonl') if r.get('finite')]
    steps=np.array([r['successful_updates'] for r in rows]);loss=np.array([r['loss'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    stride=max(1,len(rows)//2500)
    axes[0,0].plot(steps[::stride],loss[::stride],alpha=.15,lw=.6,label='individual FM MSE')
    if len(rows)>=100:axes[0,0].plot(steps[99:],np.convolve(loss,np.ones(100)/100,mode='valid'),label='rolling100')
    axes[0,0].set_yscale('log');axes[0,0].set_title('Pure masked FM loss');axes[0,0].legend()
    for name in ('encoder','flow'):
        axes[0,1].plot(steps[::stride],[r['gradient_norms'][name] for r in rows[::stride]],alpha=.7,lw=.7,label=name)
        axes[1,0].plot(steps,[r[f'lr_{name}'] for r in rows],label=name)
    axes[0,1].set_yscale('log');axes[0,1].set_title('Gradient L2 before clipping');axes[0,1].legend()
    axes[1,0].set_title('Actual learning rates');axes[1,0].legend()
    val=[]
    for path in sorted((a.output/'validation').glob('update_*.json')):
        d=json.loads(path.read_text());v=d['nfe_results']['1']['summary']
        val.append((int(path.stem.split('_')[-1]),v['frames'],v['macro']['f1'],v['macro']['iou']))
    for frames,marker in ((12,'.'),(900,'o')):
        subset=[v for v in val if v[1]==frames]
        for i,name in ((2,'F1'),(3,'IoU')):
            axes[1,1].plot([v[0] for v in subset],[v[i] for v in subset],marker=marker,label=f'{name}, dev{frames}',ls='-' if frames==900 else ':')
    axes[1,1].set_title('Online,1 NFE, native384 grid, score>0.5');axes[1,1].set_ylim(0,1);axes[1,1].legend()
    for ax in axes.flat:ax.set_xlabel('Successful optimizer updates');ax.grid(alpha=.2)
    fig.suptitle(f'HS-OccFlow Stage1 occupancy: actual progress through {steps[-1]:,} updates')
    dest=ROOT/'docs/research_v2/hs_occ384_stage1_reports/formal_training_progress.png'
    fig.savefig(dest,dpi=160);plt.close(fig);print(dest)


if __name__=='__main__':main()
