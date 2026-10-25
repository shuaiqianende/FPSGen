"""900 Development or frozen seq08/skip10041, NFE1/2/4/8 matched sweeps."""
from common import ROOT
import argparse,csv,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from fpsgen.hier_occ_flow.meanflow_training import load_config,admission
from fpsgen.hier_occ_flow.meanflow_evaluation import EvaluationModels
from fpsgen.hier_occ_flow.encoder_adaptation_runtime import datasets,config_sha
from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics
from fpsgen.hier_occ_flow.stage1_curve_metrics import fixed_budget_metrics
from fpsgen.hier_occ_flow.evaluation import fine_metrics
from fpsgen.hier_occ_flow.voxelization import voxelize_surface
from fpsgen.hier_occ_flow.dataset import input_xyz
from fpsgen.hier_occ_flow.raw_gt import RawMapSource
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha
from fpsgen.hier_occ_flow.provenance import raw_source_snapshot
from fpsgen.utils.gpu_kdtree_metrics import GPUKDTreeMetrics
sys.path.insert(0,str(ROOT/'scripts/research_v2/hier_occ_inference'))
from unified_pointcloud_metrics import evaluate_prediction


class MetricRecorder:
    def __init__(self,backend):self.backend=backend;self.last=None
    def evaluate(self,p,g):self.last=self.backend.evaluate(p,g);return self.last


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=('development900','seq08_41'),required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve partial/full evaluation; no implicit restart')
    cfg,g=load_config(a.config);gpu=admission(int(os.environ.get('CUDA_VISIBLE_DEVICES','-1')),True)
    if gpu['status']!='AVAILABLE':raise RuntimeError(json.dumps(gpu))
    torch.set_num_threads(4);a.output.mkdir(parents=True)
    _,dev,manifests=datasets(cfg,g,verify=True,verified_receipt=cfg['verified_data_receipt'])
    models=EvaluationModels(cfg,g,a.checkpoint,'cuda');raw=RawMapSource(cfg['data_root'],fast_viewpoint=True)
    metric=MetricRecorder(GPUKDTreeMetrics(('cd',),geometry=g));reference=None
    if a.split=='development900':records=[(r['sequence'].decode(),int(r['frame']),i) for i,r in enumerate(dev.index)]
    else:
        reference=Path(cfg['validation']['seq08_reference']);protocol=json.loads((reference/'protocol.json').read_text());frames=protocol['frames']
        if ([r['frame'] for r in frames]!=cfg['validation']['seq08_frames'] or any(r['sequence']!='08' for r in frames)
            or protocol['geometry_sha256']!=g.sha256):raise ValueError('Frozen41-frame manifest changed')
        sources,digest=raw_source_snapshot(cfg['data_root'],[('08',r['frame']) for r in frames])
        if digest!=protocol['raw_source_manifest_sha256']:raise ValueError('Frozen raw GT sources changed')
        atomic_json(a.output/'raw_sources.json',sources);records=[('08',r['frame'],i) for i,r in enumerate(frames)]
    protocol=dict(split=a.split,frames=len(records),nfes=[1,2,4,8],variants=['matched_fm','meanflow'],config_sha256=config_sha(cfg),
        checkpoint_sha256=models.checkpoint_sha,training_identity=models.identity,control=models.control_audit,manifests=manifests,
        gpu=gpu,point_mode='native_fine_centers',same_absolute_parent_noise=True,no_gt_candidate_selection=True,
        shared_other_stage='original Stage2 NFE4' if cfg['arm']=='S1-MF20K' else 'original Stage1 NFE2',
        shortcut_status='NOT_TESTED: separate queue has no completed compatible Shortcut checkpoint',
        formal_timing='NOT_TESTED here; quality-run timings include contention and are not speed claims')
    atomic_json(a.output/'protocol.json',protocol);rows=[]
    with (a.output/'per_frame.jsonl').open('x',buffering=1) as log,torch.no_grad():
        for sequence,frame,index in records:
            path=Path(cfg['data_root'])/sequence/'input_'/f'{frame:06d}.npy';xyz=torch.from_numpy(input_xyz(path)).cuda()
            raw_gt=raw.load(sequence,frame);_,inside=g.voxelize_points(raw_gt);target=np.ascontiguousarray(raw_gt[inside],dtype=np.float32)
            if a.split=='development900':record=dev[index]
            else:
                ids,masks,_=voxelize_surface(raw_gt,g);record=dict(sequence=sequence,frame=frame,parent_ids=ids,child_occ_mask=masks)
                meta=json.loads((reference/'targets'/f'08_{frame:06d}.json').read_text())
                if meta['input_sha256']!=file_sha(path):raise ValueError('Frozen input hash changed')
                with np.load(reference/'targets'/f'08_{frame:06d}.npz',allow_pickle=False) as saved:
                    if not np.array_equal(ids,saved['parent_ids']) or not np.array_equal(masks,saved['child_masks']):raise ValueError('Frozen surface-hit targets changed')
            obs_ids,obs_masks,_=voxelize_surface(xyz.cpu().numpy(),g)
            for nfe in (1,2,4,8):
                for variant in ('matched_fm','meanflow'):
                    out=models.generate(xyz,sequence,frame,variant=variant,nfe=nfe)
                    if not len(out['points']):raise RuntimeError('Empty prediction; no skipped/removed frame')
                    native=fine_metrics(out['ids'],out['occupied'],record['parent_ids'],record['child_occ_mask'],g,
                        observed_parent_ids=obs_ids,observed_masks=obs_masks,structural=True)
                    continuous=evaluate_prediction(out['points'],target,g,metric,device='cuda')
                    parent=parent_metrics(out['score'],record,g,models.valid,.5)
                    fixed=fixed_budget_metrics(out['score'].cpu().numpy().reshape(-1),record,models.valid.cpu().numpy().reshape(-1),[84963,120000])
                    # Separate exact bidirectional CD components, same backend.
                    parts=metric.last
                    row=dict(sequence=sequence,frame=frame,nfe=nfe,variant=variant,parent=parent,fixed_k=fixed,
                        native=native,continuous=continuous,cd_backend_components=parts,parent_ids_sha256=out['parent_ids_sha256'],
                        input_sha256=file_sha(path),occupied_child_count=len(out['points']),runtime_quality_only=out['timing'])
                    # Convert any metric scalar tensors before writing JSON.
                    row['cd_backend_components']={k:float(v) for k,v in parts.items()}
                    log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
            atomic_json(a.output/'progress.json',dict(status='RUNNING',frames_completed=index+1,frames_requested=len(records),rows=len(rows)))
            print(f"{cfg['arm']} {a.split} frames={index+1}/{len(records)}",flush=True)
    summary=[]
    for nfe in (1,2,4,8):
        for variant in ('matched_fm','meanflow'):
            subset=[r for r in rows if r['nfe']==nfe and r['variant']==variant]
            summary.append(dict(nfe=nfe,variant=variant,frames=len(subset),parent_iou=float(np.mean([r['parent']['iou'] for r in subset])),
                fine_recall_ceiling=float(np.mean([r['parent']['fine_recall_ceiling'] for r in subset])),
                native_fine_iou=float(np.mean([r['native']['fine']['iou'] for r in subset])),
                **{k:float(np.mean([r['continuous'][k] for r in subset])) for k in ('cd_m','iou05','iou02','iou01','jsd_bev','jsd_3d')}))
    from fpsgen.hier_occ_flow.meanflow_result_analysis import paired_analysis
    analysis=paired_analysis(rows,cfg['arm'])
    atomic_json(a.output/'results.json',dict(status='EVALUATED',frames=len(records),summary=summary,paired=analysis,
        protocol=protocol,formal_runtime_status='NOT_TESTED; separate exclusive paired benchmark'))
    with (a.output/'summary.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    atomic_json(a.output/'progress.json',dict(status='EVALUATED',frames_completed=len(records),rows=len(rows)))

if __name__=='__main__':main()
