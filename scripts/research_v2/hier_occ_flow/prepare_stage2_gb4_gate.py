"""Add only actual B2 GPU/DDP evidence to the preserved384 base gate."""
from common import ROOT
import argparse,json
from pathlib import Path
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--base-gate',type=Path,required=True);p.add_argument('--gpu-reports',type=Path,nargs=2,required=True)
    p.add_argument('--resume-report',type=Path);p.add_argument('--fm-overfit-report',type=Path)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg,g=load_stage2_config(args.config);gate=json.loads(args.base_gate.read_text());physical=set()
    if gate['geometry_sha256']!=g.sha256 or gate['stage1_checkpoint_sha256']!=cfg['stage1']['checkpoint_sha256']:raise ValueError('Base identity mismatch')
    for path in args.gpu_reports:
        report=json.loads(path.read_text())
        if (report['status']!='PASS_PACKED_BATCH2' or report['geometry_sha256']!=g.sha256
                or report['stage1_checkpoint_sha256']!=gate['stage1_checkpoint_sha256']):raise ValueError('B2 real GPU evidence not passed')
        if max(r['tokens_total'] for r in report['rows'])<799344:raise ValueError('Two complete maximum scenes must be measured')
        physical.add(report['physical_gpu']);gate['evidence'].append({'path':str(path.resolve()),'sha256':file_sha(path),'status':report['status']})
    if physical!={2,3}:raise ValueError('Both physical GPUs required')
    gate['packed_batch2_gpu']=True;gate['packed_batch2_ddp_resume']=False
    if args.resume_report:
        report=json.loads(args.resume_report.read_text())
        if (report['status']!='PASS_DDP_RESUME' or report.get('global_batch')!=4 or report.get('batch_per_gpu')!=2
                or report['stage1_checkpoint_sha256']!=gate['stage1_checkpoint_sha256'] or report['geometry_sha256']!=g.sha256):raise ValueError('Actual GB4 DDp restore evidence required')
        gate['packed_batch2_ddp_resume']=True
        gate['evidence'].append({'path':str(args.resume_report.resolve()),'sha256':file_sha(args.resume_report),'status':report['status']})
    gate['status']='PACKED_GB4_OVERFIT_READY' if gate['packed_batch2_ddp_resume'] else 'PACKED_GB4_DDP_SMOKE_READY'
    if args.fm_overfit_report:
        report=json.loads(args.fm_overfit_report.read_text())
        if (report['status']!='PASS_ORACLE_OVERFIT' or report['objective']!='fm' or report['geometry_sha256']!=g.sha256
            or report['stage1_checkpoint_sha256']!=gate['stage1_checkpoint_sha256']):raise ValueError('Actual FM overfit quality gate failed')
        gate['oracle_fm_overfit']=True;gate['status']='PACKED_GB4_FM_PILOT_READY'
        gate['evidence'].append({'path':str(args.fm_overfit_report.resolve()),'sha256':file_sha(args.fm_overfit_report),'status':report['status']})
    atomic_json(args.output,gate);print(gate,flush=True)


if __name__=='__main__':main()
