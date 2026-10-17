"""Validate saved experimental evidence on CPU; never runs or invents GPU tests."""
from common import ROOT
import argparse
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.stage2_evidence import compare_resume,overfit_evidence,matched_pilot_evidence,build_engineering_gate
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def main():
    parser=argparse.ArgumentParser(__doc__);sub=parser.add_subparsers(dest='action',required=True)
    resume=sub.add_parser('resume');resume.add_argument('--continuous',type=Path,required=True);resume.add_argument('--resumed',type=Path,required=True)
    overfit=sub.add_parser('overfit');overfit.add_argument('--run',type=Path,required=True);overfit.add_argument('--objective',choices=('bce','fm'),required=True)
    pilot=sub.add_parser('pilot-compare');pilot.add_argument('--bce-run',type=Path,required=True);pilot.add_argument('--fm-run',type=Path,required=True)
    gate=sub.add_parser('gate');gate.add_argument('--config',type=Path,required=True)
    gate.add_argument('--gpu-reports',type=Path,nargs='*',default=[]);gate.add_argument('--resume-report',type=Path)
    gate.add_argument('--overfit-reports',type=Path,nargs='*',default=[])
    gate.add_argument('--pilot-report',type=Path)
    for command in (resume,overfit,pilot,gate):command.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='resume':
        report=compare_resume(torch.load(args.continuous,map_location='cpu'),torch.load(args.resumed,map_location='cpu'))
        report['checkpoints']={name:{'path':str(path.resolve()),'sha256':file_sha(path)} for name,path in
                               (('continuous',args.continuous),('resumed',args.resumed))}
    elif args.action=='overfit':report=overfit_evidence(args.run,args.objective)
    elif args.action=='pilot-compare':report=matched_pilot_evidence(args.bce_run,args.fm_run)
    else:
        cfg,geometry=load_stage2_config(args.config)
        report=build_engineering_gate(geometry.sha256,cfg['stage1']['checkpoint_sha256'],gpu_reports=args.gpu_reports,
                                     resume_report=args.resume_report,overfit_reports=args.overfit_reports,pilot_report=args.pilot_report)
    atomic_json(args.output,report);print(report,flush=True)
    if report['status'].startswith('FAIL'):raise SystemExit(2)


if __name__=='__main__':main()
