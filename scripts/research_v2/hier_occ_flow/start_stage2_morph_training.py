"""Actual data -> GPU2/3 -> GB4 restore -> fresh100K-frame FM launch.

No checkpoint/model is inherited from the smoke or the historical experiment.
Every transition records its real status; failed gates terminate this launcher.
"""
from common import ROOT
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
import torch
from fpsgen.hier_occ_flow.stage2_runtime import load_stage2_config,dataset_identity,require_training_gate,source_sha,config_sha
from fpsgen.hier_occ_flow.storage import atomic_json,file_sha


def state(path,phase,**values):
    value={'phase':phase,'time_epoch':time.time(),'controller_pid':os.getpid(),**values}
    atomic_json(path,value);print(json.dumps(value,ensure_ascii=False),flush=True)


def verify_launch(output):
    start=torch.load(output/'checkpoints/step000000.pt',map_location='cpu')
    current=torch.load(output/'checkpoints/step000020.pt',map_location='cpu')
    launch=json.loads((output/'launch.json').read_text())
    changed=[k for k,v in start['model'].items() if not torch.equal(v,current['model'][k])]
    if (start['successful_updates']!=0 or start['optimizer']['state'] or launch['optimizer_entries']!=0
        or launch['transition'] is not None or launch['global_batch']!=4
        or current['successful_updates']!=20 or current['sampler']['successful_frame_exposures']!=80
        or not changed or current['parameter_sync_pass'] is not True or len(current['rank_rng'])!=2):
        raise ValueError('Fresh Stage2 initialization/actual updates/DDP checkpoint proof failed')
    for name,value in start['model'].items():
        if not torch.equal(value,start['ema'][name]):raise ValueError('Initial EMA differs from fresh online')
    rows={}
    for rank in (0,1):
        entries=[json.loads(line) for line in (output/f'train_rank{rank}.jsonl').read_text().splitlines() if line.strip()]
        last=next(r for r in reversed(entries) if r['finite'])
        if last['successful_updates']<20 or last['gradient_norm']<=0 or last['loss'] is None:
            raise ValueError('Missing finite nonzero actual rank updates')
        rows[str(rank)]={k:last[k] for k in ('successful_updates','successful_frame_exposures','loss','gradient_norm',
            'amp_skips','seconds','tokens','peak_allocated_bytes','peak_reserved_bytes')}
    return {'status':'PASS_FRESH_RUNNING','initialization':'scratch_stage2','inherited_stage2_weights':False,
        'initial_optimizer_entries':len(start['optimizer']['state']),'initial_successful_updates':0,
        'initial_model_sha256':start['rank_hashes'][0]['online'],
        'initial_ema_sha256':start['rank_hashes'][0]['ema'],
        'initial_checkpoint_sha256':file_sha(output/'checkpoints/step000000.pt'),
        'update20_checkpoint_sha256':file_sha(output/'checkpoints/step000020.pt'),
        'changed_model_state_entries':len(changed),'global_batch':4,'rank_progress':rows}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight-output',type=Path,required=True)
    a=p.parse_args();cfg,g=load_stage2_config(a.config);smoke,_=load_stage2_config(a.smoke_config)
    a.output=a.output.resolve();a.preflight_output=a.preflight_output.resolve()
    a.preflight_output.mkdir(parents=True,exist_ok=True)
    status=a.preflight_output/'controller_status.json'
    if status.exists() or (a.output/'checkpoints').exists():raise FileExistsError('Fresh-only controller/output')
    env=os.environ.copy();env['HS_STAGE2_PYTHON']=sys.executable
    def run(command,**kw):
        print('Executing:',command,flush=True)
        subprocess.run(command,check=True,env=env,**kw)
    gate=Path(cfg['engineering_gate'])
    if gate!=Path(smoke['engineering_gate']):raise ValueError('Smoke/formal gate paths differ')
    if any(cfg.get(k)!=smoke.get(k) for k in ('train_dataset','development_dataset','target_data_version',
        'target_transform_sha256','target_processor_sha256','model','candidates','stage1')):
        raise ValueError('Smoke/formal target/model/condition mismatch')
    try:
        state(status,'WAITING_FOR_COMPLETE_MORPH_DATA',formal_started=False,output=str(a.output))
        while True:
            progress={};ready=True
            for name,key in (('train','train_dataset'),('development','development_dataset')):
                root=Path(cfg[key]);path=root/'build_state.json'
                s=json.loads(path.read_text()) if path.exists() else {}
                progress[name]={'completed_frames':s.get('completed_frames',0),'status':s.get('status','PENDING')}
                ready&=s.get('status')=='COMPLETE' and (root/'manifest.json').exists()
            if ready:break
            state(status,'WAITING_FOR_COMPLETE_MORPH_DATA',formal_started=False,progress=progress)
            time.sleep(30)
        train,dev,ident=dataset_identity(cfg,g,verify=True)
        state(status,'MORPH_DATA_VERIFIED',train_frames=len(train),development_frames=len(dev),identity=ident)
        run([sys.executable,'-m','pytest','-q','tests/test_stage2_real_score_gt_only.py',
            'tests/test_hier_occ_morphology.py','tests/test_stage2_frame_budget.py','tests/test_stage2_batching.py'])
        paths=[];jobs=[]
        state(status,'GPU2_GPU3_FULL_SCENE_MORPH_SMOKE',formal_started=False)
        for gpu in (2,3):
            out=a.preflight_output/f'gpu{gpu}_morph_batch2.json';paths.append(out)
            log=(a.preflight_output/f'gpu{gpu}_morph_batch2.log').open('w')
            child_env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)}
            proc=subprocess.Popen([sys.executable,'-u','scripts/research_v2/hier_occ_flow/benchmark_stage2_384_batch.py',
                '--config',str(a.smoke_config),'--output',str(out),'--physical-gpu',str(gpu)],
                env=child_env,stdout=log,stderr=subprocess.STDOUT)
            jobs.append((proc,log))
        codes=[]
        for proc,log in jobs:codes.append(proc.wait());log.close()
        if any(codes):raise RuntimeError(f'Actual morphology GPU smoke failed: {codes}')
        command=[sys.executable,'scripts/research_v2/hier_occ_flow/prepare_stage2_morph_gate.py',
            '--config',str(a.config),'--gpu-reports',*[str(p) for p in paths],'--output',str(gate)]
        run(command)
        smoke_out=a.preflight_output/'ddp_resume32'
        state(status,'MORPH_GB4_DDP_32_AND_RESTORE',formal_started=False)
        env['HS_STAGE2_SMOKE_CONFIG']=str(a.smoke_config)
        run(['bash','scripts/research_v2/hier_occ_flow/smoke_stage2_384_resume.sh',str(smoke_out)])
        run(command+['--resume-report',str(smoke_out/'resume_comparison.json')])
        require_training_gate(cfg,g)
        frozen={'config_sha256':config_sha(cfg),'source_sha256':source_sha(),**ident,
            'stage1_checkpoint_sha256':cfg['stage1']['checkpoint_sha256'],'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'target_frame_exposures':100000,'target_optimizer_updates':25000,'global_batch':4,
            'gpu_ids':[2,3],'initialization':'fresh_random_stage2','smoke_weights_reused':False}
        atomic_json(a.preflight_output/'frozen_training_identity.json',frozen)
        state(status,'STARTING_FRESH_FORMAL',formal_started=False,identity=frozen)
        child=subprocess.Popen(['bash','scripts/research_v2/hier_occ_flow/launch_stage2_384.sh',
            '--config',str(a.config),'--output',str(a.output)],env={**env,'CUDA_VISIBLE_DEVICES':'2,3'})
        checked=False
        while child.poll() is None:
            if not checked and (a.output/'checkpoints/step000020.pt').exists():
                proof=verify_launch(a.output);proof.update(identity=frozen,formal_launcher_pid=child.pid)
                atomic_json(a.output/'launch_verification.json',proof);checked=True
                state(status,'FORMAL_RUNNING_VERIFIED',formal_started=True,output=str(a.output),proof=proof)
            time.sleep(10)
        if child.returncode:raise RuntimeError(f'Formal training exited with {child.returncode}; preserve logs/checkpoints')
        summary=json.loads((a.output/'summary.json').read_text())
        phase='FORMAL_COMPLETE' if summary['formal_training_complete'] else 'FORMAL_PAUSED'
        state(status,phase,formal_started=True,successful_updates=summary['successful_updates'],
            successful_frame_exposures=summary['successful_frame_exposures'],output=str(a.output))
    except BaseException as exc:
        state(status,'FAILED_OR_BLOCKED',exception=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
