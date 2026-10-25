"""Run only registered EF1/EF2; defer exclusive speed tests, preserve all gates."""
from common import ROOT
import argparse
import fcntl
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from run_encoder_adaptation_phases import Queue, P0, read
from fpsgen.hier_occ_flow.adaptation_gpu_slot import adaptation_admission
from fpsgen.hier_occ_flow.storage import atomic_json


class SharedQueue(Queue):
    def __init__(self, folder):
        if os.environ.get('DESESCENE_SHARED_ADAPTATION') != '1':
            raise ValueError('Explicit shared-run opt-in required')
        super().__init__(folder, False)
        previous = self.folder/'registration.json'
        data = read(previous)
        data.update(branch='research/desescene-encoder-adaptation-shared-v1',
                    shared_training_user_authorized=True, allocator_limit_gib=12,
                    shared_admission_minimum_free_gib=14,
                    timing_deferred=True, phase_order=['P1'],
                    p2_requires_completed_p1=True,
                    science_changes='none: admission/scheduling only; fresh source identity')
        atomic_json(previous,data)

    def status(self, state, **extra):
        super().status(state, **extra)
        print(state, extra, flush=True)

    def idle(self, gpu):
        while True:
            self.intact()
            admission=adaptation_admission(gpu)
            if admission['status']=='AVAILABLE': return
            atomic_json(self.folder/f'gpu{gpu}_wait.json',
                        dict(status='WAITING_SHARED_HEADROOM', admission=admission, epoch=time.time()))
            time.sleep(20)

    @contextmanager
    def lease(self, preferred=2):
        # One project job per physical card; retain deterministic EF assignment.
        self.idle(preferred)
        with self.gpu_lock:
            if preferred in self.gpu_leases: raise RuntimeError('Project GPU already leased')
            self.gpu_leases.add(preferred)
        try: yield preferred
        finally:
            with self.gpu_lock: self.gpu_leases.remove(preferred)

    def execute(self):
        previous=P0/'gpu2_transfer/summary.json'
        if read(previous)['status']!='COMPLETE': raise RuntimeError('P0 quality is not complete')
        receipt=read(read(previous)['full_lineage_audit_receipt'])
        if receipt['status']!='PASS': raise RuntimeError('Original data content checks did not pass')
        for arm in ('position','attention'):
            folder=P0/(arm+'_20k')/'evaluation'
            main=read(folder/'primary900_nfe4.json')['nfe_results']['4']['rows']
            diagnostic=read(folder/'mechanisms64/summary.json')
            if len(main)!=900 or diagnostic['frames']!=64 or diagnostic['status']!='COMPLETE':
                raise RuntimeError('Full original quality/diagnostic artifacts required')
        atomic_json(self.folder/'P0_gate.json',dict(status='QUALITY_COMPLETE',
            quality_reference=str(previous),original_full_data_audit='PASS',
            branch_activity='DEFERRED; not claimed complete',exclusive_timing='DEFERRED',
            authority='User requests direct training before exclusive timing'))
        # No artifact from the old source identity is relabeled or reused as a new GPU smoke.
        data=self.folder/'data_access_audit.json'
        if not data.exists():
            old=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-encoder-adaptation-v1/experiments/desescene_encoder_adaptation_v1/data_access_audit.json')
            audit=read(old)
            if audit['status']!='PASS' or not audit['shard_hashes_verified'] or audit['missing_original_inputs']:
                raise RuntimeError('Original registered data audit missing')
            shutil.copy2(old,data)
        self.status('P1_ENGINEERING_THEN_TRAINING_SHARED', exclusive_timing='DEFERRED')
        def worker(arm,gpu):
            with self.lease(gpu):
                self.engineering(arm,gpu)
                self.train(arm,gpu)
                self.evaluate(arm,gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending=[pool.submit(worker,arm,gpu) for arm,gpu in [('EF1',2),('EF2',3)]]
            errors=[]
            for job in pending:
                try: job.result()
                except Exception as exc: errors.append(f'{type(exc).__name__}: {exc}')
        if errors: raise RuntimeError('; '.join(errors))
        with self.lease(2):
            self.evaluate('EF0',2,reference=True)
            out=self.folder/'P0/audit'
            self.run('P0_audit',2,'complete_local_state_audit.py',
                ['--p0-root',P0,'--baseline-root',__import__('run_encoder_adaptation_phases').B0,
                 '--output',out],out/'summary.json')
        self.status('P1_QUALITY_COMPLETE_EXCLUSIVE_TIMING_DEFERRED',
                    p2_formal_training_started=False, p3_formal_training_started=False)


def main():
    folder=ROOT/'experiments/desescene_encoder_adaptation_v1'
    folder.mkdir(parents=True,exist_ok=True)
    # Hold the original coordinator's lock for this full lifetime, too: no duplicate continuation.
    legacy=Path('/data-12/M2024-HWZ/FPSGen/worktrees/desescene-encoder-adaptation-v1/experiments/desescene_encoder_adaptation_v1/supervisor.lock').open('a')
    fcntl.flock(legacy,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue=SharedQueue(folder)
    try: queue.execute()
    except Exception as exc:
        queue.status('BLOCKED_REVIEW_REQUIRED',reason=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__': main()
