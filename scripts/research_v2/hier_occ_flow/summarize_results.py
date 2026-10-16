"""Publish small measured audit reports; never promote smoke to pilot/formal results."""
from common import ROOT
import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from fpsgen.hier_occ_flow.storage import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--run-root',type=Path,default=Path('experiments/hier_occ_flow'))
    args=p.parse_args(); run=args.run_root; docs=ROOT/'docs/research_v2'; out=docs/'hs_occflow_reports'; out.mkdir(parents=True,exist_ok=True)
    copies={'resolution_audit16/resolution_audit.json':'resolution_audit.json',
            'resolution_audit16/raw_source_manifest.json':'raw_source_manifest.json',
            'io_256_18_final.json':'io_256_18.json','io_352_8_final.json':'io_352_8.json',
            'candidate_audit_dev00_500.json':'candidate_audit_dev00_500.json','model_audit.json':'model_audit.json',
            'gpu2_preflight.json':'gpu2_preflight.json','gpu3_preflight.json':'gpu3_preflight.json',
            'cpu_real_resume_audit.json':'cpu_real_resume_audit.json',
            'cpu_local_smoke32_r2/validation/update_000032.json':'cpu_local_smoke32_development2.json'}
    for source,target in copies.items(): shutil.copyfile(run/source,out/target)
    data=json.loads((out/'resolution_audit.json').read_text()); res=data['summary']
    cpu=json.loads((out/'cpu_real_resume_audit.json').read_text()); models=json.loads((out/'model_audit.json').read_text())
    suites=ET.parse(run/'tests_cpu.xml').getroot()
    totals={key:sum(int(suite.get(key,0)) for suite in suites.iter('testsuite')) for key in ('tests','failures','errors','skipped')}
    totals['passed']=totals['tests']-totals['failures']-totals['errors']-totals['skipped']
    atomic_json(out/'cpu_tests.json',totals)
    io={name:json.loads((out/f'io_{name}.json').read_text()) for name in ('256_18','352_8')}
    candidate=json.loads((out/'candidate_audit_dev00_500.json').read_text())
    decision={'scope':'CPU representation preflight, not final network selection','provisional_main':'256_18',
        'provisional_ablation':'352_8','geometry_locked':False,'geometry_sha256':res['256_18']['geometry_sha256'],
        'formal_ready':False,'reason':'Similar/finer quantization with 38.5% fewer median parent tokens vs 352+8 and reuse of existing 256 Stage1; GPU quality/memory/latency and matched BCE/FM pilots remain unmeasured'}
    atomic_json(out/'geometry_decision.json',decision)
    gate={'cpu_geometry_data_tests':'PASS','raw_source_reference_exact':data['reference']['exact_array_equal'],
        'real_cpu_resume':cpu['status'],'full_scene_cpu_local_fm_successful_updates':32,
        'gpu_admission':'BLOCKED_BY_EXISTING_STUDENT_RUN','ptv3_gpu_forward_backward':'NOT_EXECUTED',
        'gpu_cpu_voxelization_consistency':'NOT_EXECUTED','full_dataset_preprocessed':False,
        'processed_frames':16,'train_frames_processed':8,'development_frames_processed':8,
        'ptv3_bce_overfit_updates':0,'ptv3_fm_overfit_updates':0,'matched_1000_update_pilots':False,
        'geometry_sha256':res['256_18']['geometry_sha256'],'geometry_locked':False,'formal_ready':False}
    atomic_json(out/'engineering_gate.json',gate)
    table='| Setting | Parent tokens P50 / sample max | Packed MB/frame | Est. train GB | Quantization P90 (cm) |\n|---|---:|---:|---:|---:|\n'
    for name,row in res.items():
        table+=f"| {name} | {row['sample_token_quantiles']['p50']:,.0f} / {row['sample_token_quantiles']['p100']:,.0f} | {row['mean_packed_bytes']/1e6:.3f} | {row['estimated_train_label_bytes']/1e9:.2f} | {row['mean_point_quantization_p90_m']*100:.2f} |\n"
    docs.joinpath('HS_OCCFLOW_DATA_AUDIT.md').write_text(f'''# HS-OccFlow data audit — measured CPU milestone

Branch: `research/hs-occflow`, base `612cc976cc7f632eb1e1e1e3a765ff546fcf1acc`.
Existing worktree modifications, checkpoints and ongoing jobs were preserved.

The frozen numeric frame pairing/time-block protocol contains 17,870 train,
900 development and 360 guard frames, with no seq08. This milestone processed
**16 real frames only**: sequences00/05/09/10, frames0/200(train),500/550(dev).
It does not characterize population P99 or the true largest training scene.

GT comes from `map_clean.npy` plus calibrated poses and raw scans, using the
existing `_full_candidate()` before voxel downsampling, Poisson, or fixed-count
sampling. A cKDTree crops the map first and preserves row order. Frame00/000000
was checked against a complete-map reference scan: **1,322,038 rows exactly equal**.
Raw candidates across this cohort range from564,780 to1,624,505 points.
Map, pose, calibration, raw-scan and source-script SHA values are archived in
[raw source manifest](hs_occflow_reports/raw_source_manifest.json).
Canonical source manifest SHA: `{data['source_manifest_sha256']}`.

This is **surface-hit occupancy**. Missing map points do not prove free space.
The original map construction's exact0.1m provenance is not established here.
Randomly retaining90% of raw candidates preserves95.39% of native fine support
on256+18 (mean over16 frames). This sensitivity must be considered in evaluation.
Raw-vs-historical-Poisson native support IoU is0.3391 for256+18 and0.3720 for352+8;
historical180k labels are therefore not substituted for this task's raw targets.

Labels are sorted uint32 parent IDs plus uint8(K8) or uint32(K18) masks.
No dense fine grid, full parent XYZ, float occupancy, or raw candidate cache is
persisted. Versioned index files and shard hashes are committed by a single
atomic manifest replacement. Explicit resume verifies source/geometry/version
and checksums; orphan partial shards are ignored/replaced. The tests cover
resume, corruption and parent/child bit ordering.

{table}
Capacity estimates multiply **sample mean** label bytes by17,870; they exclude
input LiDAR, optional feature caches and checkpoints. Only this small cohort
exists locally. Full preprocessing is gated on a locked geometry decision.

See [all64 frame/geometry audit rows](hs_occflow_reports/resolution_audit.json).
''')
    docs.joinpath('HS_OCCFLOW_RESOLUTION_AUDIT.md').write_text(f'''# Resolution and resource preflight

{table}
The provisional next configuration is256+18, with352+8 retained as an ablation.
Median parent tokens decrease38.54%; dense Stage1 cells decrease47.11% vs352.
Both settings have similar15cm physical-grid center representational IoU against
the **same raw map target**:0.69585 and0.69338. This is quantization/representation
analysis, **not trained model IoU**. 256+18 and384+8 share exactly the same fine
grid768x768x64; their fine geometry statistics agree, while parent token counts differ.
Geometry is **not locked for formal experiments** until real GPU measurements
and matched BCE/FM overfit/pilot tests pass. See [decision](hs_occflow_reports/geometry_decision.json).

CPU/NumPy/Torch indexing and world-space feature sampling pass tests. CUDA indexing
parity, complete Stage1 numerical forward shapes at352/384, PTv3 CUDA permutation,
and P99/max GPU forward/backward are **not executed**. All four candidates satisfy
divisibility by8. Static XYZ pyramids for352 are352/176/88/44 with Z32/16/8/4;
they are recorded as expected shapes, never as measured forwards.

Stage1's old VoxDet internals are XYZ. The independent condition adapter explicitly
converts feature volumes to ZYX and samples normalized world coordinates with
`align_corners=False`. Oracle352 parents may sample the frozen256 encoder in world
space; a256 density field is explicitly rejected as352 predicted parent occupancy.
The old U-Net uses shape-derived down/up sampling; no historical geometry checks
were relaxed. New-resolution convolution weight compatibility is not a quality result.

GPU2/3 already have compute PIDs1294615/1294616. Admission snapshots report free
memory6,784/1,420MiB. No HS-OccFlow CUDA network was launched. The old PointFlow
Student run remains active. System root had~3.5GiB free; all new data stays on/data-12.
See [GPU2 gate](hs_occflow_reports/gpu2_preflight.json), [GPU3 gate](hs_occflow_reports/gpu3_preflight.json).

## I/O measured on the actual sparse cohort

Warm-cache128 random reads,8 indexed train frames; real int64 ID/mask payloads
are transferred through DataLoader IPC. Pin memory/GPU waiting and concurrent-rank
throughput are unmeasured. No global page cache was flushed. LZ4/Zstd are unavailable
in the reused environment, so they are explicitly omitted, not simulated.

| Geometry | Format | Workers | Warm frames/s | CPU core equivalents |\n|---|---|---:|---:|---:|\n'''+''.join(f"| {name} | {r['format']} | {r['workers']} | {r['frames_per_second']:.1f} | {r['main_and_workers_cpu_core_equivalents']:.2f} |\n" for name,d in io.items() for r in d['rows'] if r['pass']==1)+'''
The cohort fits in page cache. This is not a full-dataset cold-cache benchmark.
The0-worker mmap path is currently fastest; workers/prefetch are not assumed to
improve it. Files/index sizes are compared on matching frame IDs in the JSON reports.
GPU end-to-end data stalls remain a required gate.
''')
    docs.joinpath('HS_OCCFLOW_PILOT_RESULTS.md').write_text(f'''# Smoke and pilot status — no proposed-model quality claim

The new and legacy geometry/voxel suites pass64 tests;1 CUDA parity test is skipped.
Both real PTv3 variants construct on CPU; Stage2 construction forbids external
checkpoint loading in tests. Frozen Stage1 weights load strictly and stay frozen.
GPU forward/backward, FlashAttention runtime, AMP and DDP have **not** been tested
for HS-OccFlow because the allowed GPUs are occupied.

## CPU full-scene Local-FM mechanics smoke

This is an extra observed-mask-only ablation, width16, without PTv3 or LiDAR
pyramid. It uses8 real training frames and all occupied parents plus25% nearby
empty parents. Full scene sizes are50,452–159,628 tokens; no spatial truncation.
32 successful FP32 updates completed, gradients finite and nonzero. FM loss
{cpu['first_loss']:.6f} → {cpu['last_loss']:.6f}. Median CPU step was{cpu['cpu_step_seconds_median']:.3f}s.
The EMA development2 results at1/2/4/8NFE have native macro IoU~0.24555 and recall
~0.4993, essentially a random-like initial endpoint. **This is not useful occupancy
generation or evidence that FM beats BCE.** It does not satisfy the proposed PTv3
overfit/pilot gate and none of its weights will initialize a formal model.

Continuous32 vs fresh20+resume12 are bit-identical for model, EMA, optimizer,
scheduler, scaler state, RNG, sampler cursor, loss sequence and frame order.
Model tensor SHA: `{cpu['model_tensor_sha256']}`.
See [real resume audit](hs_occflow_reports/cpu_real_resume_audit.json).

Two engineering issues were found and corrected: disabled CPU autocast still
selected FP16 in Torch2.0; and sandbox IPC disallowed multi-worker tensor sharing.
The first failed step0 output is preserved; the corrected CPU run used a fresh
directory. I/O reruns used host IPC permissions and no GPU.

## Requested GPU experiments

| Experiment | Actual updates | Status |\n|---|---:|---|\n| Oracle PTv3+BCE8–16-frame overfit |0|blocked by GPU admission|\n| Oracle PTv3+FM matched overfit |0|blocked by GPU admission|\n| LiDAR-pyramid Local FM1K pilot |0|blocked by GPU admission|\n| Matched BCE/FM1K, then5K pilots |0|not started|\n| Formal experiment |0|not ready|\n
CPU smoke/temporary synthetic unit tests are not substituted for these experiments.
The gate remains `formal_ready=false`.
''')
    cand_table='| Raw count policy | Parents | Precision | Recall | Fine recall ceiling |\n|---|---:|---:|---:|---:|\n'
    for row in candidate['rows']:
        cand_table+=f"| {row['selection']} | {row['active_parent_count']:,} | {row['precision']:.4f} | {row['recall']:.4f} | {row['fine_occupancy_recall_ceiling']:.4f} |\n"
    docs.joinpath('HS_OCCFLOW_FINAL_RESULTS.md').write_text('''# Formal results: NOT AVAILABLE

No formal HS-OccFlow training was launched; no paper claim is supported yet.
Representation, I/O and CPU mechanics results are separate milestones. GPU smoke,
full-data preprocessing, matched BCE/FM overfit and1K/5K pilots remain required.
Geometry and candidate/endpoint thresholds are not frozen for formal comparison.

## Existing Stage1 candidate ceiling diagnostic

One development frame00/000500, existing input-only VoxDet25K density vs **new raw
map occupancy**. This is not900-frame calibration. The checkpoint was trained on
historical Poisson density; the changed target support is an explicit domain gap.

'''+cand_table+'''
At raw-count0.5,11.75% of fine GT support is unreachable by Stage2. At0.01 this
falls to5.50%, with more false-positive parents/tokens. Thresholds cannot be
selected from this one frame, and seq08 was not accessed for this decision.

Next execution order: free allowed GPU admission → actual Stage1/parent PTv3
shapes and memory/permutation → locked geometry → full versioned labels → matched
oracle BCE/FM overfit →1K pilots → only after quality gates5K/predicted-parent work.
No100K queue or Stage2 PointFlow/Teacher/Sinkhorn job has been created.
''')
    print('Reports written:',docs)

if __name__=='__main__': main()
