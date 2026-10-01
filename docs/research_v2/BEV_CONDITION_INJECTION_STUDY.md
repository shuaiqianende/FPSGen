# Stage-1 BEV condition-injection study

This document defines the reproducible campaign for measuring whether the
Stage-1 BEV velocity backbones use LiDAR, vehicle and road conditions. It is a
condition-injection study, not a new-backbone comparison: HDiT-S, DiP-S/16 and
NCSNpp-S keep their established cores and PointPillar frontend.

## Fixed contract

- Objective: direct Flow Matching velocity prediction on the native
  `[density, maximum-height, occupancy]` BEV target, weighted `[1, 2, 1]`.
- Data: `gt_possion`, SemanticKITTI train sequences `00–07,09,10`, 180,000
  points, and the existing eight uniformly sampled condition states.
- Training: eager FP32, no compilation, batch 8, learning rate `1e-4`, seed
  42, five epochs and one checkpoint per epoch. GPU2 and GPU3 are the only
  campaign devices; GPU1's pre-existing NCSNpp C0 run is left untouched.
- Compatibility: C0 remains the already completed baseline. New experiments
  use `model.condition.api_version: 2`; old C0 checkpoints retain their
  original state-dict contract.

## Condition API and ablations

Every formal config is materialized from
`configs/research_v2/condition_ablation/phase1.yaml` and exposes the common
API below. All projections are bias-free so an inactive all-zero condition
stays exactly zero. All injection gates start at `0.1`.

```yaml
model:
  condition:
    spatial: true
    global: true
    fusion: shared       # shared | separate
    native: false
    gate_init: 0.1
    bias: false
```

| ID | Variant | Spatial | Global | Fusion | Native placement |
| --- | --- | :---: | :---: | --- | --- |
| C0 | existing baseline | existing | existing | existing | none |
| C1 | `spatial` | yes | no | shared | none |
| C2 | `global` | no | yes | shared | none |
| C3 | `separate` | yes | yes | separate first projection | none |
| C4 | `native` | yes | yes | shared | backbone-specific |

Native C4 placement is deliberately minimal: HDiT receives multi-level
mapping modulation, DiP supplies Local Detailer condition maps, and NCSNpp
supplies level-wise time modulation. The coordinator checks that adapter
parameters stay under 5% of each core before a run starts.

## Guardrails and measurements

For every C1–C4 run, `scripts/run_bev_condition_campaign.py` performs a
three-step batch-1 smoke, then verifies finite loss and nonzero/finite
condition gradients with formal batch 8 before starting the five-epoch run.
The derivative probe has one in-memory optimizer warmup step: this is needed
for DiP's zero-initialized AdaLN output to expose its otherwise valid global
condition derivative, and never changes the saved formal run. Workers are
restartable and skip a variant only when its epoch-04 checkpoint, B100
condition-use JSON, and B20 summary all exist. The Lightning module records
gate values and condition feature norms every 100 steps. For C3 it additionally
records independent `lidar_norm`, `vehicle_norm` and `road_norm` before the
first shared fusion layer. A
CUDA-synchronized profiler records step and data-ready timing every 50 steps.
Its bottleneck rule is median data gap greater than 20% of median step time or
p95 gap greater than twice the median gap; this is evidence for tune-first,
not for a large cache/preload change.

After each epoch-04 checkpoint, the coordinator runs:

- `scripts/eval_bev_condition_usage.py`: sequence 08, B100, seed 20261001,
  times `0.1,0.3,0.5,0.7,0.9`, all eight condition modes, correct/zero/cyclic
  shuffle conditions, and `Gzero`, `Gshuffle`, `Δv`.
- `scripts/eval_bevflow_lidar_only.py`: the fixed B20 screen evaluation with
  one sample/frame and no visual artifacts.

## Reproducing and reporting

Start one worker per permitted physical GPU from the repository root:

```bash
tmux new-session -d -s bev_cond_gpu2 'cd /data-12/M2024-HWZ/FPSGen && source env/venv_pt20_cu117/bin/activate && exec python scripts/run_bev_condition_campaign.py --gpu 2'
tmux new-session -d -s bev_cond_gpu3 'cd /data-12/M2024-HWZ/FPSGen && source env/venv_pt20_cu117/bin/activate && exec python scripts/run_bev_condition_campaign.py --gpu 3'
```

The campaign does not select checkpoints by validation. Once artifacts are
available, generate an auditable live table with:

```bash
source env/venv_pt20_cu117/bin/activate
python scripts/summarize_bev_condition_campaign.py
```

It writes `outputs/condition_campaign/phase1_summary.{json,md}`. Missing
checkpoints, probes or evaluations render as `—`, so an unfinished run cannot
silently enter ranking.

`scripts/run_bev_condition_coordinator.py` is started once in its own tmux
session. It owns no CUDA device while waiting for the C1–C4 barrier. At the
barrier it reruns every C0–C4 B20 screen with the fixed generation seed
`20261001`, writes `phase1_summary.csv` and `phase1_ranking.json`, and ranks
only within each backbone. The score uses final sampled 500-step FM loss,
condition-use gains and the four screen metrics with the predeclared weights.
Runs with both `Gshuffle_100` and `Gshuffle_111` at or below `0.01` are marked
`condition_insensitive`.

The coordinator then trains each backbone's two ranked policies from scratch
at seed 123 for eight epochs, using one serial process on GPU2 and one on
GPU3. Every candidate receives condition-usage B100 and LiDAR-only B100 with
three generation seeds, both at seed 20261001. It emits
`phase2_summary.csv`, `seed_stability.csv` and
`final_condition_comparison.csv`; it deliberately stops there and never
starts a long formal run. The historical C0 checkpoint remains immutable for
Phase-1; if C0 is selected for Phase-2, it uses the same shared spatial/global
policy through the v2 compact, bias-free adapter so it passes the common <5%
adapter/core fairness audit.

## 2026-10-01 interim result: NCSNpp-S versus original FPSGenBEV

This is a loss-only interim comparison, recorded while the NCSNpp-S Spatial
and Global Phase-1 runs are still in progress. Of the completed new
backbone-policy pairs, NCSNpp-S with C3 `separate` is the current candidate:
it has the lowest completed epoch-04 sampled loss (`0.1168`) and a nonzero
condition-use response (`Gshuffle_100=0.1816`, `Gshuffle_111=0.2014`). It is
not a final policy selection; the missing C1/C2 results may change the
within-NCSNpp ranking.

The reference is the completed historical checkpoint
`bev_legacy_gt_possion_gpu0_bs8_5ep_epoch=04.ckpt` (the original FPSGenBEV
BEVFlowTransNet). Both rows below use `gt_possion`, the same ten train
sequences, 180,000 points, batch 8, the `[1,2,1]` flow-matching loss, learning
rate `1e-4`, and five epochs. They differ in backbone and condition adapter,
so this table establishes optimization behaviour only; it is not a causal
attribution of any difference to a single architectural choice.

Each number is the mean of scalar `train/loss_mse` samples in the stated
500-optimizer-step interval, read directly from the two TensorBoard event
files. The final partial window contains steps 11,500--11,899.

| Steps | Original FPSGenBEV | NCSNpp-S / C3 Separate |
| --- | ---: | ---: |
| 0--499 | 1.6711 | 1.0514 |
| 500--999 | 1.0084 | 0.2071 |
| 1,000--1,499 | 0.3966 | 0.1660 |
| 1,500--1,999 | 0.2352 | 0.1709 |
| 2,000--2,499 | 0.2256 | 0.1803 |
| 2,500--2,999 | 0.1900 | 0.1567 |
| 3,000--3,499 | 0.1775 | 0.1630 |
| 3,500--3,999 | 0.1480 | 0.1298 |
| 4,000--4,499 | 0.1749 | 0.1322 |
| 4,500--4,999 | 0.1273 | 0.1252 |
| 5,000--5,499 | 0.1781 | 0.1467 |
| 5,500--5,999 | 0.1541 | 0.1420 |
| 6,000--6,499 | 0.1314 | 0.1151 |
| 6,500--6,999 | 0.1491 | 0.1312 |
| 7,000--7,499 | 0.1149 | 0.1188 |
| 7,500--7,999 | 0.1196 | 0.1228 |
| 8,000--8,499 | 0.1345 | 0.1318 |
| 8,500--8,999 | 0.1215 | 0.1198 |
| 9,000--9,499 | 0.1433 | 0.1232 |
| 9,500--9,999 | 0.1161 | 0.1113 |
| 10,000--10,499 | 0.1165 | 0.1062 |
| 10,500--10,999 | 0.1462 | 0.1252 |
| 11,000--11,499 | 0.1253 | 0.1115 |
| 11,500--11,899 | 0.1447 | 0.1329 |

| Model | Epoch 0 | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Last 2,000 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FPSGenBEV | 0.7533 | 0.1698 | 0.1473 | 0.1279 | 0.1291 | 0.1326 |
| NCSNpp-S / C3 Separate | **0.3737** | **0.1427** | **0.1322** | **0.1241** | **0.1168** | **0.1182** |

NCSNpp-S therefore reaches the low-loss regime much earlier and has a lower
late-window loss in this matched budget. The difference is modest after the
first epoch, so it should not be interpreted as a finished quality claim.
The pending fair quality comparison runs `eval_bevflow_lidar_only.py` for the
original checkpoint on the exact C3 B20 manifest, one sample per frame,
`base_seed=20261001`, `save_visuals=0`, and reports density/height/occupancy
metrics beside the existing NCSNpp-S C3 result. It is intentionally deferred
until GPU2 or GPU3 becomes available, rather than competing with the active
Phase-1 training jobs.
