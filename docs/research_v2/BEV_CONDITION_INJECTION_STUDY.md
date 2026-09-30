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
three-step batch-1 smoke, verifies finite loss and nonzero/finite condition
gradients with formal batch 8, then starts the five-epoch run. The Lightning
module records gate values and condition feature norms every 100 steps. For
C3 it additionally records independent `lidar_norm`, `vehicle_norm` and
`road_norm` before the first shared fusion layer. A
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
starts a long formal run.
