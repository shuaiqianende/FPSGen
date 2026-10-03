# Research V2 index

This is the topic index. Study files retain their recorded configurations,
conclusions and dated results; live job status must be read from local logs.

| Topic | Primary records |
| --- | --- |
| Data contract / Poisson GT | [Training contract](GT_POSSSION_TRAINING_CONTRACT.md), [GT generation](GT_POISSON_GENERATION.md) |
| Teacher / endpoint refinement | [DCD teacher](DCD_TEACHER_5EPOCH_ANALYSIS.md), [Sinkhorn analysis](SINKHORN_ENDPOINT_ANALYSIS.md), [iteration ablation](SINKHORN_ITERATION_ABLATION.md) |
| Feature guidance | [Analysis](FEATURE_GUIDANCE_ANALYSIS.md) |
| DiC / PixelU | [Architecture](BEV_BACKBONE_DIC_PIXELU.md), [training dynamics](BEV_BACKBONE_TRAINING_DYNAMICS.md) |
| HDiT / DiP / NCSN++ | [Backbones](BEV_BACKBONE_HDIT_DIP_NCSNPP.md), [condition study](BEV_CONDITION_INJECTION_STUDY.md) |
| SiD2 / PixelDiT | [Implementation](BEV_BACKBONE_SID2_PIXELDIT.md), [dynamics](BEV_BACKBONE_SID2_PIXELDIT_DYNAMICS.md) |
| Spatial control | [Source audit](SPATIAL_CONTROL_SOURCE_AUDIT.md), [design](SYNFLOW_CRACKSEGFLOW_PIXELCONTROL.md), [short runs](SPATIAL_CONTROL_SHORT_RUN_RESULTS.md), [statistics](SPATIAL_CONTROL_STATISTICAL_ANALYSIS.md) |
| NCSN++ SPADE | [Study](NCSNPP_SPADE_STUDY.md) |
| Legacy training performance | [Speed audit](BEVFLOW_TRAINING_SPEED_AUDIT.md), [packed shard pilot](BEVFLOW_PACKED_SHARD_PILOT.md) |
| Formal BEV method comparison | [Protocol, architecture corrections and mandatory GPU23 gates](FORMAL_BEV_METHOD_COMPARISON.md) |
| BEV-only evaluation | [Protocol](BEVFLOW_LIDAR_ONLY_EVALUATION.md) |
| Original versus current checkpoints | [PointFlow screen and seeded crossover](ORIGINAL_FPSGEN_CHECKPOINT_COMPARISON.md) |
| PointFlow XYZ visualization | [Source / actual trajectories / endpoint figure](POINTFLOW_XYZ_TRAJECTORY_VISUALIZATION.md) |
| Real PointFlow / LiDiff / LiDPM / ScoreLiDAR / DistillationDPO trajectories | [Baseline integration, source audit and measured paths](COMPLETION_BASELINE_TRAJECTORIES.md) |
| Source provenance | [Backbone source manifest](backbone_sources.yaml) |
| Historical progress | [Dated records](progress/README.md), [original project snapshot](PROJECT_INDEX.md) |

New studies should record the question, fixed controls, source provenance,
commands/configs, acceptance tests, result artifacts and limitations. Summaries
must distinguish training loss, condition usage and generation quality. Add a
link here when a study is ready; keep historical documents at their existing paths.
