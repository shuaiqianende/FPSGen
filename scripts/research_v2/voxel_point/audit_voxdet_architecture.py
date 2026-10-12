"""Count original-config and adapted LiDAR modules, retaining all trunk layers."""

import ast
from pathlib import Path

from common import parser, read_config, save_json, ROOT
from fpsgen.voxel_point.voxdet_density import VoxDetDensity


def main():
    args = parser(__doc__).parse_args()
    cfg, spec, sha = read_config(args.config)
    # Same constructors as official config, including both 3D backbones.
    original = VoxDetDensity(spec, output_channels=20)
    adapted = VoxDetDensity(spec, **cfg["model"])
    count = lambda module: sum(p.numel() for p in module.parameters())
    modules = {name:{"original":count(getattr(original,name)),"adapted":count(getattr(adapted,name))}
               for name in ('lidar_tokenizer','voxel_backbone','voxel_neck','local_backbone','local_neck','main_head','aux_head')}
    report = {"config_sha256":sha,"upstream_commit":"e10d8280de96d4f03f54b616e57e7c6754044fa4",
        "original_parameters":count(original),"adapted_parameters":count(adapted),"modules":modules,
        "output_shape":[1,1,*spec.shape],"output_shape_note":"contract; forward gate recorded separately",
        "tokenizer_preserved":True,"backbone_preserved":True,"neck_preserved":True,
        "spatially_decoupled_encoder_preserved":True,"six_direction_refinement_layers":4,
        "input_feature_adaptation":"relative XYZ + absolute XY + norm(XYZ)/50; no intensity or labels",
        "coarse_grid":[n//2 for n in spec.shape],"coarse_cell_size":[s*2 for s in spec.size],
        "compatibility_patches":["remove registry/semantic-loss imports", "flatten parameterless Ident/VoxelAggregatorDual wrappers",
            "correct grid_sample ZYX coordinates and align_corners=False normalization"],
        "task_adaptations":["20 → 1 output channels in main/aux heads", "negative trainable final density bias",
            "FP32 softplus/count losses", "density-derived six-direction occupied-run supervision",
            "open circle/Z ROI", "no GT refinement / camera / semantic supervision"]}
    for name in ('lidar_tokenizer','voxel_backbone','voxel_neck','local_backbone','local_neck'):
        assert modules[name]['original'] == modules[name]['adapted']
    save_json(args.output/'architecture_audit.json',report)
    print(f"Official-config params {count(original):,}; density params {count(adapted):,}")


if __name__ == '__main__':
    main()
