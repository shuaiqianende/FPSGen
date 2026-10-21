"""Dedicated launcher for registered Stage1 dense-fidelity architecture pilots."""
import argparse
from pathlib import Path
import yaml

from fpsgen.hier_occ_flow.stage1_training import run_stage1, validate_config


def load_config(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg['geometry'] = yaml.safe_load((path.parent / cfg['geometry']).read_text())
    validate_config(cfg)
    if cfg.get('scope') != 'architecture_fidelity_pilot':
        raise ValueError('This entry point accepts only architecture_fidelity_pilot configs')
    return cfg


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_stage1(load_config(args.config), args.output, resume=args.resume,
               max_updates=args.max_updates, smoke=args.smoke)


if __name__ == '__main__':
    main()
