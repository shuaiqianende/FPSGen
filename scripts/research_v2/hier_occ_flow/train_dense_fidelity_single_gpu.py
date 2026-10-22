"""Run a registered one-GPU, batch-2, 20K-exposure R0/E3 matched pair."""
import argparse
from pathlib import Path
import yaml

from fpsgen.hier_occ_flow.stage1_training import SINGLE_GPU_20K_SCOPE, run_stage1, validate_config


def load_config(path: Path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg['geometry'] = yaml.safe_load((path.parent / cfg['geometry']).read_text())
    validate_config(cfg)
    if cfg.get('scope') != SINGLE_GPU_20K_SCOPE:
        raise ValueError('This entry point accepts only registered single-GPU 20K configs')
    return cfg


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-updates', type=int)
    args = parser.parse_args()
    run_stage1(load_config(args.config), args.output, max_updates=args.max_updates, smoke=args.smoke)


if __name__ == '__main__':
    main()
