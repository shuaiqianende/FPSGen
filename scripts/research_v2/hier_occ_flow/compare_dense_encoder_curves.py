"""Compare completed R0/R1/R2 PR and fixed-K curve evaluation JSON files."""
import argparse
import json
from pathlib import Path

from fpsgen.hier_occ_flow.stage1_curve_comparison import compare_stage1_curve_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--r2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"R0 Original": args.r0, "R1 Bottleneck": args.r1,
             "R2 OMD-3D": args.r2}
    runs = {name: json.loads(path.read_text()) for name, path in paths.items()}
    result = compare_stage1_curve_runs(runs)
    result["input_files"] = {name: str(path) for name, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({result['frames']} paired frames)")


if __name__ == "__main__":
    main()
