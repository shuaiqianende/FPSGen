"""Audit strict real GT, save fixed manifests and per-scene density statistics."""

from common import parser, read_config, save_json
from fpsgen.voxel_point.dataset import paired_frames, development_split, VoxelPointDataset, density_statistics, manifest_sha


def main():
    args = parser(__doc__).parse_args()
    cfg, spec, sha = read_config(args.config)
    data = cfg["data"]
    frames = paired_frames(data["root"], data["train_sequences"])
    manifest = development_split(frames, block_size=data["block_size"],
        guard_frames=data["guard_frames"], development_block=data["development_block"])
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "split_manifest.json", manifest)
    # Evenly spaced diagnostics across sequences; do not silently skip failures.
    selected = frames
    if args.max_frames:
        import numpy as np
        selected = [frames[i] for i in np.linspace(0, len(frames)-1, min(args.max_frames,len(frames)), dtype=int)]
    dataset = VoxelPointDataset(selected, spec)
    rows, anomalies = [], []
    for index, frame in enumerate(selected):
        try:
            sample = dataset[index]
            rows.append({"sequence": frame["sequence"], "frame": frame["frame"],
                         "input_points": len(sample["input_xyz"]),
                         **density_statistics(sample["gt_counts"], spec)})
        except ValueError as exc:
            anomalies.append({"frame": frame, "reason": str(exc)})
    report = {"config_sha256": sha, "manifest_sha256": manifest_sha(frames),
              "split_sizes": {s:len(manifest[s]) for s in ("train","development","excluded_guard")},
              "audited_frames":len(selected), "accepted":len(rows), "anomalies":anomalies,
              "statistics":rows, "strict_gt_contract_pass":not anomalies}
    save_json(args.output / "dataset_audit.json", report)
    print(f"Audited {len(selected)} real frames; {len(anomalies)} rejected")
    if anomalies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
