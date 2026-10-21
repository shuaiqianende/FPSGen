"""Pair and summarize frozen Stage1 PR/fixed-budget curve outputs."""
from collections import defaultdict


_METRICS = (
    "precision", "recall", "f1", "iou", "active_parents",
    "fine_recall_ceiling",
)
_FIXED_METRICS = ("precision", "recall", "f1", "iou",
                  "fine_recall_ceiling", "actual_k")


def _frame_key(row):
    return str(row["sequence"]), int(row["frame"])


def _aggregate(rows):
    if not rows:
        raise ValueError("cannot summarize an empty curve")
    result = {
        "frames": len(rows),
        "macro": {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in _METRICS if key in rows[0]
        },
        "micro": {},
    }
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    result["micro"] = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }
    return result


def compare_stage1_curve_runs(runs):
    """Validate and pair ``{variant: eval_json}`` outputs from the curve evaluator.

    The returned report carries full macro/micro PR summaries, fixed-K
    summaries, and per-frame deltas at the frozen threshold and every K.
    """
    if len(runs) < 2:
        raise ValueError("at least two variants are required")
    names = sorted(runs)
    first = runs[names[0]]
    contract_keys = (
        "development_manifest_sha256", "source_manifest_sha256", "nfe",
        "frozen_threshold", "thresholds", "fixed_candidate_budgets",
    )
    for name in names:
        run = runs[name]
        if run.get("status") != "COMPLETE":
            raise ValueError(f"{name}: curve run is not COMPLETE")
        for key in contract_keys:
            if run.get(key) != first.get(key):
                raise ValueError(f"{name}: mismatched {key}")

    indexed = {}
    for name, run in runs.items():
        rows = run.get("rows", [])
        by_key = {_frame_key(row): row for row in rows}
        if len(by_key) != len(rows):
            raise ValueError(f"{name}: duplicate frame IDs")
        indexed[name] = by_key
    expected = set(indexed[names[0]])
    if not expected or any(set(indexed[name]) != expected for name in names[1:]):
        raise ValueError("variant frame manifests differ")

    thresholds = [float(value) for value in first["thresholds"]]
    budgets = [int(value) for value in first["fixed_candidate_budgets"]]
    by_variant = {}
    normalized = {}
    for name in names:
        row_by_threshold = defaultdict(list)
        row_by_budget = defaultdict(list)
        normalized[name] = {}
        for key in sorted(expected):
            row = indexed[name][key]
            curves = {float(item["threshold"]): item
                      for item in row["threshold_curve"]}
            fixed = {int(item["requested_k"]): item
                     for item in row["fixed_budget"]}
            if set(curves) != set(thresholds) or set(fixed) != set(budgets):
                raise ValueError(f"{name} {key}: curve points or budgets differ")
            normalized[name][key] = {"threshold": curves, "fixed": fixed}
            for threshold, item in curves.items():
                row_by_threshold[threshold].append(item)
            for budget, item in fixed.items():
                row_by_budget[budget].append(item)
        by_variant[name] = {
            "threshold_pr": {
                f"{threshold:.6f}": _aggregate(row_by_threshold[threshold])
                for threshold in thresholds
            },
            "fixed_budget": {
                str(budget): _aggregate(row_by_budget[budget])
                for budget in budgets
            },
        }

    frozen = float(first["frozen_threshold"])
    pairwise = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pair_name = f"{right}-minus-{left}"
            frame_rows = []
            for key in sorted(expected):
                a = normalized[left][key]
                b = normalized[right][key]
                row = {"sequence": key[0], "frame": key[1]}
                for metric in _METRICS:
                    row[f"threshold_{metric}_delta"] = (
                        float(b["threshold"][frozen][metric])
                        - float(a["threshold"][frozen][metric])
                    )
                for budget in budgets:
                    for metric in _FIXED_METRICS:
                        row[f"k{budget}_{metric}_delta"] = (
                            float(b["fixed"][budget][metric])
                            - float(a["fixed"][budget][metric])
                        )
                frame_rows.append(row)
            pairwise[pair_name] = frame_rows

    return {
        "status": "COMPLETE",
        "variants": names,
        "frames": len(expected),
        "frame_ids": [
            {"sequence": sequence, "frame": frame}
            for sequence, frame in sorted(expected)
        ],
        "protocol": {key: first[key] for key in contract_keys},
        "summaries": by_variant,
        "paired_frame_deltas": pairwise,
    }
