import copy

import pytest

from fpsgen.hier_occ_flow.stage1_curve_comparison import compare_stage1_curve_runs


def _run(offset=0):
    rows = []
    for frame in (10, 20):
        curves = []
        for threshold, tp in ((0.25, 8 + offset), (0.5, 5 + offset)):
            active = 10
            curves.append({
                "threshold": threshold, "tp": tp, "fp": active - tp,
                "fn": 10 - tp, "precision": tp / active,
                "recall": tp / 10, "f1": 2 * tp / (20 + active - 2 * tp),
                "iou": tp / 20, "active_parents": active,
                "fine_recall_ceiling": tp / 10,
            })
        fixed = [{
            "requested_k": 10, "actual_k": 10, "tp": 6 + offset,
            "fp": 4 - offset, "fn": 4 - offset,
            "precision": (6 + offset) / 10,
            "recall": (6 + offset) / 10,
            "f1": (6 + offset) / 10, "iou": (6 + offset) / 14,
            "fine_recall_ceiling": (6 + offset) / 10,
        }]
        rows.append({
            "sequence": "08", "frame": frame,
            "threshold_curve": curves, "fixed_budget": fixed,
        })
    return {
        "status": "COMPLETE", "development_manifest_sha256": "dev",
        "source_manifest_sha256": "source", "nfe": 2,
        "frozen_threshold": 0.5, "thresholds": [0.25, 0.5],
        "fixed_candidate_budgets": [10], "rows": rows,
    }


def test_paired_curve_summary_and_deltas():
    report = compare_stage1_curve_runs({"R0": _run(), "R1": _run(1)})
    assert report["frames"] == 2
    assert report["summaries"]["R0"]["fixed_budget"]["10"]["macro"]["recall"] == 0.6
    assert report["paired_frame_deltas"]["R1-minus-R0"][0]["k10_recall_delta"] == pytest.approx(0.1)


def test_comparison_rejects_mismatched_manifest_and_frames():
    a, b = _run(), _run()
    b["development_manifest_sha256"] = "other"
    with pytest.raises(ValueError, match="development_manifest_sha256"):
        compare_stage1_curve_runs({"R0": a, "R1": b})
    b = _run()
    b["rows"].pop()
    with pytest.raises(ValueError, match="frame manifests differ"):
        compare_stage1_curve_runs({"R0": a, "R1": b})


def test_comparison_rejects_duplicate_frame_rows():
    a, b = _run(), _run()
    b["rows"].append(copy.deepcopy(b["rows"][0]))
    with pytest.raises(ValueError, match="duplicate frame IDs"):
        compare_stage1_curve_runs({"R0": a, "R1": b})
