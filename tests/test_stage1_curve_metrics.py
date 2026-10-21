import numpy as np
import pytest
from types import SimpleNamespace

from fpsgen.hier_occ_flow.stage1_evaluation import parent_metrics
from fpsgen.hier_occ_flow.stage1_curve_metrics import (
    fixed_budget_metrics, summarize_fixed_budget, threshold_curve_metrics,
)


@pytest.fixture
def sample():
    scores = np.array([0.1, 0.8, 0.6, 0.4, 0.8, 0.3, 0.8, 0.0], dtype=np.float32)
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    record = {
        'parent_ids': np.array([1, 4, 6], dtype=np.int64),
        'child_occ_mask': np.array([0b11, 0b01, 0b111], dtype=np.uint8),
    }
    return scores, record, valid


def test_strict_threshold_metrics_and_fine_ceiling(sample):
    scores, record, valid = sample
    rows = threshold_curve_metrics(scores, record, valid, [0.5, 0.8])
    assert rows[0]['active_parents'] == 4
    assert rows[0]['tp'] == 3 and rows[0]['fp'] == 1 and rows[0]['fn'] == 0
    assert rows[0]['precision'] == pytest.approx(0.75)
    assert rows[0]['recall'] == pytest.approx(1.0)
    assert rows[0]['fine_recall_ceiling'] == pytest.approx(1.0)
    # Candidate selection uses score > threshold, so values tied at 0.8 are excluded.
    assert rows[1]['active_parents'] == 0
    assert rows[1]['tp'] == 0 and rows[1]['fine_recall_ceiling'] == 0.0


def test_fixed_k_ties_are_deterministic_by_flat_parent_id(sample):
    scores, record, valid = sample
    first = fixed_budget_metrics(scores, record, valid, [2, 3])
    second = fixed_budget_metrics(scores, record, valid, [2, 3])
    assert first == second
    assert first[0]['actual_k'] == 2
    assert first[0]['tp'] == 2
    assert first[0]['precision'] == pytest.approx(1.0)
    assert first[0]['recall'] == pytest.approx(2 / 3)
    assert first[0]['fine_recall_ceiling'] == pytest.approx(0.5)
    assert first[1]['actual_k'] == 3
    assert first[1]['tp'] == 3
    assert first[1]['recall'] == pytest.approx(1.0)


def test_fixed_budget_caps_to_valid_candidates_and_summarizes(sample):
    scores, record, valid = sample
    row = fixed_budget_metrics(scores, record, valid, [100])[0]
    assert row['actual_k'] == int(valid.sum())
    summary = summarize_fixed_budget([row])
    assert summary['frames'] == 1
    assert summary['macro']['recall'] == pytest.approx(1.0)
    assert summary['micro']['recall'] == pytest.approx(1.0)


def test_rejects_invalid_scores_and_budgets(sample):
    scores, record, valid = sample
    bad_scores = scores.copy()
    bad_scores[2] = np.nan
    with pytest.raises(ValueError, match='finite'):
        threshold_curve_metrics(bad_scores, record, valid, [0.5])
    with pytest.raises(ValueError, match='positive'):
        fixed_budget_metrics(scores, record, valid, [0])


def test_threshold_curve_matches_existing_metric_definition():
    rng = np.random.default_rng(4)
    scores = rng.random(40, dtype=np.float32)
    valid = rng.random(40) > 0.2
    parent_ids = np.sort(rng.choice(np.flatnonzero(valid), 8, replace=False))
    record = {
        'parent_ids': parent_ids,
        'child_occ_mask': rng.integers(0, 256, size=len(parent_ids), dtype=np.uint8),
    }
    curve = threshold_curve_metrics(scores, record, valid, [0.1, 0.5, 0.8])
    for row in curve:
        reference = parent_metrics(scores, record, SimpleNamespace(k=8), valid, row['threshold'])
        for key in ('tp', 'fp', 'fn', 'active_parents', 'precision', 'recall', 'iou', 'fine_recall_ceiling'):
            assert row[key] == pytest.approx(reference[key])
