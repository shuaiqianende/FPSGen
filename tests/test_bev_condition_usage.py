"""Pure aggregation checks for the fixed-time condition-use evaluator."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SPEC = spec_from_file_location(
    "condition_usage", Path(__file__).parents[1] / "scripts" / "eval_bev_condition_usage.py"
)
usage = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(usage)


def _records(correct=2.0, zero=3.0, shuffled=4.0, delta=.5):
    return {"correct": [correct], "zero": [zero], "shuffle": [shuffled], "delta_v": [delta]}


def test_time_binned_gain_matches_fixed_time_observations():
    values = {time: _records() for time in usage.TIMES}
    early = usage.summarize_records(usage.merge_records(
        values[time] for time in usage.TIMES if time < .2
    ))
    early_two_bins = usage.summarize_records(usage.merge_records(
        values[time] for time in usage.TIMES if time < .4
    ))
    assert early["g_shuffle"] == .5
    assert early_two_bins["g_zero"] == 1.0 - 2.0 / 3.0


def test_zero_condition_observation_has_finite_diagnostic_ratios():
    zero = _records(correct=1.0, zero=1.0, shuffled=1.0, delta=0.0)
    summary = usage.summarize_records(zero)
    assert summary["g_zero"] == 0.0
    assert summary["g_shuffle"] == 0.0
    assert summary["delta_v"] == 0.0
