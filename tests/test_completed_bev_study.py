import csv
import json

from scripts.evaluation.evaluate_completed_bev_study import final_losses, write_csv


def test_final_losses_require_actual_final_step(tmp_path):
    path = tmp_path / "loss.csv"
    path.write_text("kind,global_step,mean_loss\ntrailing_500,298500,9.0\n"
                    "trailing_500,299000,0.1\ntrailing_200,299000,0.12\n")
    assert final_losses(path) == dict(train_loss_final500=0.1, train_loss_final200=0.12)


def test_missing_final_loss_is_not_filled_with_sparse_samples(tmp_path):
    path = tmp_path / "loss.csv"
    path.write_text("kind,global_step,mean_loss\ntrailing_500,298500,0.2\n")
    assert final_losses(path) == dict(train_loss_final500=None, train_loss_final200=None)


def test_training_csv_retains_distinct_execution_metadata(tmp_path):
    path = tmp_path / "training.csv"
    write_csv(path, [dict(method="SiD2", execution="2GPU", wall_seconds=1),
                     dict(method="DiP", execution="DDP-to-single", continuation_wall_seconds=2)])
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["wall_seconds"] == ""
    assert rows[1]["continuation_wall_seconds"] == "2"
