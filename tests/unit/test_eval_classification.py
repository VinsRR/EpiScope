from __future__ import annotations

from pathlib import Path

import pandas as pd

from eval.classification.error_analysis import containment_pro_vs_flash, load_merged_predictions, per_run_metrics, within_config_consistency
from eval.classification.metrics import jaccard_samples, multilabel_prf, subset_accuracy
from eval.common.labels import parse_label_set


def test_parse_label_set_handles_curly_quotes_and_lists() -> None:
    assert parse_label_set("['REPORTED', 'REFERENCED']") == frozenset({"REPORTED", "REFERENCED"})
    assert parse_label_set("[‘REPORTED’]") == frozenset({"REPORTED"})
    assert parse_label_set("REPORTED, REFERENCED") == frozenset({"REPORTED", "REFERENCED"})


def test_multilabel_metrics_support_dict_inputs() -> None:
    y_true = {"paper-1": "A,B", "paper-2": "A"}
    y_pred = {"paper-1": "A,B", "paper-2": "B"}

    assert jaccard_samples(y_true, y_pred, sep=",") == 0.5
    assert subset_accuracy(y_true, y_pred, sep=",") == 0.5

    prf = multilabel_prf(y_true, y_pred, average="micro", sep=",")
    assert prf["precision"] == 2 / 3
    assert prf["recall"] == 2 / 3
    assert prf["f1"] == 2 / 3


def test_classification_eval_loads_and_summarizes_runs(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.tsv"
    pd.DataFrame(
        [
            {"paper_id": "paper-1", "availability_classification": "['REPORTED']", "geo_classification": "['EUROPE']", "data_type_classification": "['TRADITIONAL']", "ptype_classification": "['EMPIRICAL']"},
            {"paper_id": "paper-2", "availability_classification": "['REFERENCED']", "geo_classification": "['ASIA']", "data_type_classification": "['SYNTHETIC']", "ptype_classification": "['INFERENCE']"},
        ]
    ).to_csv(gt_path, sep="\t", index=False)

    run_root = tmp_path / "outputs"
    run_dir = run_root / "data-accessibility" / "gemini-2-5-pro" / "temperature_0.0" / "grobid" / "run-a"
    run_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"paper_id": "paper-1", "classification": "['REPORTED']"},
            {"paper_id": "paper-2", "classification": "['NOT_STATED']"},
        ]
    ).to_csv(run_dir / "final_1.tsv", sep="\t", index=False)

    run_dir_2 = run_root / "data-accessibility" / "gemini-2-5-flash" / "temperature_0.0" / "grobid" / "run-b"
    run_dir_2.mkdir(parents=True)
    pd.DataFrame(
        [
            {"paper_id": "paper-1", "classification": "['REPORTED']"},
            {"paper_id": "paper-2", "classification": "['REFERENCED']"},
        ]
    ).to_csv(run_dir_2 / "final_1.tsv", sep="\t", index=False)

    run_dir_3 = run_root / "data-accessibility" / "guided_lsa" / "grobid" / "run-c"
    run_dir_3.mkdir(parents=True)
    pd.DataFrame(
        [
            {"paper_id": "paper-1", "classification": "['REPORTED']"},
            {"paper_id": "paper-2", "classification": "['REFERENCED']"},
        ]
    ).to_csv(run_dir_3 / "final_1.tsv", sep="\t", index=False)

    per_run = per_run_metrics(ground_truth_path=gt_path, run_roots=[run_root])
    assert len(per_run) == 3
    assert set(per_run["task"]) == {"data-accessibility"}
    assert "unknown" in set(per_run["temperature"])

    merged = load_merged_predictions(ground_truth_path=gt_path, run_roots=[run_root])
    assert len(merged) == 6
    assert merged["is_incorrect"].sum() == 1

    consistency = within_config_consistency(merged)
    assert set(consistency["model"]) == {"gemini-2-5-pro", "gemini-2-5-flash", "guided_lsa"}

    containment = containment_pro_vs_flash(merged, temperature="0.0")
    assert "data-accessibility" in set(containment["task"])
    availability_row = containment[containment["task"] == "data-accessibility"].iloc[0]
    assert availability_row["n_pro_errors"] == 1
    assert availability_row["n_flash_errors"] == 0
