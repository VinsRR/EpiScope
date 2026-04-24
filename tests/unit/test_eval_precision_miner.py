from __future__ import annotations

import json
from pathlib import Path

from eval.ragas.precision_miner_eval import (
    PrecisionMinerReferenceItem,
    _score_items,
    load_precision_miner_cases,
)


def test_load_precision_miner_cases_resolves_relative_paths_and_normalizes_items(tmp_path: Path) -> None:
    paper_path = tmp_path / "doc.txt"
    paper_path.write_text("Example content", encoding="utf-8")

    dataset_path = tmp_path / "cases.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "case_id": "case-1",
                "miner_kind": "find_data_sources",
                "paper_path": "doc.txt",
                "reference_answer": "The study used NHANES.",
                "reference_items": [
                    {
                        "canonical_name": "NHANES",
                        "aliases": ["National Health and Nutrition Examination Survey"],
                        "role_in_study": "external_dataset",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_precision_miner_cases(dataset_path)

    assert len(cases) == 1
    assert cases[0].paper_path == str(paper_path.resolve())
    assert cases[0].reference_items[0].canonical_name == "NHANES"
    assert cases[0].reference_items[0].metadata["role_in_study"] == "external_dataset"


def test_score_items_tracks_required_recall_and_explanation_coverage() -> None:
    predicted_items = [
        {
            "name": "NHANES",
            "url": None,
            "explanation": "The paper explicitly uses NHANES survey microdata.",
            "raw_text": "We used NHANES data.",
        },
        {
            "name": "Unmatched source",
            "url": None,
            "explanation": "Noise",
            "raw_text": "Noise",
        },
    ]
    reference_items = [
        PrecisionMinerReferenceItem(
            canonical_name="NHANES",
            aliases=["National Health and Nutrition Examination Survey"],
            expected_explanation_contains=["survey", "microdata"],
            required=True,
        ),
        PrecisionMinerReferenceItem(
            canonical_name="BRFSS",
            required=False,
        ),
    ]

    scores = _score_items(predicted_items, reference_items)

    assert scores["item_precision"] == 0.5
    assert scores["item_recall"] == 0.5
    assert scores["required_item_recall"] == 1.0
    assert scores["explanation_coverage"] == 1.0
