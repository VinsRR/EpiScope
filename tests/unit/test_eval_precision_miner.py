from pathlib import Path

from eval.ragas.precision_miner_eval import (
    PrecisionMinerReferenceItem,
    _score_items,
    build_precision_miner_cases_from_references,
    build_precision_miner_references,
    read_data_accessibility_tsv,
)


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


def test_build_precision_miner_references_from_sampled_and_data_accessibility_outputs(tmp_path: Path) -> None:
    sampled = tmp_path / "sampled_papers_full.csv"
    sampled.write_text(
        "\t".join(["paper_id", "title", "journal", "year", "filepath"]) + "\n"
        "10.1/example\tExample Paper\tExample Journal\t2020\texample.pdf\n",
        encoding="utf-8",
    )

    run_dir = (
        tmp_path
        / "outputs"
        / "data-accessibility"
        / "gemini"
        / "temperature_0.0"
        / "grobid"
        / "run-a"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "final_1.tsv").write_text(
        "\t".join(["paper_id", "classification", "evidence"]) + "\n"
        "10.1/example\t['REFERENCED']\t"
        "The study used World Health Organization surveillance data. "
        "Supplementary Table S1 is available at https://example.org/s1.csv.\n",
        encoding="utf-8",
    )

    references = build_precision_miner_references(
        sampled_papers_path=sampled,
        data_accessibility_roots=[tmp_path / "outputs"],
    )
    cases = build_precision_miner_cases_from_references(references)

    assert bool(references.loc[0, "expected_data_source"]) is True
    assert bool(references.loc[0, "expected_supplementary_link"]) is True
    assert "World Health Organization" in references.loc[0, "data_source_reference_items_json"]
    assert len(cases) == 2
    assert {case.miner_kind for case in cases} == {
        "find_data_sources",
        "find_supplementary_links",
    }
    assert all(case.paper_id == "10.1/example" for case in cases)
    assert cases[0].metadata["expected_present"] is True


def test_build_precision_miner_references_errors_on_missing_sampled_outputs(tmp_path: Path) -> None:
    sampled = tmp_path / "sampled_papers_full.csv"
    sampled.write_text(
        "paper_id\ttitle\n10.1/missing\tMissing Paper\n",
        encoding="utf-8",
    )
    run_dir = (
        tmp_path
        / "outputs"
        / "data-accessibility"
        / "gemini"
        / "temperature_0.0"
        / "grobid"
        / "run-a"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "final_1.tsv").write_text(
        "paper_id\tclassification\tevidence\n10.1/other\t['REFERENCED']\tUses CDC data.\n",
        encoding="utf-8",
    )

    try:
        build_precision_miner_references(
            sampled_papers_path=sampled,
            data_accessibility_roots=[tmp_path / "outputs"],
        )
    except ValueError as exc:
        assert "missing 1 sampled paper_id" in str(exc)
    else:
        raise AssertionError("Expected missing sampled paper_id error")


def test_read_data_accessibility_tsv_requires_reference_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.tsv"
    path.write_text("paper_id\tclassification\n10.1/example\t['REFERENCED']\n", encoding="utf-8")

    try:
        read_data_accessibility_tsv(path)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
        assert "evidence" in str(exc)
    else:
        raise AssertionError("Expected malformed TSV error")
