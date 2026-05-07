from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.run_classification_baselines as baseline_runner
from episcope.db.in_memory_academic_db import InMemoryAcademicDB
from episcope.schemas import PaperMetadata, StructuredSection
from episcope.workflows.classification.baselines import (
    MajorityLabelBaseline,
    PrototypeSimilarityBaseline,
    TopicModelBaseline,
    ground_truth_column,
    metadata_from_mapping,
    parse_label_names,
)
from episcope.workflows.classification.schemas import (
    DataAccessibility,
    PaperType,
)


def test_parse_label_names_and_majority_leave_one_out() -> None:
    assert parse_label_names("[‘REPORTED’, 'REFERENCED']") == (
        "REFERENCED",
        "REPORTED",
    )

    records = [
        {"paper_id": "p1", "ptype_classification": "['EMPIRICAL']"},
        {"paper_id": "p2", "ptype_classification": "['EMPIRICAL']"},
        {"paper_id": "p3", "ptype_classification": "['INFERENCE']"},
    ]
    baseline = MajorityLabelBaseline.from_records(
        classifier_kind="paper_type",
        records=records,
        ground_truth_column=ground_truth_column("paper_type"),
        leave_one_out=True,
    )

    prediction = baseline.predict("p3", metadata_from_mapping(records[2]))
    assert prediction.result.classification == [PaperType.EMPIRICAL]


def test_prototype_similarity_uses_label_templates() -> None:
    record = {
        "paper_id": "p1",
        "title": "Data available from the corresponding author upon reasonable request",
    }
    baseline = PrototypeSimilarityBaseline(
        classifier_kind="data_accessibility",
        records=[record],
    )
    prediction = baseline.predict("p1", metadata_from_mapping(record), record)

    assert prediction.result.classification
    assert prediction.result.classification[0] in {
        DataAccessibility.AVAILABLE_UPON_REQUEST,
        DataAccessibility.OPEN,
    }


def test_lsa_topic_baseline_maps_topics_to_majority_labels() -> None:
    records = [
        {"paper_id": "p1", "title": "hospital surveillance outbreak case counts", "ptype_classification": "['EMPIRICAL']"},
        {"paper_id": "p2", "title": "field surveillance confirmed cases and deaths", "ptype_classification": "['EMPIRICAL']"},
        {"paper_id": "p3", "title": "bayesian seir model inference reproduction number", "ptype_classification": "['INFERENCE']"},
        {"paper_id": "p4", "title": "mechanistic model estimates epidemic parameters", "ptype_classification": "['INFERENCE']"},
    ]
    baseline = TopicModelBaseline(
        classifier_kind="paper_type",
        model_kind="lsa",
        records=records,
        ground_truth_records=records,
        ground_truth_column=ground_truth_column("paper_type"),
        n_topics=2,
    )

    prediction = baseline.predict("p1", metadata_from_mapping(records[0]), records[0])
    assert prediction.result.classification
    assert prediction.result.extras["topic_model"] == "lsa"


def test_paper_text_from_db_uses_sections_as_stored() -> None:
    db = InMemoryAcademicDB(backup_file=None)
    db.insert(
        "paper-1",
        "metadata",
        "grobid",
        PaperMetadata(title="Useful title", abstract="Useful abstract").to_dict(),
    )
    db.insert(
        "paper-1",
        "sections",
        "grobid",
        [
            StructuredSection(
                title="Methods",
                content="We analyzed surveillance data.",
                section_type="methods",
            ).to_dict(),
            StructuredSection(
                title="References",
                content="Smith et al. A cited paper.",
                section_type="references",
            ).to_dict(),
        ],
    )

    metadata, text, stats = baseline_runner.paper_text_from_db(db, "paper-1", "grobid")

    assert metadata.title == "Useful title"
    assert "surveillance data" in text
    assert "Smith et al" in text
    assert stats["section_count"] == 2
    assert stats["used_section_count"] == 2


def test_run_classification_baselines_script_writes_eval_compatible_tsv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "papers.tsv"
    pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "title": "Case counts reported in Table 1",
                "availability_classification": "['REPORTED']",
                "geo_classification": "['EUROPE']",
                "data_type_classification": "['TRADITIONAL']",
                "ptype_classification": "['EMPIRICAL']",
            },
            {
                "paper_id": "p2",
                "title": "Data are available in Zenodo",
                "availability_classification": "['OPEN']",
                "geo_classification": "['EUROPE']",
                "data_type_classification": "['TRADITIONAL']",
                "ptype_classification": "['EMPIRICAL']",
            },
        ]
    ).to_csv(input_csv, sep="\t", index=False)
    output_dir = tmp_path / "outputs"
    db = InMemoryAcademicDB(backup_file=None)
    for paper_id, title, section in (
        ("p1", "Case counts reported in Table 1", "We report surveillance counts."),
        ("p2", "Data are available in Zenodo", "Open data are deposited online."),
    ):
        db.insert(
            paper_id,
            "metadata",
            "grobid",
            PaperMetadata(title=title, abstract="").to_dict(),
        )
        db.insert(
            paper_id,
            "sections",
            "grobid",
            [StructuredSection(title="Methods", content=section).to_dict()],
        )

    monkeypatch.setattr(baseline_runner, "build_mongo_db", lambda settings: db)

    records = baseline_runner.load_records(input_csv, "\t")
    baseline_runner.run_once(
        baseline_runner.Settings(
            input_csv=str(input_csv),
            ground_truth_csv=str(input_csv),
            base_output_dir=str(output_dir),
            mongo_uri="mongodb://example.invalid",
            classifier_kinds=("data_accessibility",),
            baseline_kinds=("lsa",),
            topic_n_topics=2,
            checkpoint_every=10,
        ),
        classifier_kind="data_accessibility",
        baseline_kind="lsa",
        repeat_idx=1,
        records=records,
        ground_truth_records=records,
    )

    files = list(output_dir.rglob("final_1.tsv"))
    assert len(files) == 1
    result = pd.read_csv(files[0], sep="\t")
    assert set(result["paper_id"]) == {"p1", "p2"}
    assert "classification" in result.columns
    assert not any("temperature_" in part for part in files[0].parts)
