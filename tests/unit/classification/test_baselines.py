from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.run_classification_baselines as baseline_runner
from episcope.db.in_memory_academic_db import InMemoryAcademicDB
from episcope.schemas import PaperMetadata, StructuredSection
from classification.baselines import (
    MajorityLabelBaseline,
    PrototypeSimilarityBaseline,
    SUPERVISED_BASELINES,
    TOPIC_MODEL_BASELINES,
    SupervisedCVBaseline,
    TopicModelBaseline,
    ground_truth_column,
    metadata_from_mapping,
    parse_label_names,
)
from classification.baselines.topic_models import _seed_topic_list
from classification.runners.families import family_parser
from classification.topic_modeling import TopicExplorer
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


def test_topic_k_values_scale_with_task_label_count() -> None:
    settings = baseline_runner.Settings(
        topic_k_multipliers=(0.5, 1.0, 2.0),
        topic_k_values=(),
    )

    assert baseline_runner.task_label_count("data_accessibility") == 6
    assert baseline_runner.topic_k_values_for_task(settings, "data_accessibility") == (
        3,
        6,
        12,
    )


def test_optional_topic_modes_expose_bertopic_and_top2vec_capabilities() -> None:
    assert "bertopic_guided" in TOPIC_MODEL_BASELINES
    assert "bertopic_semisupervised" in TOPIC_MODEL_BASELINES
    assert "top2vec_contextual" in TOPIC_MODEL_BASELINES
    assert "supervised_bertopic" in SUPERVISED_BASELINES
    assert baseline_runner.normalize_baseline_kind("bertopic_supervised") == "supervised_bertopic"


def test_bertopic_guided_seed_topics_are_built_from_classifier_protocol() -> None:
    seeds = _seed_topic_list("paper_type")
    flattened = {term for seed in seeds for term in seed}

    assert len(seeds) >= 3
    assert "empirical" in flattened
    assert "inference" in flattened


def test_topic_explorer_can_cluster_documents_for_eda() -> None:
    fitted = TopicExplorer(model_kind="lsa", n_topics=2).fit_texts(
        [
            "hospital surveillance outbreak case counts",
            "bayesian seir model inference reproduction number",
        ],
        ids=["paper-1", "paper-2"],
    )

    assert set(fitted.document_topics()["paper_id"]) == {"paper-1", "paper-2"}
    assert set(fitted.clusters()).issubset({0, 1})
    assert not fitted.topics().empty


def test_family_runners_expose_narrow_baseline_choices() -> None:
    topic_choices = family_parser(
        family="topic",
        description="topic",
    )._option_string_actions["--baseline-kind"].choices
    supervised_choices = family_parser(
        family="supervised",
        description="supervised",
    )._option_string_actions["--baseline-kind"].choices

    assert "bertopic_guided" in topic_choices
    assert "supervised_tfidf_logreg" not in topic_choices
    assert "supervised_bertopic" in supervised_choices
    assert "lda" not in supervised_choices


def test_supervised_tfidf_baseline_runs_kfold_predictions() -> None:
    records = [
        {"paper_id": "p1", "_metadata_text": "data deposited on zenodo repository", "availability_classification": "['OPEN']"},
        {"paper_id": "p2", "_metadata_text": "supplementary csv files are publicly available", "availability_classification": "['OPEN']"},
        {"paper_id": "p3", "_metadata_text": "data available from author upon reasonable request", "availability_classification": "['AVAILABLE_UPON_REQUEST']"},
        {"paper_id": "p4", "_metadata_text": "access requires committee approval and data use agreement", "availability_classification": "['AVAILABLE_UPON_REQUEST']"},
    ]
    baseline = SupervisedCVBaseline(
        classifier_kind="data_accessibility",
        baseline_kind="supervised_tfidf_logreg",
        records=records,
        ground_truth_records=records,
        ground_truth_column=ground_truth_column("data_accessibility"),
        cv_mode="kfold",
        cv_folds=2,
        min_df=1,
    )

    prediction = baseline.predict("p1", metadata_from_mapping(records[0]), records[0])
    assert prediction.result.classification
    assert prediction.result.extras["cv_mode"] == "kfold"


def test_supervised_topic_settings_do_not_sweep_topic_k_values() -> None:
    settings = baseline_runner.Settings(
        topic_n_topics=10,
        topic_k_values=(3, 6),
        supervised_cv_modes=("kfold", "leave_one_out"),
    )

    expanded = baseline_runner.settings_for_baseline(
        settings,
        classifier_kind="data_accessibility",
        baseline_kind="supervised_lsa_logreg",
    )

    assert [(item.topic_n_topics, item.supervised_cv_modes[0]) for item in expanded] == [
        (10, "kfold"),
        (10, "leave_one_out"),
    ]


def test_supervised_cv_defaults_to_fixed_kfold() -> None:
    assert baseline_runner.Settings().supervised_cv_modes == ("kfold",)


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
    assert "Useful abstract" in text
    assert "surveillance data" in text
    assert "Smith et al" in text
    assert stats["text_source"] == "mongo_full_text"
    assert stats["text_scope"] == "full_text"
    assert stats["section_count"] == 2
    assert stats["used_section_count"] == 2
    assert stats["metadata_abstract_char_count"] == len("Useful abstract")


def test_paper_text_from_db_can_use_abstract_only() -> None:
    db = InMemoryAcademicDB(backup_file=None)
    db.insert(
        "paper-1",
        "metadata",
        "grobid",
        PaperMetadata(
            title="Title should not be included",
            abstract="Only this abstract should be used.",
        ).to_dict(),
    )
    db.insert(
        "paper-1",
        "sections",
        "grobid",
        [
            StructuredSection(
                title="Methods",
                content="Section text should not be included.",
            ).to_dict(),
        ],
    )

    metadata, text, stats = baseline_runner.paper_text_from_db(
        db,
        "paper-1",
        "grobid",
        text_scope="abstract",
    )

    assert metadata.title == "Title should not be included"
    assert text == "Only this abstract should be used."
    assert "Title should not be included" not in text
    assert "Section text" not in text
    assert stats["text_source"] == "mongo_abstract"
    assert stats["text_scope"] == "abstract"
    assert stats["section_count"] == 1
    assert stats["used_section_count"] == 0
    assert stats["metadata_abstract_char_count"] == len("Only this abstract should be used.")
    assert stats["abstract_section_count"] == 0


def test_paper_text_from_db_abstract_only_falls_back_to_abstract_section() -> None:
    db = InMemoryAcademicDB(backup_file=None)
    db.insert(
        "paper-1",
        "metadata",
        "grobid",
        PaperMetadata(title="Useful title", abstract="").to_dict(),
    )
    db.insert(
        "paper-1",
        "sections",
        "grobid",
        [
            StructuredSection(
                title="Abstract",
                content="This is an abstract section from Mongo sections.",
                section_type="introduction",
            ).to_dict(),
            StructuredSection(
                title="Methods",
                content="This methods text should not be included.",
                section_type="methods",
            ).to_dict(),
        ],
    )

    _, text, stats = baseline_runner.paper_text_from_db(
        db,
        "paper-1",
        "grobid",
        text_scope="abstract",
    )

    assert text == "This is an abstract section from Mongo sections."
    assert "methods text" not in text
    assert stats["section_count"] == 2
    assert stats["used_section_count"] == 1
    assert stats["metadata_abstract_char_count"] == 0
    assert stats["abstract_section_count"] == 1


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
            topic_k_multipliers=(),
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
    assert set(result["text_scope"]) == {"full_text"}
    assert "lsa-k_2" in files[0].parts
    assert "full_text" in files[0].parts
    assert not any("temperature_" in part for part in files[0].parts)


def test_run_classification_baselines_marks_missing_abstract_as_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "papers.tsv"
    pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "availability_classification": "['OPEN']",
                "geo_classification": "['EUROPE']",
                "data_type_classification": "['TRADITIONAL']",
                "ptype_classification": "['EMPIRICAL']",
            },
            {
                "paper_id": "p2",
                "availability_classification": "['REFERENCED']",
                "geo_classification": "['ASIA']",
                "data_type_classification": "['SYNTHETIC']",
                "ptype_classification": "['INFERENCE']",
            },
        ]
    ).to_csv(input_csv, sep="\t", index=False)
    output_dir = tmp_path / "outputs"
    db = InMemoryAcademicDB(backup_file=None)
    db.insert(
        "p1",
        "metadata",
        "grobid",
        PaperMetadata(abstract="This paper has an abstract.").to_dict(),
    )
    db.insert(
        "p1",
        "sections",
        "grobid",
        [StructuredSection(title="Methods", content="Methods text.").to_dict()],
    )
    db.insert(
        "p2",
        "metadata",
        "grobid",
        PaperMetadata(abstract="").to_dict(),
    )
    db.insert(
        "p2",
        "sections",
        "grobid",
        [StructuredSection(title="Methods", content="No abstract here.").to_dict()],
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
            baseline_kinds=("majority",),
            text_scope="abstract",
            checkpoint_every=10,
        ),
        classifier_kind="data_accessibility",
        baseline_kind="majority",
        repeat_idx=1,
        records=records,
        ground_truth_records=records,
    )

    files = list(output_dir.rglob("final_1.tsv"))
    assert len(files) == 1
    result = pd.read_csv(files[0], sep="\t")
    skipped = result[result["paper_id"] == "p2"].iloc[0]
    evaluated = result[result["paper_id"] == "p1"].iloc[0]
    assert skipped["evaluation_status"] == "skipped"
    assert skipped["classification"] == "__SKIPPED__"
    assert skipped["skip_reason"] == "missing_abstract"
    assert evaluated["evaluation_status"] == "evaluated"
    assert "abstract" in files[0].parts
