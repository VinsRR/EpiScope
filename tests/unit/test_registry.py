from __future__ import annotations

import json

import pytest

from episcope.workflows import registry
from episcope.workflows.classification.config import (
    BaseClassifierConfig,
    DataAccessibilityClassifierConfig,
)
from episcope.workflows.precision_miner.config import (
    FindDataSourcesConfig,
    PrecisionMinerConfig,
)


def _classifier_spec_dict() -> dict:
    return {
        "key": "study_design",
        "kind": "classifier",
        "label": "Study design",
        "multi_label": False,
        "default_label": "unclear",
        "labels": [
            {"code": "cohort", "name": "Cohort", "definition": "Follows groups."},
            {"code": "unclear", "name": "Unclear", "definition": "No information."},
        ],
    }


# ---------------------------------------------------------------------------
# Built-in dispatch
# ---------------------------------------------------------------------------
def test_build_classifier_config_returns_typed_config_with_top_k() -> None:
    config = registry.build_classifier_config("data_accessibility", top_k=7)
    assert isinstance(config, DataAccessibilityClassifierConfig)
    assert isinstance(config, BaseClassifierConfig)
    assert config.top_k == 7


def test_build_classifier_config_accepts_any_builtin_key() -> None:
    config = registry.build_classifier_config("geo", top_k=3)
    assert config.top_k == 3
    assert type(config).__name__ == "GeoClassifierConfig"


def test_build_precision_miner_config_returns_typed_config_with_top_k() -> None:
    config = registry.build_precision_miner_config("find_data_sources", top_k=9)
    assert isinstance(config, FindDataSourcesConfig)
    assert isinstance(config, PrecisionMinerConfig)
    assert config.top_k == 9


def test_unknown_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown classifier kind"):
        registry.build_classifier_config("does_not_exist", top_k=1)
    with pytest.raises(ValueError, match="Unknown precision-miner kind"):
        registry.build_precision_miner_config("does_not_exist", top_k=1)


def test_catalogs_cover_every_kind_and_are_serializable() -> None:
    classifiers = registry.classifier_catalog()
    miners = registry.miner_catalog()
    assert {c["key"] for c in classifiers} == set(registry.CLASSIFIERS)
    assert {m["key"] for m in miners} == set(registry.MINERS)
    for entry in classifiers + miners:
        assert set(entry) == {"key", "label", "description", "source"}
        assert all(isinstance(value, str) for value in entry.values())
        assert entry["source"] == "builtin"


# ---------------------------------------------------------------------------
# Declarative tasks (config-as-data)
# ---------------------------------------------------------------------------
def test_register_declarative_classifier_builds_and_appears_in_catalog() -> None:
    spec = registry.TaskSpec.model_validate(_classifier_spec_dict())
    registry.register_task(spec)

    assert "study_design" in registry.CLASSIFIERS
    config = registry.build_classifier_config("study_design", top_k=5)
    assert isinstance(config, BaseClassifierConfig)
    assert config.top_k == 5
    assert set(config.category_labels) == {"cohort", "unclear"}

    entry = next(
        e for e in registry.classifier_catalog() if e["key"] == "study_design"
    )
    assert entry["source"] == "declarative"
    assert entry["label"] == "Study design"


def test_declarative_classifier_parses_and_filters_codes() -> None:
    from episcope.workflows.classification.parsing import ClassificationResponseParser

    spec = registry.TaskSpec.model_validate(_classifier_spec_dict())
    parser = ClassificationResponseParser(
        registry.build_classifier_config_from_spec(spec)
    )

    good = parser.parse('{"reasoning": "x", "classification": ["cohort"]}')
    assert good.classification == ["cohort"]

    # Unknown codes are dropped in favour of the task's default label.
    fallback = parser.parse('{"reasoning": "x", "classification": ["nope"]}')
    assert fallback.classification == ["unclear"]


def test_register_declarative_miner_builds() -> None:
    spec = registry.TaskSpec.model_validate(
        {
            "key": "find_funding",
            "kind": "miner",
            "retrieval_templates": ["Who funded this study?"],
        }
    )
    registry.register_task(spec)

    config = registry.build_precision_miner_config("find_funding", top_k=8)
    assert isinstance(config, PrecisionMinerConfig)
    assert config.top_k == 8
    assert config.retrieval_templates == ["Who funded this study?"]


def test_load_task_file_registers(tmp_path) -> None:
    path = tmp_path / "study_design.json"
    path.write_text(json.dumps(_classifier_spec_dict()), encoding="utf-8")

    spec = registry.load_task_file(path)
    assert spec.key == "study_design"
    assert "study_design" in registry.CLASSIFIERS


def test_cannot_override_builtin_key() -> None:
    spec = registry.TaskSpec.model_validate(
        {"key": "geo", "kind": "classifier", "labels": [{"code": "x"}]}
    )
    with pytest.raises(ValueError, match="built-in"):
        registry.register_task(spec)


def test_build_from_spec_does_not_register() -> None:
    spec = registry.TaskSpec.model_validate(
        {"key": "scratch_task", "kind": "classifier", "labels": [{"code": "a"}, {"code": "b"}]}
    )
    config = registry.build_classifier_config_from_spec(spec)
    assert "scratch_task" not in registry.CLASSIFIERS
    assert set(config.category_labels) == {"a", "b"}


def test_classifier_task_requires_labels() -> None:
    with pytest.raises(ValueError):
        registry.TaskSpec.model_validate({"key": "empty", "kind": "classifier"})


def test_task_key_must_be_identifier_like() -> None:
    with pytest.raises(ValueError):
        registry.TaskSpec.model_validate(
            {"key": "Bad Key", "kind": "miner", "retrieval_templates": ["x"]}
        )
