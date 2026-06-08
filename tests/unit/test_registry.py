from __future__ import annotations

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


def test_generated_enums_match_registry_keys() -> None:
    assert [m.value for m in registry.ClassifierKind] == list(registry.CLASSIFIERS)
    assert [m.value for m in registry.PrecisionMinerKind] == list(registry.MINERS)


def test_build_classifier_config_returns_typed_config_with_top_k() -> None:
    config = registry.build_classifier_config("data_accessibility", top_k=7)
    assert isinstance(config, DataAccessibilityClassifierConfig)
    assert isinstance(config, BaseClassifierConfig)
    assert config.top_k == 7


def test_build_classifier_config_accepts_enum_member() -> None:
    config = registry.build_classifier_config(registry.ClassifierKind("geo"), top_k=3)
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
        assert set(entry) == {"key", "label", "description"}
        assert all(isinstance(value, str) for value in entry.values())


def test_cli_uses_registry_enums() -> None:
    from episcope import episcope as cli

    assert cli.ClassifierKind is registry.ClassifierKind
    assert cli.PrecisionMinerKind is registry.PrecisionMinerKind


def test_api_request_models_use_registry_enums() -> None:
    pytest.importorskip("fastapi")
    from episcope.api import ClassificationRequest, PrecisionMinerRequest

    assert (
        ClassificationRequest.model_fields["classifier_kind"].annotation
        is registry.ClassifierKind
    )
    assert (
        PrecisionMinerRequest.model_fields["miner_kind"].annotation
        is registry.PrecisionMinerKind
    )
