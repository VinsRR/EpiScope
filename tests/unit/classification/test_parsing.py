from __future__ import annotations

import pytest

from episcope.workflows.classification.config import DataAccessibilityClassifierConfig
from episcope.workflows.classification.parsing import (
    ClassificationParseError,
    ClassificationResponseParser,
)
from episcope.workflows.classification.schemas import DataAccessibility


def test_parser_accepts_fenced_json_and_maps_codes() -> None:
    config = DataAccessibilityClassifierConfig()
    parser = ClassificationResponseParser(config)

    response = """```json
    {
      "reasoning": "The paper says the data are deposited in a public repository.",
      "confidence": 0.92,
      "class_probabilities": {"A": 0.92, "B": 0.08},
      "classification": ["A"]
    }
    ```"""

    result = parser.parse(response)

    assert result.classification == [DataAccessibility.OPEN]
    assert result.confidence == pytest.approx(0.92)
    assert result.class_probabilities == {"A": 0.92, "B": 0.08}
    assert result.evidence["reasoning"].startswith("The paper says")


def test_parser_raises_when_no_json_can_be_found() -> None:
    config = DataAccessibilityClassifierConfig()
    parser = ClassificationResponseParser(config)

    with pytest.raises(ClassificationParseError, match="No JSON object found"):
        parser.parse("This answer is plain text and never returns JSON.")


def test_fallback_result_uses_default_classification_and_uniform_probs() -> None:
    config = DataAccessibilityClassifierConfig()
    parser = ClassificationResponseParser(config)

    result = parser.create_fallback_result()

    assert result.classification == config.default_classification
    assert result.confidence == 0.0
    assert len(result.class_probabilities) == len(config.category_labels)
    for probability in result.class_probabilities.values():
        assert probability == pytest.approx(1 / len(config.category_labels))
    assert "defaulting" in result.evidence["reasoning"].lower()
