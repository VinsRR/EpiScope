"""Tests for the plain-language error translation added in usability Phase 1:

1. humanize_task_spec_error() strips pydantic's internal framing from
   TaskSpec ValidationErrors.
2. humanize_error() delegates to it for ValidationErrors and otherwise
   matches known infra-jargon messages against a plain-language hint table.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from episcope.services.errors import humanize_error
from episcope.workflows import registry
from episcope.workflows.registry import humanize_task_spec_error


def _validation_error(data: dict) -> ValidationError:
    with pytest.raises(ValidationError) as exc_info:
        registry.TaskSpec.model_validate(data)
    return exc_info.value


# ---------------------------------------------------------------------------
# humanize_task_spec_error
# ---------------------------------------------------------------------------
def test_bad_key_format_strips_pydantic_framing() -> None:
    exc = _validation_error({"key": "Bad Key", "kind": "miner", "retrieval_templates": ["x"]})
    message = humanize_task_spec_error(exc)

    assert "Task definition is invalid:" in message
    assert "key must contain only lowercase letters" in message
    # Pydantic's internal framing must not leak through.
    assert "type=value_error" not in message
    assert "errors.pydantic.dev" not in message
    assert "input_value" not in message


def test_missing_labels_message_is_plain() -> None:
    exc = _validation_error({"key": "empty", "kind": "classifier"})
    message = humanize_task_spec_error(exc)

    assert "must define at least one label" in message
    assert "errors.pydantic.dev" not in message


def test_duplicate_codes_message_is_plain() -> None:
    exc = _validation_error(
        {
            "key": "dup",
            "kind": "classifier",
            "labels": [{"code": "a"}, {"code": "a"}],
        }
    )
    message = humanize_task_spec_error(exc)
    assert "duplicate label codes" in message


def test_missing_required_field_is_humanized() -> None:
    exc = _validation_error({"kind": "classifier", "labels": [{"code": "a"}]})
    message = humanize_task_spec_error(exc)
    assert "key" in message
    assert "is missing" in message


def test_non_validation_error_passes_through_unchanged() -> None:
    exc = ValueError("some other failure")
    assert humanize_task_spec_error(exc) == "some other failure"


# ---------------------------------------------------------------------------
# humanize_error
# ---------------------------------------------------------------------------
def test_humanize_error_delegates_to_task_spec_for_validation_errors() -> None:
    exc = _validation_error({"key": "Bad Key", "kind": "miner", "retrieval_templates": ["x"]})
    assert humanize_error(exc) == humanize_task_spec_error(exc)


@pytest.mark.parametrize(
    "message,expected_hint",
    [
        (
            "Mongo metadata backend requires --mongo-uri or MONGO_URI.",
            "No paper-metadata database is configured.",
        ),
        (
            "Could not infer an embedding provider from model name 'weird'. "
            "Pass provider=... or set EPISCOPE_EMBED_PROVIDER.",
            "EpiScope doesn't recognize this embedding model name.",
        ),
        (
            "Could not connect to Qdrant at http://localhost:6333 while "
            "opening collection 'x'.",
            "The vector search database isn't reachable.",
        ),
        (
            "Collection 'x' is not in named-vector mode. Please migrate or recreate it.",
            "This vector collection was built with an older, incompatible format.",
        ),
        (
            "Qdrant collection 'x' was not found at http://localhost:6333, "
            "and dense_dim was not provided to create it.",
            "This corpus hasn't been indexed yet, or the collection name doesn't match.",
        ),
        (
            "VectorDB does not have a dense embedding model configured.",
            "This corpus hasn't been indexed yet. Run `episcope index ...` first.",
        ),
    ],
)
def test_humanize_error_prepends_plain_language_hint(message: str, expected_hint: str) -> None:
    result = humanize_error(ValueError(message))
    assert result.startswith(expected_hint)
    # The original, actionable technical detail is preserved, not discarded.
    assert message in result


def test_humanize_error_falls_back_to_original_message_when_unmatched() -> None:
    exc = ValueError("some completely unrelated failure")
    assert humanize_error(exc) == "some completely unrelated failure"
