"""Tests for the three declarative-task authoring hardening items:

1. str.format brace footgun → clear error message
2. multi_label=False actually truncates to one label
3. tasks new / tasks validate CLI subcommands
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from episcope.episcope import app
from episcope.workflows import registry
from episcope.workflows.classification.parsing import ClassificationResponseParser
from episcope.workflows.classification.prompting import (
    ClassificationPromptBuilder,
    _safe_format,
)
runner = CliRunner()


# ---------------------------------------------------------------------------
# 1. str.format brace footgun
# ---------------------------------------------------------------------------
def test_safe_format_passes_through_normal_template() -> None:
    result = _safe_format(
        "Hello {name}, count={n_categories}",
        template_name="test",
        supported=("name", "n_categories"),
        name="world",
        n_categories=3,
    )
    assert result == "Hello world, count=3"


def test_safe_format_raises_on_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="unrecognised placeholder") as exc_info:
        _safe_format(
            "Hello {unknown_key}",
            template_name="user_prompt_template",
            supported=("title", "abstract"),
            title="A",
            abstract="B",
        )
    msg = str(exc_info.value)
    assert "user_prompt_template" in msg
    assert "unknown_key" in msg
    assert "title" in msg               # supported list is in the message
    assert "'{{'" in msg or "{{" in msg  # escape hint is present


def test_safe_format_raises_on_inline_json_braces() -> None:
    # A user who pastes a JSON example like {"key": "value"} in their prompt
    # gets a clear ValueError (KeyError path) rather than a raw Python exception.
    with pytest.raises(ValueError, match="unrecognised placeholder"):
        _safe_format(
            'Set format like {"key": "value"} here',
            template_name="system_prompt",
            supported=("n_categories",),
            n_categories=2,
        )


def test_safe_format_raises_on_unbalanced_brace() -> None:
    # A single unmatched } raises a ValueError (formatting-error path).
    with pytest.raises(ValueError, match="formatting error"):
        _safe_format(
            "Unexpected } in prompt",
            template_name="system_prompt",
            supported=("n_categories",),
            n_categories=2,
        )


def test_classification_prompt_builder_raises_on_bad_template(monkeypatch) -> None:
    spec = registry.TaskSpec.model_validate({
        "key": "bad_tmpl",
        "kind": "classifier",
        "user_prompt_template": "Title: {title} Bad brace: {oops}",
        "labels": [{"code": "a"}, {"code": "b"}],
    })
    config = registry.build_classifier_config_from_spec(spec)
    builder = ClassificationPromptBuilder(config)

    from episcope.schemas.paper import PaperMetadata
    metadata = PaperMetadata(title="Test", abstract="Abstract")

    with pytest.raises(ValueError, match="user_prompt_template"):
        builder.build_initial_prompt(metadata, [])


# ---------------------------------------------------------------------------
# 2. multi_label=False enforcement in the parser
# ---------------------------------------------------------------------------
def _make_single_label_parser() -> ClassificationResponseParser:
    spec = registry.TaskSpec.model_validate({
        "key": "single_label_test",
        "kind": "classifier",
        "multi_label": False,
        "default_label": "unclear",
        "labels": [
            {"code": "cohort", "name": "Cohort"},
            {"code": "rct", "name": "RCT"},
            {"code": "unclear", "name": "Unclear"},
        ],
    })
    config = registry.build_classifier_config_from_spec(spec)
    assert config.multi_label is False
    return ClassificationResponseParser(config)


def test_single_label_task_truncates_multi_label_response() -> None:
    parser = _make_single_label_parser()
    result = parser.parse('{"reasoning": "x", "classification": ["cohort", "rct"]}')
    assert result.classification == ["cohort"]


def test_single_label_task_keeps_single_response() -> None:
    parser = _make_single_label_parser()
    result = parser.parse('{"reasoning": "x", "classification": ["rct"]}')
    assert result.classification == ["rct"]


def test_multi_label_task_keeps_all_labels() -> None:
    spec = registry.TaskSpec.model_validate({
        "key": "multi_label_test",
        "kind": "classifier",
        "multi_label": True,
        "default_label": "unclear",
        "labels": [
            {"code": "cohort"},
            {"code": "rct"},
            {"code": "unclear"},
        ],
    })
    config = registry.build_classifier_config_from_spec(spec)
    parser = ClassificationResponseParser(config)
    result = parser.parse('{"reasoning": "x", "classification": ["cohort", "rct"]}')
    assert result.classification == ["cohort", "rct"]


# ---------------------------------------------------------------------------
# 3. tasks new / tasks validate CLI subcommands
# ---------------------------------------------------------------------------
def test_tasks_new_classifier_scaffold_is_valid_json() -> None:
    result = runner.invoke(app, ["tasks", "new", "--kind", "classifier"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["kind"] == "classifier"
    assert "labels" in data
    assert len(data["labels"]) >= 1
    assert all("code" in lbl for lbl in data["labels"])


def test_tasks_new_miner_scaffold_is_valid_json() -> None:
    result = runner.invoke(app, ["tasks", "new", "--kind", "miner"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["kind"] == "miner"
    assert "retrieval_templates" in data


def test_tasks_new_requires_kind() -> None:
    result = runner.invoke(app, ["tasks", "new"])
    assert result.exit_code == 1


def test_tasks_new_unknown_kind_exits_nonzero() -> None:
    result = runner.invoke(app, ["tasks", "new", "--kind", "bogus"])
    assert result.exit_code == 1


def test_tasks_validate_good_file(tmp_path) -> None:
    spec_file = tmp_path / "study_design.json"
    spec_file.write_text(json.dumps({
        "key": "study_design",
        "kind": "classifier",
        "labels": [{"code": "cohort"}, {"code": "unclear"}],
        "default_label": "unclear",
    }), encoding="utf-8")

    result = runner.invoke(app, ["tasks", "validate", "--task-file", str(spec_file)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
    assert "study_design" in result.output


def test_tasks_validate_bad_json(tmp_path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ this is not json", encoding="utf-8")
    result = runner.invoke(app, ["tasks", "validate", "--task-file", str(bad_file)])
    assert result.exit_code == 1


def test_tasks_validate_schema_error(tmp_path) -> None:
    # classifier task with no labels
    bad_spec = tmp_path / "bad_spec.json"
    bad_spec.write_text(
        json.dumps({"key": "broken", "kind": "classifier", "labels": []}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["tasks", "validate", "--task-file", str(bad_spec)])
    assert result.exit_code == 1


def test_tasks_validate_requires_task_file() -> None:
    result = runner.invoke(app, ["tasks", "validate"])
    assert result.exit_code == 1


def test_tasks_unknown_action_exits_nonzero() -> None:
    result = runner.invoke(app, ["tasks", "oops"])
    assert result.exit_code == 1
