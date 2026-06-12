from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from episcope.episcope import app
from episcope.rag.provenance import Provenance


runner = CliRunner()


@pytest.fixture(autouse=True)
def _json_output_by_default(monkeypatch) -> None:
    """Default CLI tests to JSON output so they can assert on structure.

    Individual tests can still pass ``--format human`` to override this.
    """
    monkeypatch.setenv("EPISCOPE_OUTPUT_FORMAT", "json")


@pytest.fixture(autouse=True)
def _default_gemini_key(monkeypatch) -> None:
    """Provide a dummy key so generation commands resolve to (the monkeypatched)
    ``_build_generator`` instead of the keyless Ollama fallback. Tests that
    exercise the fallback delete it explicitly.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


class _FakeEmbedder:
    model_name = "fake-embedder/1"
    dim = 4

    def embed_text(self, text: str):
        size = float(len(text))
        lower = text.lower()
        return [
            1.0,
            float("zenodo" in lower),
            float("supplementary" in lower),
            size % 11.0,
        ]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


class _FakeClassifierGenerator:
    model_id = "fake-generator"

    def generate(self, contexts, **kwargs):
        return Provenance(
            answer="""
            {
              "reasoning": "The paper explicitly says the data are on Zenodo.",
              "confidence": 0.92,
              "class_probabilities": {"A": 0.92, "E": 0.08},
              "classification": ["A"]
            }
            """,
            evidences=[],
        )


class _FakeAnswerGenerator:
    def generate(self, contexts, **kwargs):
        return Provenance(answer="The paper says the data are on Zenodo.", evidences=[])


def test_inspect_command_reads_text_file(tmp_path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(paper)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["paper_id"] == "paper"
    assert payload["section_count"] >= 1
    assert payload["reference_count"] == 0


def test_index_and_papers_use_local_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    db_backup = tmp_path / "academic_db.json"

    index_result = runner.invoke(
        app,
        [
            "index",
            str(paper),
            "--index-dir",
            str(index_dir),
            "--db-backup",
            str(db_backup),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert index_result.exit_code == 0
    index_payload = json.loads(index_result.stdout)
    assert index_payload["status"] == "ok"
    assert index_payload["papers"][0]["paper_id"] == "paper"

    papers_result = runner.invoke(
        app,
        [
            "papers",
            "--db-backup",
            str(db_backup),
        ],
    )

    assert papers_result.exit_code == 0
    papers_payload = json.loads(papers_result.stdout)
    assert papers_payload["paper_ids"] == ["paper"]


def test_workspace_init_index_papers_and_ask(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakeAnswerGenerator(),
    )

    workspace = tmp_path / "my-review"
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    init_result = runner.invoke(app, ["init", str(workspace), "--name", "My Review"])

    assert init_result.exit_code == 0
    init_payload = json.loads(init_result.stdout)
    assert init_payload["workspace"] == str(workspace.resolve())
    assert (workspace / "episcope.toml").exists()
    assert (workspace / "papers").is_dir()
    assert (workspace / "index").is_dir()

    index_result = runner.invoke(
        app,
        [
            "index",
            str(paper),
            "--workspace",
            str(workspace),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert index_result.exit_code == 0
    index_payload = json.loads(index_result.stdout)
    assert index_payload["workspace"] == str(workspace.resolve())
    assert index_payload["papers"][0]["paper_id"] == "paper"
    assert (workspace / "metadata.json").exists()

    papers_result = runner.invoke(app, ["papers", "--workspace", str(workspace)])

    assert papers_result.exit_code == 0
    papers_payload = json.loads(papers_result.stdout)
    assert papers_payload["paper_ids"] == ["paper"]

    ask_result = runner.invoke(
        app,
        [
            "ask",
            "Where are the data?",
            "--workspace",
            str(workspace),
        ],
    )

    assert ask_result.exit_code == 0
    ask_payload = json.loads(ask_result.stdout)
    assert ask_payload["answer"] == "The paper says the data are on Zenodo."
    assert ask_payload["source_count"] >= 1


def test_explore_path_runs_without_llm_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "explore",
            "Zenodo",
            "--path",
            str(paper),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"] is None
    assert payload["retrieval_count"] >= 1
    assert any("Zenodo" in chunk["text"] for chunk in payload["retrieved_chunks"])


def test_ask_path_generates_answer_from_local_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakeAnswerGenerator(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "ask",
            "Where are the data?",
            "--path",
            str(paper),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["answer"] == "The paper says the data are on Zenodo."
    assert payload["source_count"] >= 1


def test_version_flag_prints_version() -> None:
    import episcope

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert episcope.__version__ in result.stdout


def test_inspect_human_format_overrides_env(tmp_path) -> None:
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect", str(paper), "--format", "human"])

    assert result.exit_code == 0
    assert "Paper: paper" in result.stdout
    assert "Sections:" in result.stdout


def test_doctor_reports_ok_when_provider_key_present(monkeypatch) -> None:
    monkeypatch.setenv("EPISCOPE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = runner.invoke(app, ["doctor", "--no-probe"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    sections = {check["section"] for check in payload["checks"]}
    assert {"Environment", "LLM"} <= sections


def test_doctor_warns_when_provider_key_missing(monkeypatch) -> None:
    # A missing cloud key is a warning, not a failure: you can run keyless via
    # Ollama. doctor still succeeds (exit 0).
    monkeypatch.setenv("EPISCOPE_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor", "--no-probe"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert any(
        check["status"] == "warn" and check["label"] == "OPENAI_API_KEY"
        for check in payload["checks"]
    )


def test_classify_file_uses_transient_local_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakeClassifierGenerator(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "classify",
            "--file",
            str(paper),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["paper_id"] == "paper"
    assert payload["result"]["classification"] == ["open"]


def _study_design_spec() -> dict:
    return {
        "key": "study_design",
        "kind": "classifier",
        "label": "Study design",
        "multi_label": False,
        "default_label": "unclear",
        "labels": [
            {
                "code": "cohort",
                "name": "Cohort",
                "definition": "Follows groups over time.",
                "examples": ["We followed a cohort of exposed individuals."],
            },
            {
                "code": "unclear",
                "name": "Unclear",
                "definition": "Not enough information.",
            },
        ],
    }


class _FakeStudyDesignGenerator:
    model_id = "fake-study-design"

    def generate(self, contexts, **kwargs):
        return Provenance(
            answer=(
                '{"reasoning": "a cohort followed over time", '
                '"confidence": 0.9, "classification": ["cohort"]}'
            ),
            evidences=[],
        )


def test_classify_with_task_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakeStudyDesignGenerator(),
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nWe followed a cohort of exposed individuals for a year.",
        encoding="utf-8",
    )
    spec_file = tmp_path / "study_design.json"
    spec_file.write_text(json.dumps(_study_design_spec()), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "classify",
            "--file",
            str(paper),
            "--task-file",
            str(spec_file),
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["paper_id"] == "paper"
    assert payload["result"]["classification"] == ["cohort"]


def test_classify_with_workspace_task(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakeStudyDesignGenerator(),
    )

    workspace = tmp_path / "ws"
    init_result = runner.invoke(app, ["init", str(workspace)])
    assert init_result.exit_code == 0
    tasks_dir = workspace / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    (tasks_dir / "study_design.json").write_text(
        json.dumps(_study_design_spec()), encoding="utf-8"
    )

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nWe followed a cohort of exposed individuals for a year.",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "classify",
            "--file",
            str(paper),
            "--workspace",
            str(workspace),
            "--classifier-kind",
            "study_design",
            "--embed-model",
            "fake-embedder/1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["result"]["classification"] == ["cohort"]


def test_tasks_command_lists_builtin_and_declarative(tmp_path) -> None:
    workspace = tmp_path / "ws"
    runner.invoke(app, ["init", str(workspace)])
    tasks_dir = workspace / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    (tasks_dir / "study_design.json").write_text(
        json.dumps(_study_design_spec()), encoding="utf-8"
    )

    result = runner.invoke(app, ["tasks", "--workspace", str(workspace)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    keys = {entry["key"] for entry in payload["classifiers"]}
    assert "data_accessibility" in keys
    assert "study_design" in keys
    declared = next(
        entry for entry in payload["classifiers"] if entry["key"] == "study_design"
    )
    assert declared["source"] == "declarative"


# ---------------------------------------------------------------------------
# Keyless Ollama fallback for generation
# ---------------------------------------------------------------------------
def test_resolve_generator_passthrough_for_explicit_provider(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.setattr(cli, "_build_generator", lambda p, m, t: ("built", p, m))
    result = cli._resolve_generator(cli.LLMProvider.nollm, None, 0.0, task="ask")
    assert result == ("built", cli.LLMProvider.nollm, None)


def test_resolve_generator_uses_gemini_when_key_present(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setattr(cli, "_build_generator", lambda p, m, t: ("built", p, m))
    result = cli._resolve_generator(cli.LLMProvider.gemini, None, 0.0, task="ask")
    assert result[1] == cli.LLMProvider.gemini


def test_resolve_generator_ask_falls_back_to_small_ollama_model(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_ollama_models", lambda: [cli._OLLAMA_SMALL_MODEL])
    monkeypatch.setattr(cli, "_build_generator", lambda p, m, t: ("ollama", p, m))
    result = cli._resolve_generator(cli.LLMProvider.gemini, None, 0.0, task="ask")
    assert result == ("ollama", cli.LLMProvider.ollama, cli._OLLAMA_SMALL_MODEL)


def test_resolve_generator_workflow_uses_strong_default(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_ollama_models", lambda: [cli._OLLAMA_STRONG_MODEL])
    monkeypatch.setattr(cli, "_build_generator", lambda p, m, t: ("ollama", p, m))
    result = cli._resolve_generator(cli.LLMProvider.gemini, None, 0.0, task="classify")
    assert result[2] == cli._OLLAMA_STRONG_MODEL


def test_resolve_generator_errors_without_key_or_ollama(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_ollama_models", lambda: None)
    with pytest.raises(ValueError, match="ollama pull"):
        cli._resolve_generator(cli.LLMProvider.gemini, None, 0.0, task="ask")


def test_resolve_generator_errors_when_model_not_pulled(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_ollama_models", lambda: ["some-other-model"])
    with pytest.raises(ValueError, match="not pulled"):
        cli._resolve_generator(cli.LLMProvider.gemini, None, 0.0, task="ask")


def test_ask_without_key_or_ollama_shows_install_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("episcope.episcope._ollama_models", lambda: None)

    paper = tmp_path / "paper.txt"
    paper.write_text("The dataset is on Zenodo.", encoding="utf-8")

    result = runner.invoke(app, ["ask", "Where are the data?", "--path", str(paper)])

    assert result.exit_code == 1
    assert "Ollama" in result.output
    assert "ollama pull" in result.output
