from __future__ import annotations

import json
import types

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


class _FakePrecisionMinerGenerator:
    model_id = "fake-generator"

    def generate(self, contexts, **kwargs):
        return Provenance(
            answer="""
            {
              "description": "Data sources mentioned in the paper.",
              "items": [
                {"name": "Zenodo", "url": null, "explanation": "Dataset is hosted here."}
              ]
            }
            """,
            evidences=[],
        )


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


# ---------------------------------------------------------------------------
# --quality preset resolution
# ---------------------------------------------------------------------------
def test_resolve_value_precedence_explicit_beats_quality_beats_workspace() -> None:
    from episcope import episcope as cli

    # Explicit flag (current != default) always wins.
    assert cli._resolve_value(300, 600, cli.QualityPreset.fast, 800, object(), 900) == 300
    # No explicit flag: quality preset wins over workspace.
    assert cli._resolve_value(600, 600, cli.QualityPreset.fast, 800, object(), 900) == 800
    # No explicit flag, no quality: workspace wins over default.
    assert cli._resolve_value(600, 600, None, None, object(), 900) == 900
    # Nothing set: default.
    assert cli._resolve_value(600, 600, None, None, None, None) == 600


def test_ask_path_accepts_quality_preset(tmp_path, monkeypatch) -> None:
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
        app, ["ask", "Where are the data?", "--path", str(paper), "--quality", "fast"]
    )

    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# _resolve_retrieval_mode_for_vectordb (hybrid/sparse without sparse-capable
# storage, or without the local ML stack for the sparse embedder)
# ---------------------------------------------------------------------------
def test_resolve_retrieval_mode_dense_only_passthrough() -> None:
    from episcope import episcope as cli

    fake_vectordb = types.SimpleNamespace(capabilities=lambda: {"sparse": False})
    result = cli._resolve_retrieval_mode_for_vectordb(
        fake_vectordb, cli.RetrievalMode.dense_only, False
    )
    assert result == cli.RetrievalMode.dense_only


def test_resolve_retrieval_mode_clamps_when_storage_lacks_sparse(capsys) -> None:
    from episcope import episcope as cli

    fake_vectordb = types.SimpleNamespace(capabilities=lambda: {"sparse": False})
    result = cli._resolve_retrieval_mode_for_vectordb(
        fake_vectordb, cli.RetrievalMode.hybrid, False
    )
    assert result == cli.RetrievalMode.dense_only
    assert "sparse-capable storage" in capsys.readouterr().err


def test_resolve_retrieval_mode_errors_when_explicit_and_storage_lacks_sparse() -> None:
    from episcope import episcope as cli

    fake_vectordb = types.SimpleNamespace(capabilities=lambda: {"sparse": False})
    with pytest.raises(ValueError, match="sparse-capable storage"):
        cli._resolve_retrieval_mode_for_vectordb(
            fake_vectordb, cli.RetrievalMode.hybrid, True
        )


def test_resolve_retrieval_mode_clamps_when_local_ml_missing(monkeypatch, capsys) -> None:
    from episcope import episcope as cli

    monkeypatch.setattr(cli, "_local_ml_available", lambda: False)
    fake_vectordb = types.SimpleNamespace(capabilities=lambda: {"sparse": True})
    result = cli._resolve_retrieval_mode_for_vectordb(
        fake_vectordb, cli.RetrievalMode.hybrid, False
    )
    assert result == cli.RetrievalMode.dense_only
    assert "local ML stack" in capsys.readouterr().err


def test_resolve_retrieval_mode_errors_when_explicit_and_local_ml_missing(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.setattr(cli, "_local_ml_available", lambda: False)
    fake_vectordb = types.SimpleNamespace(capabilities=lambda: {"sparse": True})
    with pytest.raises(ValueError, match="local ML stack"):
        cli._resolve_retrieval_mode_for_vectordb(
            fake_vectordb, cli.RetrievalMode.hybrid, True
        )


def test_resolve_retrieval_mode_passes_through_when_fully_satisfied() -> None:
    from episcope import episcope as cli

    fake_vectordb = types.SimpleNamespace(capabilities=lambda: {"sparse": True})
    result = cli._resolve_retrieval_mode_for_vectordb(
        fake_vectordb, cli.RetrievalMode.hybrid, True
    )
    assert result == cli.RetrievalMode.hybrid


def test_explore_workspace_with_balanced_quality_clamps_on_file_backend(
    tmp_path, monkeypatch
) -> None:
    """Regression test: the default (file-backed) workspace can't serve hybrid
    retrieval; a --quality preset asking for it should degrade gracefully
    instead of raising, matching the --path/--file clamp behavior.
    """
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    workspace = tmp_path / "ws"
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    assert runner.invoke(app, ["init", str(workspace)]).exit_code == 0
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
    assert index_result.exit_code == 0, index_result.output

    result = runner.invoke(
        app,
        ["explore", "Zenodo", "--workspace", str(workspace), "--quality", "balanced"],
    )

    assert result.exit_code == 0, result.output
    assert "sparse-capable storage" in result.output
    payload = json.loads(result.stdout)
    assert payload["retrieval_count"] >= 1


def test_explore_path_with_balanced_quality_does_not_raise(tmp_path, monkeypatch) -> None:
    """Regression test: --quality balanced/accurate resolves retrieval_mode to
    'hybrid', which --path's transient (dense-only-only) index can't serve.
    This used to raise "--path mode only supports --retrieval-mode dense-only."
    even though the user never asked for hybrid retrieval themselves - a
    preset should never surface an error the user didn't ask for.
    """
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    for quality in ("balanced", "accurate"):
        result = runner.invoke(
            app,
            ["explore", "Zenodo", "--path", str(paper), "--quality", quality],
        )
        assert result.exit_code == 0, (quality, result.output)
        # The dropped retrieval-mode effect must be surfaced, not silent.
        assert "only supports dense-only retrieval" in result.output


def test_explore_path_without_quality_prints_no_retrieval_mode_note(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    paper = tmp_path / "paper.txt"
    paper.write_text("A short title\n\nSome content.", encoding="utf-8")

    result = runner.invoke(app, ["explore", "content", "--path", str(paper)])

    assert result.exit_code == 0, result.output
    assert "only supports dense-only retrieval" not in result.output


def test_explore_path_with_explicit_hybrid_still_errors(tmp_path, monkeypatch) -> None:
    """An explicit --retrieval-mode conflicting with --path must still be a
    clear, actionable error - only preset-driven mismatches are silenced.
    """
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    paper = tmp_path / "paper.txt"
    paper.write_text("A short title\n\nSome content.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["explore", "content", "--path", str(paper), "--retrieval-mode", "hybrid"],
    )

    assert result.exit_code == 1
    assert "dense-only" in result.output


def test_classify_file_with_quality_preset_does_not_raise(tmp_path, monkeypatch) -> None:
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
        ["classify", "--file", str(paper), "--quality", "accurate"],
    )

    assert result.exit_code == 0, result.output
    assert "only supports dense-only retrieval" in result.output


def test_precision_miner_file_with_quality_preset_does_not_raise(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.setattr(
        "episcope.episcope._build_generator",
        lambda provider, model, temperature: _FakePrecisionMinerGenerator(),
    )
    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    for quality in ("fast", "balanced", "accurate"):
        result = runner.invoke(
            app,
            ["precision-miner", "--file", str(paper), "--quality", quality],
        )
        assert result.exit_code == 0, (quality, result.output)
        # "fast" resolves to dense_only already, so no retrieval-mode
        # mismatch note; "balanced"/"accurate" resolve to hybrid, which
        # --file mode can't serve, so the drop must be surfaced.
        has_note = "only supports dense-only retrieval" in result.output
        assert has_note == (quality != "fast"), (quality, result.output)


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


def test_doctor_checks_anthropic_key(monkeypatch) -> None:
    monkeypatch.setenv("EPISCOPE_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(app, ["doctor", "--no-probe"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert any(
        check["status"] == "warn" and check["label"] == "ANTHROPIC_API_KEY"
        for check in payload["checks"]
    )


# ---------------------------------------------------------------------------
# quickstart
# ---------------------------------------------------------------------------
def _no_services_reachable(url: str, *, timeout: float = 3.0):
    return False, None, "connection refused"


@pytest.fixture
def _isolate_quickstart_probes(monkeypatch):
    """Keep quickstart tests fast/offline regardless of the real environment.

    quickstart has no --no-probe escape hatch (checking connectivity is the
    point), so without this a developer's real ambient MONGO_URI/mongo_uri
    (dotenv legacy name, see settings.py) would make these tests attempt a
    real, slow MongoDB connection.
    """
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("mongo_uri", raising=False)
    monkeypatch.setattr("episcope.episcope._probe_http", _no_services_reachable)
    monkeypatch.setattr(
        "episcope.episcope._check_mongo", lambda uri: ("skip", "not checked", "")
    )
    return monkeypatch


def test_quickstart_ollama_writes_provider_and_reports_ready(
    tmp_path, monkeypatch, _isolate_quickstart_probes
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "episcope.episcope._probe_http",
        lambda url, **_: (True, 200, "") if "11434" in url else _no_services_reachable(url),
    )

    result = runner.invoke(app, ["quickstart"], input="5\n")

    assert result.exit_code == 0, result.output
    assert "You're ready" in result.output
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "ollama" in env_text


def test_quickstart_gemini_writes_key_to_env(
    tmp_path, monkeypatch, _isolate_quickstart_probes
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, ["quickstart"], input="1\nfake-key-123\n")

    assert result.exit_code == 0, result.output
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "gemini" in env_text
    assert "fake-key-123" in env_text
    # LLM key presence is an "ok" check even though GROBID/Qdrant/Mongo are
    # unreachable - those are warnings, not failures, for the local CLI path.
    assert "All required checks passed" in result.output


def test_quickstart_leaves_existing_key_untouched_when_blank(
    tmp_path, monkeypatch, _isolate_quickstart_probes
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=already-set-key\n", encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "already-set-key")

    result = runner.invoke(app, ["quickstart"], input="1\n\n")

    assert result.exit_code == 0, result.output
    assert "leave blank to keep it" in result.output
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "already-set-key" in env_text


def test_quickstart_invalid_choice_aborts(
    tmp_path, monkeypatch, _isolate_quickstart_probes
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["quickstart"], input="99\n")

    assert result.exit_code == 1
    assert "Invalid choice" in result.output


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


def test_classify_unknown_kind_fails_before_parsing_file(tmp_path, monkeypatch) -> None:
    embedder_calls: list[str] = []
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: embedder_calls.append(model_name) or _FakeEmbedder(),
    )
    paper = tmp_path / "paper.txt"
    paper.write_text("A short title\n\nSome content.", encoding="utf-8")

    result = runner.invoke(
        app, ["classify", "--file", str(paper), "--classifier-kind", "not_a_real_kind"]
    )

    assert result.exit_code == 1
    assert "Unknown classifier kind" in result.output
    # The (potentially slow) PDF-parse + embedding step must not run for a
    # request that was always going to fail on an unknown kind.
    assert embedder_calls == []


def test_precision_miner_unknown_kind_fails_before_parsing_file(
    tmp_path, monkeypatch
) -> None:
    embedder_calls: list[str] = []
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: embedder_calls.append(model_name) or _FakeEmbedder(),
    )
    paper = tmp_path / "paper.txt"
    paper.write_text("A short title\n\nSome content.", encoding="utf-8")

    result = runner.invoke(
        app,
        ["precision-miner", "--file", str(paper), "--miner-kind", "not_a_real_kind"],
    )

    assert result.exit_code == 1
    assert "Unknown precision-miner kind" in result.output
    assert embedder_calls == []


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
# Provider dispatch
# ---------------------------------------------------------------------------
def test_build_generator_dispatches_anthropic_provider(monkeypatch) -> None:
    from episcope import episcope as cli
    from episcope.clients import AnthropicClient

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    generator = cli._build_generator(cli.LLMProvider.anthropic, "claude-sonnet-5", 0.0)
    assert isinstance(generator.client, AnthropicClient)
    assert generator.model == "claude-sonnet-5"


def test_build_generator_anthropic_requires_explicit_model(monkeypatch) -> None:
    from episcope import episcope as cli

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(ValueError, match="--llm-model is required"):
        cli._build_generator(cli.LLMProvider.anthropic, None, 0.0)


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


def test_resolve_generator_ignores_gemini_model_name_on_ollama_fallback(
    monkeypatch,
) -> None:
    """A workspace's configured Gemini model must not leak into the Ollama
    fallback lookup (regression: previously caused a spurious 'model not
    pulled' error even though the Ollama default was available)."""
    from episcope import episcope as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_ollama_models", lambda: [cli._OLLAMA_SMALL_MODEL])
    monkeypatch.setattr(cli, "_build_generator", lambda p, m, t: ("ollama", p, m))

    result = cli._resolve_generator(
        cli.LLMProvider.gemini, "gemini-2.5-flash", 0.0, task="ask"
    )

    assert result == ("ollama", cli.LLMProvider.ollama, cli._OLLAMA_SMALL_MODEL)


def test_resolve_generator_ignores_gemini_model_name_for_workflow_task(
    monkeypatch,
) -> None:
    from episcope import episcope as cli

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_ollama_models", lambda: [cli._OLLAMA_STRONG_MODEL])
    monkeypatch.setattr(cli, "_build_generator", lambda p, m, t: ("ollama", p, m))

    result = cli._resolve_generator(
        cli.LLMProvider.gemini, "gemini-2.5-flash", 0.0, task="classify"
    )

    assert result[2] == cli._OLLAMA_STRONG_MODEL


def test_ask_without_key_or_ollama_shows_extractive_fallback(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "episcope.episcope.EmbedderFactory.get_embedder",
        lambda model_name, **_: _FakeEmbedder(),
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("episcope.episcope._ollama_models", lambda: None)

    paper = tmp_path / "paper.txt"
    paper.write_text(
        "A short title\n\nThe dataset is publicly available on Zenodo.",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["ask", "Where are the data?", "--path", str(paper)])

    assert result.exit_code == 0
    assert "quickstart" in result.stderr
    payload = json.loads(result.stdout)
    assert payload["answer"] is None
    assert payload["sources"]
    assert "Zenodo" in payload["sources"][0]["text"]


# ---------------------------------------------------------------------------
# studio
# ---------------------------------------------------------------------------
class _FakeStudioProc:
    def __init__(self, args, env=None):
        self.args = args
        self.env = env or {}
        self.returncode = None
        self.terminate_called = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True

    def wait(self, timeout=None):
        self.returncode = 0
        return self.returncode


@pytest.fixture
def _studio_popen(monkeypatch):
    # `studio` requires the `server`/`ui` extras (uvicorn, streamlit), which
    # aren't part of the base package dependencies - skip rather than fail
    # in environments (e.g. the base tox testenv) that only install those.
    pytest.importorskip("uvicorn")
    pytest.importorskip("streamlit")
    created: list[_FakeStudioProc] = []

    def fake_popen(args, env=None):
        proc = _FakeStudioProc(args, env=env)
        created.append(proc)
        return proc

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    # Simulate Ctrl-C on the first poll-loop tick so studio's cleanup path runs
    # instead of looping forever waiting on the (fake) subprocesses.
    monkeypatch.setattr(
        "time.sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    return created


def test_studio_launches_api_and_ui_and_cleans_up_on_interrupt(
    tmp_path, monkeypatch, _studio_popen
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("MONGO_URI", raising=False)

    result = runner.invoke(
        app, ["studio", "--api-port", "9199", "--ui-port", "9198"]
    )

    assert result.exit_code == 0, result.output
    assert len(_studio_popen) == 2
    api_proc, ui_proc = _studio_popen
    assert "uvicorn" in api_proc.args
    assert "episcope.api:app" in api_proc.args
    assert "streamlit" in ui_proc.args
    assert api_proc.terminate_called
    assert ui_proc.terminate_called
    # No papers indexed in this fresh tmp_path: the empty-index warning fires.
    assert "No papers are indexed yet" in result.output


def test_studio_sigterm_is_routed_through_the_same_cleanup_as_ctrl_c(
    tmp_path, monkeypatch, _studio_popen
) -> None:
    # Python's default SIGTERM handling bypasses except/finally entirely, so
    # a plain `kill`/process-manager stop would otherwise orphan the API/UI
    # subprocesses - studio must install a handler that converts SIGTERM
    # into the same KeyboardInterrupt-based cleanup path Ctrl-C already uses.
    import signal

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("MONGO_URI", raising=False)

    registered_handlers: dict[int, object] = {}
    real_signal = signal.signal

    def capturing_signal(sig, handler):
        registered_handlers[sig] = handler
        return real_signal(sig, handler) if sig != signal.SIGTERM else None

    monkeypatch.setattr(signal, "signal", capturing_signal)

    result = runner.invoke(app, ["studio", "--api-port", "9193", "--ui-port", "9192"])

    assert result.exit_code == 0, result.output
    assert signal.SIGTERM in registered_handlers
    with pytest.raises(KeyboardInterrupt):
        registered_handlers[signal.SIGTERM](signal.SIGTERM, None)


def test_studio_points_at_workspace_index_when_qdrant_unset(
    tmp_path, monkeypatch, _studio_popen
) -> None:
    from episcope.workspace import create_workspace

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("MONGO_URI", raising=False)
    ws = create_workspace(tmp_path / "my-review")

    result = runner.invoke(
        app,
        ["studio", "--workspace", str(ws.root), "--api-port", "9197", "--ui-port", "9196"],
    )

    assert result.exit_code == 0, result.output
    api_proc, _ = _studio_popen
    assert api_proc.env.get("EPISCOPE_LOCAL_INDEX_DIR") == str(
        ws.resolve_path(ws.index_dir)
    )
    assert api_proc.env.get("EPISCOPE_LOCAL_METADATA_BACKUP") == str(
        ws.resolve_path(ws.metadata_path)
    )


def test_studio_does_not_override_explicit_qdrant_url(
    tmp_path, monkeypatch, _studio_popen
) -> None:
    from episcope.workspace import create_workspace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QDRANT_URL", "http://external-qdrant:6333")
    ws = create_workspace(tmp_path / "my-review")

    result = runner.invoke(
        app,
        ["studio", "--workspace", str(ws.root), "--api-port", "9195", "--ui-port", "9194"],
    )

    assert result.exit_code == 0, result.output
    api_proc, _ = _studio_popen
    assert "EPISCOPE_LOCAL_INDEX_DIR" not in api_proc.env
