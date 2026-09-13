from __future__ import annotations

from dataclasses import replace

from epilens.settings import env
from epilens.workspace import (
    LEGACY_WORKSPACE_FILE,
    WORKSPACE_FILE,
    create_workspace,
    find_workspace,
    load_workspace,
    state_root,
    workspace_lock_path,
    write_workspace_config,
)


def test_new_workspace_uses_epilens_names_and_lightweight_parser(tmp_path) -> None:
    workspace = create_workspace(tmp_path / "review")

    assert workspace.config_path.name == WORKSPACE_FILE
    assert workspace.loader == "pdfminer"
    assert workspace.config_path.is_file()
    assert workspace_lock_path(workspace).name == ".epilens.lock"


def test_legacy_workspace_is_discovered_and_written_in_place(tmp_path) -> None:
    root = tmp_path / "review"
    root.mkdir()
    legacy_config = root / LEGACY_WORKSPACE_FILE
    legacy_config.write_text(
        '[workspace]\nname = "Legacy review"\n\n[indexing]\nloader = "unstructured"\n',
        encoding="utf-8",
    )

    workspace = find_workspace(root / "nested")

    # find_workspace accepts a not-yet-created descendant and walks upward.
    assert workspace is not None
    assert workspace.name == "Legacy review"
    assert workspace.config_path == legacy_config
    assert workspace_lock_path(workspace).name == ".episcope.lock"

    write_workspace_config(replace(workspace, name="Updated review"))
    assert load_workspace(root).name == "Updated review"
    assert not (root / WORKSPACE_FILE).exists()


def test_new_environment_name_wins_over_legacy_name(monkeypatch) -> None:
    monkeypatch.delenv("EPILENS_LLM_MODEL", raising=False)
    monkeypatch.setenv("EPISCOPE_LLM_MODEL", "legacy-model")
    assert env("EPILENS_LLM_MODEL") == "legacy-model"

    monkeypatch.setenv("EPILENS_LLM_MODEL", "new-model")
    assert env("EPILENS_LLM_MODEL") == "new-model"


def test_state_root_reuses_existing_legacy_directory(tmp_path) -> None:
    legacy = tmp_path / ".episcope"
    legacy.mkdir()
    assert state_root(tmp_path) == legacy

    current = tmp_path / ".epilens"
    current.mkdir()
    assert state_root(tmp_path) == current
