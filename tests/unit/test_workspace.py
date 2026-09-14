from __future__ import annotations

from dataclasses import replace

from epilens.workspace import (
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


def test_workspace_is_discovered_and_updated(tmp_path) -> None:
    root = tmp_path / "review"
    created = create_workspace(root, name="Review")

    workspace = find_workspace(root / "nested")

    # find_workspace accepts a not-yet-created descendant and walks upward.
    assert workspace is not None
    assert workspace.name == "Review"
    assert workspace.config_path == created.config_path

    write_workspace_config(replace(workspace, name="Updated review"))
    assert load_workspace(root).name == "Updated review"


def test_state_root_uses_epilens_directory(tmp_path) -> None:
    assert state_root(tmp_path) == tmp_path / ".epilens"
