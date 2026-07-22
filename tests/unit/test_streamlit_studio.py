from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from episcope.ui.api_client import StudioApiClient  # noqa: E402


def test_studio_home_renders_with_no_workspaces(monkeypatch) -> None:
    monkeypatch.setattr(
        StudioApiClient,
        "workspaces",
        lambda self: {"root": "/tmp/workspaces", "active_workspace_id": None, "items": []},
    )
    monkeypatch.setattr(
        StudioApiClient,
        "health",
        lambda self: {
            "checks": {
                "llm_provider": "gemini",
                "llm_api_key_configured": False,
            }
        },
    )
    app_path = Path(__file__).resolve().parents[2] / "src" / "episcope" / "ui" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=15)
    assert not app.exception
    assert any("EpiScope Studio" in title.value for title in app.title)
