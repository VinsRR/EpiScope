from __future__ import annotations

from pathlib import Path

import pytest

from eval.common.config import load_json_config


def test_load_json_config_reads_expected_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"doc_id":"doc-1","testset_size":5}', encoding="utf-8")

    loaded = load_json_config(config_path, allowed_keys={"doc_id", "testset_size"})

    assert loaded == {"doc_id": "doc-1", "testset_size": 5}


def test_load_json_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"doc_id":"doc-1","unknown":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config key"):
        load_json_config(config_path, allowed_keys={"doc_id"})
