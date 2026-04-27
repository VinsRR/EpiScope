from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path, *, allowed_keys: set[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config at {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a single JSON object.")

    if allowed_keys is not None:
        unknown = sorted(set(data) - allowed_keys)
        if unknown:
            raise ValueError(
                f"Unknown config key(s) in {config_path}: {unknown}. "
                f"Allowed keys are: {sorted(allowed_keys)}"
            )

    return data
