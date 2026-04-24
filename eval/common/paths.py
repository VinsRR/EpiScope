from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .tasks import CANONICAL_TASK_NAMES


def discover_result_files(run_roots: Iterable[str | Path], pattern: str = "final_*.tsv") -> list[Path]:
    files: list[Path] = []
    for root in run_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        files.extend(sorted(root_path.rglob(pattern)))
    return sorted(set(files))


def parse_result_path(path: str | Path) -> dict[str, str]:
    """Infer task/model/temperature from a run file path."""
    path = Path(path)
    parts = path.parts

    task_index = None
    task_value = None
    for idx, part in enumerate(parts):
        if part in CANONICAL_TASK_NAMES:
            task_index = idx
            task_value = CANONICAL_TASK_NAMES[part]
            break
    if task_index is None or task_value is None:
        raise ValueError(f"Could not infer task from path: {path}")

    if len(parts) <= task_index + 1:
        raise ValueError(f"Could not infer model from path: {path}")
    model = parts[task_index + 1]

    temperature = "unknown"
    for part in parts[task_index + 2 :]:
        match = re.match(r"temperature_([0-9]+(?:\.[0-9]+)?)", part)
        if match:
            temperature = match.group(1)
            break

    return {
        "task": task_value,
        "model": model,
        "temperature": temperature,
        "source_file": str(path),
    }
