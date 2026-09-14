#!/usr/bin/env python3
"""Remove generated packaging directories before building a release."""

from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def clean_release_artifacts(root: Path = PROJECT_ROOT) -> list[Path]:
    """Delete only known, reproducible build outputs below *root*."""
    targets = [root / "build", root / "dist", *root.glob("src/*.egg-info")]
    removed: list[Path] = []

    for path in targets:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            continue
        removed.append(path)

    return removed


def main() -> None:
    removed = clean_release_artifacts()
    if not removed:
        print("No stale release artifacts found.")
        return
    for path in removed:
        print(f"Removed {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
