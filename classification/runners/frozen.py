from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = "Run frozen transformer embedding classification baselines (no fine-tuning)."


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="frozen", description=DESCRIPTION)


if __name__ == "__main__":
    main()
