from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = "Run simple majority and prototype-similarity classification baselines."


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="simple", description=DESCRIPTION)


if __name__ == "__main__":
    main()
