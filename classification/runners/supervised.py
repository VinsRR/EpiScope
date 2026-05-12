from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = "Run supervised classification baselines with k-fold or leave-one-out CV."


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="supervised", description=DESCRIPTION)


if __name__ == "__main__":
    main()
