from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = "Run unsupervised/guided topic-model classification baselines."


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="topic", description=DESCRIPTION)


if __name__ == "__main__":
    main()
