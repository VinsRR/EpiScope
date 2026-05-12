from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = "Run the zero-shot metadata LLM classification baseline."


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="llm", description=DESCRIPTION)


if __name__ == "__main__":
    main()
