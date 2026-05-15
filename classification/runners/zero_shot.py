from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = (
    "Run zero-shot classification baselines (no labeled training examples): "
    "majority prior, prototype similarity (TF-IDF / sentence embeddings), "
    "BERTopic guided, and Top2Vec contextual."
)


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="zero_shot", description=DESCRIPTION)


if __name__ == "__main__":
    main()
