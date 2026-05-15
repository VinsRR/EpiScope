from __future__ import annotations

from collections.abc import Sequence

from .families import run_family

DESCRIPTION = (
    "Run unsupervised corpus-structure baselines (LSA, LDA, NMF, PLSA, "
    "BERTopic, Top2Vec). These are not competitive classifiers — they audit "
    "whether the classification categories have natural clustering signal in "
    "the corpus."
)


def main(argv: Sequence[str] | None = None) -> None:
    run_family(argv, family="unsupervised", description=DESCRIPTION)


if __name__ == "__main__":
    main()
