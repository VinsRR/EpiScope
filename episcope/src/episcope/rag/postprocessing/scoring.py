from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from episcope.schemas import SearchResult


class Scorer(ABC):
    """Callable that assigns a float score to a SearchResult.

    Scorers are stateless and composable. Combine them with + to produce
    an AdditiveScorer, or subclass for custom behaviour.
    """

    @abstractmethod
    def __call__(self, result: SearchResult) -> float:
        ...

    def __add__(self, other: "Scorer") -> "AdditiveScorer":
        return AdditiveScorer([self, other])


# ---------------------------------------------------------------------------
# Concrete scorers
# ---------------------------------------------------------------------------

class SimilarityScorer(Scorer):
    """Scales the result's raw embedding similarity score."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, result: SearchResult) -> float:
        return result.similarity_score * self.weight


class ArtifactPresenceScorer(Scorer):
    """Rewards results that contain citable artifacts (URLs, DOIs, accessions).

    Weights are relative to each other and to the other scorers in the
    pipeline — tune them as a group, not in isolation.
    Default ordering (url > doi > accession) reflects how actionable each
    artifact type is for data-availability verification.
    """

    def __init__(
        self,
        url_weight: float = 1.0,
        doi_weight: float = 0.9,
        accession_weight: float = 0.75,
    ):
        self.url_weight = url_weight
        self.doi_weight = doi_weight
        self.accession_weight = accession_weight

    def __call__(self, result: SearchResult) -> float:
        score = 0.0
        if result.artifacts.get("urls"):
            score += self.url_weight
        if result.artifacts.get("dois"):
            score += self.doi_weight
        if result.artifacts.get("accessions"):
            score += self.accession_weight
        return score


class AvailabilityTermScorer(Scorer):
    """Rewards results whose text contains data-availability vocabulary.

    The raw `availability_score` from the artifact extractor is a term-
    frequency count; `weight` controls how much it influences the ranking.
    """

    def __init__(self, weight: float = 0.5):
        self.weight = weight

    def __call__(self, result: SearchResult) -> float:
        return result.artifacts.get("availability_score", 0) * self.weight


class AdditiveScorer(Scorer):
    """Sums the outputs of multiple scorers.

    Prefer building via the + operator on individual scorers rather than
    constructing this directly.
    """

    def __init__(self, scorers: List[Scorer]):
        self.scorers = scorers

    def __call__(self, result: SearchResult) -> float:
        return sum(s(result) for s in self.scorers)

    def __add__(self, other: Scorer) -> "AdditiveScorer":
        return AdditiveScorer(self.scorers + [other])


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def default_scorer() -> AdditiveScorer:
    """Balanced scorer for general retrieval."""
    return SimilarityScorer(1.0) + ArtifactPresenceScorer() + AvailabilityTermScorer()


def data_availability_scorer() -> AdditiveScorer:
    """Upweights artifact presence and availability vocabulary.

    Tuned for the data-availability classification task where the key
    signal (a repository link, DOI, or accession number) is highly
    localised and may not score highly on embedding similarity alone.
    """
    return (
        SimilarityScorer(0.5)
        + ArtifactPresenceScorer(url_weight=2.0, doi_weight=1.8, accession_weight=1.5)
        + AvailabilityTermScorer(weight=1.2)
    )
