from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Sequence

from episcope.rag.generation.base import Generator
from episcope.rag.provenance import Provenance
from episcope.schemas import SearchResult


def make_search_result(
    *,
    id: str,
    paper_id: str,
    text: str,
    score: float,
    section_type: str = "Methods",
    title: str = "",
    source: str = "test",
) -> SearchResult:
    return SearchResult(
        id=id,
        paper_id=paper_id,
        text=text,
        section_type=section_type,
        title=title,
        similarity_score=score,
        rank_score=score,
        source=source,
        artifacts={},
    )


class StaticRetriever:
    """Simple fake retriever that returns predefined results per query."""

    def __init__(self, responses_by_query: Mapping[str, Sequence[SearchResult]] | None = None) -> None:
        self.responses_by_query = {
            query: list(results) for query, results in (responses_by_query or {}).items()
        }
        self.calls: List[Dict[str, object]] = []
        self.index_version = "test-index"

    def retrieve_by_paper(
        self,
        query: str,
        paper_id: str,
        *,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filter: Dict[str, object] | None = None,
    ) -> List[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "paper_id": paper_id,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "filter": filter,
            }
        )
        return [deepcopy(result) for result in self.responses_by_query.get(query, [])][:top_k]


class SequenceGenerator(Generator):
    """Fake generator that returns a sequence of canned answers."""

    def __init__(self, answers: Iterable[str]) -> None:
        self.answers = list(answers)
        self.calls: List[Dict[str, object]] = []

    def generate(self, *args, **kwargs) -> Provenance:
        self.calls.append({"args": args, "kwargs": kwargs})
        if not self.answers:
            raise RuntimeError("No canned answers left in SequenceGenerator.")
        return Provenance(answer=self.answers.pop(0), evidences=[])


class RecordingCrossEncoderReranker:
    """Fake cross-encoder reranker with query-specific text scores."""

    def __init__(self, scores_by_query: Mapping[str, Mapping[str, float]]) -> None:
        self.scores_by_query = {
            query: dict(scores) for query, scores in scores_by_query.items()
        }
        self.calls: List[Dict[str, object]] = []

    def rerank(self, query: str, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        self.calls.append({"query": query, "texts": [result.text for result in results], "top_k": top_k})
        scores = self.scores_by_query.get(query, {})

        rescored: List[SearchResult] = []
        for result in results:
            updated = deepcopy(result)
            updated.rank_score = scores.get(result.text, updated.rank_score)
            rescored.append(updated)

        rescored.sort(key=lambda item: item.rank_score, reverse=True)
        return rescored[:top_k]
