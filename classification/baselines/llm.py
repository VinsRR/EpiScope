from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any, List, Optional

from episcope.clients import GeminiClient, OllamaClient, OpenAIClient, OpenRouterClient
from episcope.rag.generation.llm_generator import LLMGenerator
from episcope.schemas import PaperMetadata as _PaperMetadata
from episcope.schemas import SearchResult
from classification.baselines.common import (
    BaselinePrediction,
    classifier_config,
)
from episcope.workflows.classification.parsing import ClassificationResponseParser
from episcope.workflows.classification.prompting import ClassificationPromptBuilder
from episcope.workflows.classification.runner import ClassificationRunner

logger = logging.getLogger(__name__)


def build_llm_generator(
    *,
    provider: str,
    model: str,
    temperature: float = 0.0,
) -> LLMGenerator:
    if provider == "gemini":
        client = GeminiClient()
    elif provider == "openrouter":
        client = OpenRouterClient()
    elif provider == "openai":
        client = OpenAIClient()
    elif provider == "ollama":
        client = OllamaClient()
    else:
        raise ValueError(
            f"Unknown provider={provider!r}. Use one of gemini, openrouter, openai, ollama."
        )
    return LLMGenerator(client=client, model=model, temperature=temperature)


class MetadataOnlyLLMBaseline:
    """Zero-shot LLM baseline using metadata and the label protocol, no retrieval."""

    name = "metadata_llm"
    # Do not skip papers that lack body text — we only need the metadata fields.
    skip_if_no_text = False

    def __init__(
        self,
        *,
        classifier_kind: str,
        generator: LLMGenerator,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.config = classifier_config(classifier_kind)
        self.prompt_builder = ClassificationPromptBuilder(self.config)
        self.response_parser = ClassificationResponseParser(self.config)
        self.runner = ClassificationRunner(
            generator=generator,
            config=self.config,
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
        )

    def predict(
        self,
        paper_id: str,
        metadata: _PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        # Use the raw metadata fetched from Mongo (title + real abstract + keywords)
        # rather than the enriched version that substitutes full body text into the
        # abstract field. This keeps the baseline strictly metadata-only so it is a
        # clean ablation of the RAG retrieval contribution.
        raw = (record or {}).get("_metadata")
        source = raw if isinstance(raw, _PaperMetadata) else metadata
        attempt = self.runner.run(source, chunks=[], expect_no_chunks=True)
        return BaselinePrediction(
            result=attempt.result,
            raw_response=attempt.raw_response,
            prompt_messages=attempt.messages,
        )


def _payload_to_search_result(payload: Mapping[str, Any]) -> SearchResult:
    """Construct a SearchResult from a Qdrant payload dict.

    Mirrors the field handling in BaseRetriever._to_search_result without
    requiring a retriever instance.
    """
    return SearchResult(
        id=str(payload.get("id", "")),
        paper_id=str(payload.get("paper_id", "")),
        text=str(payload.get("text", "")),
        section_type=str(payload.get("section_type", "other")),
        section_title=str(payload.get("section_title", payload.get("title", ""))),
        title=str(payload.get("title", payload.get("section_title", ""))),
        is_metadata=bool(payload.get("is_metadata", False)),
        similarity_score=0.0,
        source="random",
        rank_score=0.0,
    )


def load_body_chunks_from_qdrant(qdrant_db: Any, paper_id: str) -> List[SearchResult]:
    """Fetch all non-metadata chunks indexed for a paper.

    The random-chunk LLM baseline samples from the same chunk distribution the
    real RAG retriever sees. We exclude chunks flagged as metadata so the
    sample reflects body content; title/abstract reach the prompt via the
    PaperMetadata path, just as in the full pipeline.
    """
    try:
        payloads = qdrant_db.get_points(namespace=str(paper_id))
    except Exception as exc:
        logger.warning("Qdrant chunk fetch failed for %s: %s", paper_id, exc)
        return []
    chunks: List[SearchResult] = []
    for payload in payloads or ():
        if not isinstance(payload, Mapping):
            continue
        if bool(payload.get("is_metadata", False)):
            continue
        text = str(payload.get("text", "") or "").strip()
        if not text:
            continue
        chunks.append(_payload_to_search_result(payload))
    return chunks


def _sample_random_chunks(
    chunks: Sequence[SearchResult],
    *,
    k: int,
    paper_id: str,
    seed: int,
    repeat_idx: int = 1,
) -> List[SearchResult]:
    """Sample up to k chunks uniformly at random with a per-paper-per-repeat seed.

    The seed depends on (global seed, paper_id, repeat_idx) so:
      - repeats are reproducible within a configuration,
      - different repeats see different random selections (useful when reporting
        across-run variance),
      - the same configuration always selects the same chunks for the same paper.
    """
    if k <= 0 or not chunks:
        return []
    if len(chunks) <= k:
        return list(chunks)
    digest = hashlib.sha256(
        f"{seed}|{paper_id}|{repeat_idx}".encode("utf-8")
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return rng.sample(list(chunks), k=k)


class RandomChunkLLMBaseline:
    """LLM classification with K randomly-sampled body chunks instead of retrieval.

    Acts as the retrieval-quality ablation against full EpiScope. The pipeline,
    prompts, generator, and chunk count are held fixed; only the chunk
    *selection* changes — from similarity-retrieved to uniform-random over the
    same per-paper chunk pool indexed in Qdrant.
    """

    name = "random_chunk_llm"
    # Papers without indexed body chunks cannot participate in this ablation;
    # they fall through to the standard "missing text" skip path.
    skip_if_no_text = False

    def __init__(
        self,
        *,
        classifier_kind: str,
        generator: LLMGenerator,
        qdrant_db: Any,
        k: int = 10,
        seed: int = 13,
    ) -> None:
        self.classifier_kind = classifier_kind
        self.config = classifier_config(classifier_kind)
        self.prompt_builder = ClassificationPromptBuilder(self.config)
        self.response_parser = ClassificationResponseParser(self.config)
        self.runner = ClassificationRunner(
            generator=generator,
            config=self.config,
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
        )
        self.qdrant_db = qdrant_db
        self.k = int(k)
        self.seed = int(seed)
        self._repeat_idx = 1
        self._chunk_cache: dict[str, List[SearchResult]] = {}

    def set_repeat_idx(self, repeat_idx: int) -> None:
        """Set the current repeat index so per-repeat sampling is reproducible."""
        self._repeat_idx = int(repeat_idx)

    def _chunks_for(self, paper_id: str) -> List[SearchResult]:
        pid = str(paper_id)
        if pid not in self._chunk_cache:
            self._chunk_cache[pid] = load_body_chunks_from_qdrant(self.qdrant_db, pid)
        return self._chunk_cache[pid]

    def predict(
        self,
        paper_id: str,
        metadata: _PaperMetadata,
        record: Mapping[str, object] | None = None,
    ) -> BaselinePrediction:
        raw = (record or {}).get("_metadata")
        source = raw if isinstance(raw, _PaperMetadata) else metadata
        pool = self._chunks_for(paper_id)
        sampled = _sample_random_chunks(
            pool,
            k=self.k,
            paper_id=str(paper_id),
            seed=self.seed,
            repeat_idx=self._repeat_idx,
        )
        # If the paper has no indexed body chunks at all, we still call the
        # runner with an empty chunk list rather than silently substituting
        # metadata-only behaviour: the resulting row will be diagnostically
        # identical to a metadata-only call, but provenance fields will record
        # that this paper had zero candidates.
        expect_no_chunks = not sampled
        attempt = self.runner.run(source, chunks=sampled, expect_no_chunks=expect_no_chunks)
        extras = dict(attempt.result.extras or {})
        extras.update(
            {
                "random_chunk_pool_size": len(pool),
                "random_chunk_sampled": len(sampled),
                "random_chunk_k": self.k,
                "random_chunk_seed": self.seed,
                "random_chunk_repeat_idx": self._repeat_idx,
            }
        )
        attempt.result.extras = extras
        return BaselinePrediction(
            result=attempt.result,
            raw_response=attempt.raw_response,
            prompt_messages=attempt.messages,
        )


def usage_snapshot_from_baseline(baseline: Any) -> Any | None:
    runner = getattr(baseline, "runner", None)
    generator = getattr(runner, "generator", None)
    client = getattr(generator, "client", None)
    usage_snapshot = getattr(client, "usage_snapshot", None)
    if callable(usage_snapshot):
        return usage_snapshot()
    return None


def usage_delta(after: Any, before: Any) -> dict[str, int]:
    fields = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "call_count",
    )
    return {
        field: int(getattr(after, field, 0) or 0) - int(getattr(before, field, 0) or 0)
        for field in fields
    }
