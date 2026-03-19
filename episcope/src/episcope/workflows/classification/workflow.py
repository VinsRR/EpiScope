import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import ValidationError

from episcope.db.academic_db import AcademicDB
from episcope.rag.generation.base import Generator
from episcope.rag.interfaces import AbstractRetriever
from episcope.rag.postprocessing.reranker import Reranker           
from episcope.schemas import PaperMetadata, SearchResult
from episcope.workflows.base import AbstractRAG
from episcope.workflows.classification.config import BaseClassifierConfig, PaperTypeClassifierConfig
from episcope.workflows.classification.schemas import ClassificationResult
from episcope.rag.provenance import (     
    RankedChunk,
    Evidence,
    Provenance,
    CompletionSample,
    ClassificationOutput,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _LLMAttemptResult:
    """Outcome of a single generation + parse attempt."""
    result: ClassificationResult
    raw_response: str
    messages: List[Dict[str, str]]
    all_samples: List[CompletionSample] = field(default_factory=list)


class ClassificationParseError(Exception):
    """Raised when an LLM response cannot be parsed into a ClassificationResult."""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class PaperClassifier(AbstractRAG):
    """Multi-modal paper classification using RAG and semantic similarity."""

    def __init__(
        self,
        retriever: AbstractRetriever,
        generator: Generator,
        strategy_name: Optional[str] = None,
        config: Optional[BaseClassifierConfig] = None,
        academic_db: Optional[AcademicDB] = None,
        reranker: Optional[Reranker] = None,
    ):
        super().__init__(retriever, generator)
        self.config = config or PaperTypeClassifierConfig()
        self.academic_db = academic_db
        self.strategy_name = strategy_name
        self.reranker = reranker

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def run(
        self,
        paper_id: str,
        metadata: Optional[PaperMetadata] = None,
    ) -> ClassificationOutput:
        """Run the full classification workflow for a single paper.

        Returns a ClassificationOutput bundling the structured result,
        full provenance, and the raw prompt/response trace for fine-tuning.
        """
        metadata = self._resolve_metadata(paper_id, metadata)
        chunks = self.get_relevant_chunks(metadata, paper_id, top_k=self.config.top_k)
        attempt = self._llm_classify(metadata, chunks)

        provenance = Provenance(
            answer=attempt.raw_response or "",
            evidences=self._build_evidences(paper_id, chunks),
        )
        return ClassificationOutput(
            paper_id=paper_id,
            metadata=metadata,
            result=attempt.result,
            provenance=provenance,
            prompt_messages=attempt.messages,
            raw_llm_response=attempt.raw_response,
            all_samples=attempt.all_samples,
        )

    # -------------------------------------------------------------------------
    # Metadata resolution
    # -------------------------------------------------------------------------

    def _resolve_metadata(
        self, paper_id: str, metadata: Optional[PaperMetadata]
    ) -> PaperMetadata:
        if self.academic_db:
            return self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
        if metadata is None:
            raise ValueError("metadata must be provided when academic_db is not available.")
        return metadata

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def get_relevant_chunks(
        self,
        metadata: PaperMetadata,
        paper_id: str,
        top_k: int = 10,
    ) -> List[RankedChunk]:
        """Retrieve, globally deduplicate, rank, and optionally rerank chunks.

        The pipeline is:
            fan-out retrieval (all categories × all templates)
            → cross-category deduplication + global ranking  → top_k
            → optional neural reranking                       → config.top_k

        Pass a larger top_k (e.g. 30) to feed the reranker a broad candidate
        set, then let it reduce to config.top_k for the prompt.
        """
        aggregated = self._retrieve_all(paper_id, top_k)
        chunks = self._deduplicate_and_rank(aggregated, top_k)

        if self.reranker:
            query = (
                getattr(self.config, "reranker_query", None)
                or self._default_reranker_query()
            )
            chunks = self._apply_reranker(query, chunks)

        return chunks

    def _retrieve_all(
        self,
        paper_id: str,
        top_k: int,
    ) -> Dict[str, Dict[str, float]]:
        """Fan out across all category templates.

        Returns {category: {text: best_score}} — the best score each text
        achieved within a category, before cross-category dedup.
        """
        category_best: Dict[str, Dict[str, float]] = defaultdict(dict)
        for category, templates in self.config.template_paragraphs.items():
            for query in templates:
                for chunk in self.retriever.retrieve_by_paper(query, paper_id, top_k=top_k):
                    text = chunk.text.strip()
                    if not text:
                        continue
                    current = category_best[category].get(text, float("-inf"))
                    if chunk.similarity_score > current:
                        category_best[category][text] = chunk.similarity_score
        return category_best

    def _deduplicate_and_rank(
        self,
        category_best: Dict[str, Dict[str, float]],
        top_k: int,
    ) -> List[RankedChunk]:
        """Cross-category dedup + global ranking.

        Each text survives only under the category where it scored highest.
        Returns a flat list sorted by score descending, truncated to top_k.
        """
        # For each text, find the category where it scored highest
        winner: Dict[str, str] = {}  # text -> winning category
        for category, chunks in category_best.items():
            for text, score in chunks.items():
                current = winner.get(text)
                if current is None or score > category_best[current][text]:
                    winner[text] = category

        ranked = [
            RankedChunk(
                text=text,
                score=category_best[category][text],
                category=category,
            )
            for text, category in winner.items()
        ]
        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked[:top_k]

    def _apply_reranker(
        self,
        query: str,
        chunks: List[RankedChunk],
    ) -> List[RankedChunk]:
        """Bridge between RankedChunk and the Reranker interface.

        Converts to SearchResult (what Reranker expects), reranks, then
        converts back — preserving category via index since reranker may
        reorder arbitrarily.
        """
        results = [
            SearchResult(id=str(i), text=c.text, similarity_score=c.score)
            for i, c in enumerate(chunks)
        ]
        index_to_category = {str(i): c.category for i, c in enumerate(chunks)}

        reranked = self.reranker.rerank(query, results, top_k=self.config.top_k)

        return [
            RankedChunk(
                text=r.text,
                score=r.rank_score,
                category=index_to_category.get(r.id, "unknown"),
            )
            for r in reranked
        ]

    def _default_reranker_query(self) -> str:
        """Synthesise a reranker query from category labels when none is configured."""
        labels = ", ".join(self.config.category_labels.values())
        return f"Data availability: {labels}"

    # -------------------------------------------------------------------------
    # Classification
    # -------------------------------------------------------------------------

    def _llm_classify(
        self,
        metadata: PaperMetadata,
        chunks: List[RankedChunk],
    ) -> _LLMAttemptResult:
        """LLM-based classification with self-correcting retries.

        Collects all attempts (successful or not) in `all_samples` so that
        RL-based training can use them as a preference signal.
        """
        if not chunks:
            logger.warning("No relevant chunks found for classification.")

        messages = self._build_initial_prompt(metadata, chunks)
        samples: List[CompletionSample] = []
        last_raw: Optional[str] = None

        for attempt in range(1, self.config.max_validation_retries + 1):
            # --- Generation ---
            try:
                generation = self.generator.generate(
                    contexts=[[c.text for c in chunks]],
                    message_builder=lambda **_: messages,
                    format="json",
                )
                last_raw = generation.answer
            except Exception as exc:
                logger.warning("LLM generation attempt %d failed: %s", attempt, exc)
                continue

            # --- Parsing ---
            try:
                result = self._parse_classification_response(last_raw)
                samples.append(
                    CompletionSample(messages=list(messages), completion=last_raw, parsed_ok=True)
                )
                return _LLMAttemptResult(
                    result=result,
                    raw_response=last_raw,
                    messages=list(messages),
                    all_samples=samples,
                )
            except ClassificationParseError as exc:
                logger.warning(
                    "Parse attempt %d/%d failed: %s",
                    attempt, self.config.max_validation_retries, exc,
                )
                logger.debug("Raw response:\n%s", last_raw)
                samples.append(
                    CompletionSample(messages=list(messages), completion=last_raw, parsed_ok=False)
                )
                if attempt < self.config.max_validation_retries:
                    messages = self._append_correction_turn(messages, last_raw, exc)

        logger.error(
            "Classification failed after %d attempt(s). Returning fallback result.",
            self.config.max_validation_retries,
        )
        return _LLMAttemptResult(
            result=self._create_fallback_result(),
            raw_response=last_raw or "",
            messages=list(messages),
            all_samples=samples,
        )

    @staticmethod
    def _append_correction_turn(
        messages: List[Dict[str, str]],
        bad_response: str,
        error: Exception,
    ) -> List[Dict[str, str]]:
        return messages + [
            {"role": "assistant", "content": bad_response},
            {
                "role": "user",
                "content": (
                    f"Your previous response was not valid. Error: {error}\n"
                    "Please return ONLY a single valid JSON object that matches the schema. "
                    "Do not include any extra text, markdown fences, or explanation."
                ),
            },
        ]

    # -------------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------------

    def _build_initial_prompt(
        self,
        metadata: PaperMetadata,
        chunks: List[RankedChunk],
    ) -> List[Dict[str, str]]:
        schema = self.config.output_schema.model_json_schema()
        categories = "\n".join(
            f"{key} – {value}" for key, value in self.config.category_labels.items()
        )
        user_prompt = self.config.user_prompt_template.format(
            categories=categories,
            title=metadata.title,
            abstract=metadata.abstract or "N/A",
            keywords=", ".join(metadata.keywords or []),
            chunks_info=self._format_chunks_for_prompt(chunks),
            schema=json.dumps(schema),
            **self.config.extra_output_fields,
        )
        system_prompt = self.config.system_prompt.format(
            n_categories=len(self.config.category_labels),
            category_labels=", ".join(self.config.category_labels.values()),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_chunks_for_prompt(self, chunks: List[RankedChunk]) -> str:
        """Group chunks by category for prompt readability.

        The underlying list is flat and globally ranked; grouping is purely
        presentational so the LLM can see which label each snippet supports.
        """
        by_category: Dict[str, List[RankedChunk]] = defaultdict(list)
        for chunk in chunks:
            by_category[chunk.category].append(chunk)

        parts: List[str] = []
        for category, cat_chunks in by_category.items():
            avg = np.mean([c.score for c in cat_chunks])
            lines = [f"\n{category.upper()} (avg {avg:.3f}):"]
            lines += [f"  • {c.score:.3f}: {c.text}" for c in cat_chunks]
            parts.append("\n".join(lines))
        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    def _build_evidences(
        self, paper_id: str, chunks: List[RankedChunk]
    ) -> List[Evidence]:
        model_id = getattr(self.generator, "model_id", None)
        prompt_id = getattr(self.config, "prompt_id", None)
        index_version = getattr(self.retriever, "index_version", None)
        return [
            Evidence(
                paper_id=paper_id,
                snippet=chunk.text,
                section=chunk.category,
                index_version=index_version,
                model_id=model_id,
                prompt_id=prompt_id,
            )
            for chunk in chunks
        ]

    # -------------------------------------------------------------------------
    # Response parsing  (raises ClassificationParseError on any failure)
    # -------------------------------------------------------------------------

    def _parse_classification_response(self, response_content: str) -> ClassificationResult:
        raw = response_content.strip()
        parsed = self._parse_json(raw)
        classification = self._extract_classification(parsed)
        extras = self._extract_extras(parsed)
        return ClassificationResult(
            classification=classification,
            confidence=float(parsed.confidence) if getattr(parsed, "confidence", None) is not None else 0.0,
            class_probabilities=parsed.class_probabilities or {},
            evidence={"reasoning": parsed.reasoning},
            extras=extras,
        )

    def _parse_json(self, raw: str) -> Any:
        try:
            return self.config.output_schema.model_validate_json(raw)
        except Exception:
            pass

        candidate = self._extract_json_candidate(raw)
        if candidate is None:
            raise ClassificationParseError("No JSON object found in model output.")

        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ClassificationParseError(f"Extracted JSON is not valid JSON: {exc}") from exc

        try:
            return self.config.output_schema.model_validate_json(candidate)
        except ValidationError as exc:
            raise ClassificationParseError(f"JSON does not match schema: {exc}") from exc
        except Exception as exc:
            raise ClassificationParseError(f"Unexpected validation error: {exc}") from exc

    @staticmethod
    def _extract_json_candidate(text: str) -> Optional[str]:
        for pat in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
            m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()

        start = text.find("{")
        if start == -1:
            return None

        depth, in_string, escape = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1].strip()
        return None

    def _extract_classification(self, parsed: Any) -> List[Any]:
        if hasattr(parsed, "classification"):
            raw_cls = parsed.classification
            if isinstance(raw_cls, list):
                mapped = [
                    self.config.classification_mapping[c]
                    for c in raw_cls
                    if c in self.config.classification_mapping
                ]
                return mapped or self.config.default_classification
            mapped = self.config.classification_mapping.get(raw_cls)
            return [mapped] if mapped is not None else self.config.default_classification

        if hasattr(parsed, "primary_label"):
            return [parsed.primary_label]

        raise ClassificationParseError(
            "Parsed output has neither 'classification' nor 'primary_label'."
        )

    @staticmethod
    def _extract_extras(parsed: Any) -> Dict[str, Any]:
        extras: Dict[str, Any] = {}
        if hasattr(parsed, "extras") and parsed.extras:
            extras = (
                parsed.extras.model_dump()
                if hasattr(parsed.extras, "model_dump")
                else dict(parsed.extras)
            )
        if hasattr(parsed, "secondary_labels") and parsed.secondary_labels:
            extras["secondary_labels"] = parsed.secondary_labels
        return extras

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    def _create_fallback_result(self) -> ClassificationResult:
        n = len(self.config.category_labels)
        uniform = 1 / n if n else 0.0
        return ClassificationResult(
            classification=self.config.default_classification,
            confidence=0.0,
            class_probabilities={label: uniform for label in self.config.category_labels.values()},
            evidence={"reasoning": "Classification failed; defaulting to unclear."},
            extras={},
        )