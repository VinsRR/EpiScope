from typing import List, Optional

from episcope.db.academic_db import AcademicDB
from episcope.rag.generation.base import Generator
from episcope.schemas import PaperMetadata, SearchResult
from episcope.rag.retrieval.base import BaseRetriever
from episcope.workflows.base import AbstractRAG
from episcope.workflows.classification.config import (
    BaseClassifierConfig,
    PaperTypeClassifierConfig,
)
from episcope.workflows.classification.evidence_selection import (
    ClassificationEvidenceSelector,
)
from episcope.workflows.classification.evidence_reranking import (
    EvidenceRerankingStrategy,
    GlobalCrossEncoderReranker,
    NoOpEvidenceReranker,
    WithinLabelCrossEncoderReranker,
)
from episcope.workflows.classification.output import (
    ClassificationDecision,
    ClassificationTrace,
    ClassificationTrainingRecord,
    DetailedClassificationResult,
)
from episcope.workflows.classification.parsing import ClassificationResponseParser
from episcope.workflows.classification.prompting import ClassificationPromptBuilder
from episcope.workflows.classification.runner import ClassificationRunner
from episcope.rag.provenance import (
    Evidence,
    Provenance,
)

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class PaperClassifier(AbstractRAG):
    """Multi-modal paper classification using RAG and semantic similarity."""

    @classmethod
    def with_global_cross_encoder(
        cls,
        *,
        retriever: BaseRetriever,
        generator: Generator,
        model_name: str,
        config: Optional[BaseClassifierConfig] = None,
        strategy_name: Optional[str] = None,
        academic_db: Optional[AcademicDB] = None,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> "PaperClassifier":
        classifier_config = config or PaperTypeClassifierConfig()
        evidence_reranker = GlobalCrossEncoderReranker.from_huggingface(
            model_name=model_name,
            top_k=top_k,
            device=device,
            batch_size=batch_size,
        )
        return cls(
            retriever=retriever,
            generator=generator,
            strategy_name=strategy_name,
            config=classifier_config,
            academic_db=academic_db,
            evidence_reranker=evidence_reranker,
        )

    @classmethod
    def with_within_label_cross_encoder(
        cls,
        *,
        retriever: BaseRetriever,
        generator: Generator,
        model_name: str,
        config: Optional[BaseClassifierConfig] = None,
        strategy_name: Optional[str] = None,
        academic_db: Optional[AcademicDB] = None,
        top_k: Optional[int] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> "PaperClassifier":
        classifier_config = config or PaperTypeClassifierConfig()
        evidence_reranker = WithinLabelCrossEncoderReranker.from_huggingface(
            model_name=model_name,
            top_k=top_k,
            device=device,
            batch_size=batch_size,
        )
        return cls(
            retriever=retriever,
            generator=generator,
            strategy_name=strategy_name,
            config=classifier_config,
            academic_db=academic_db,
            evidence_reranker=evidence_reranker,
        )

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: Generator,
        strategy_name: Optional[str] = None,
        config: Optional[BaseClassifierConfig] = None,
        academic_db: Optional[AcademicDB] = None,
        evidence_reranker: Optional[EvidenceRerankingStrategy] = None,
    ):
        super().__init__(retriever, generator)
        self.config = config or PaperTypeClassifierConfig()
        self.academic_db = academic_db
        self.strategy_name = strategy_name
        self.evidence_reranker = evidence_reranker or NoOpEvidenceReranker()
        self.evidence_selector = ClassificationEvidenceSelector(
            retriever=self.retriever,
            config=self.config,
            evidence_reranker=self.evidence_reranker,
        )
        self.prompt_builder = ClassificationPromptBuilder(self.config)
        self.response_parser = ClassificationResponseParser(self.config)
        self.classification_runner = ClassificationRunner(
            generator=self.generator,
            config=self.config,
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
        )

    def run(
        self,
        paper_id: str,
        metadata: Optional[PaperMetadata] = None,
    ) -> ClassificationDecision:
        """Run classification and return the app-facing decision only."""
        detailed = self.run_detailed(paper_id, metadata=metadata)
        return detailed.decision

    def run_detailed(
        self,
        paper_id: str,
        metadata: Optional[PaperMetadata] = None,
    ) -> DetailedClassificationResult:
        """Run classification and return the full detailed result."""
        metadata = self._resolve_metadata(paper_id, metadata)
        chunks = self.get_relevant_chunks(paper_id, top_k=self.config.top_k)
        attempt = self.classification_runner.run(metadata, chunks)

        decision = ClassificationDecision(
            paper_id=paper_id,
            metadata=metadata,
            result=attempt.result,
            top_evidence=list(chunks),
        )
        provenance = Provenance(
            answer=attempt.raw_response or "",
            evidences=self._build_evidences(paper_id, chunks),
        )
        trace = ClassificationTrace(
            prompt_messages=attempt.messages,
            raw_llm_response=attempt.raw_response,
        )
        training = ClassificationTrainingRecord(
            paper_id=paper_id,
            result=attempt.result,
            prompt_messages=attempt.messages,
            raw_llm_response=attempt.raw_response,
            all_samples=attempt.all_samples,
        )
        detailed_result = DetailedClassificationResult(
            decision=decision,
            provenance=provenance,
            trace=trace,
            training=training,
        )
        return detailed_result

    # -------------------------------------------------------------------------
    # Metadata resolution
    # -------------------------------------------------------------------------

    def _resolve_metadata(
        self, paper_id: str, metadata: Optional[PaperMetadata]
    ) -> PaperMetadata:
        if self.academic_db:
            return self.academic_db.get_paper_metadata(paper_id, self.strategy_name)
        if metadata is None:
            raise ValueError(
                "metadata must be provided when academic_db is not available."
            )
        return metadata

    def get_relevant_chunks(
        self,
        paper_id: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        return self.evidence_selector.select(paper_id, top_k=top_k)

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    def _build_evidences(
        self, paper_id: str, chunks: List[SearchResult]
    ) -> List[Evidence]:
        model_id = getattr(self.generator, "model_id", None)
        prompt_id = getattr(self.config, "prompt_id", None)
        index_version = getattr(self.retriever, "index_version", None)
        return [
            Evidence(
                paper_id=paper_id,
                snippet=chunk.text,
                section=chunk.artifacts.get("category", "unknown"),
                index_version=index_version,
                model_id=model_id,
                prompt_id=prompt_id,
            )
            for chunk in chunks
        ]
