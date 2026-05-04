from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SimpleRagQaCase:
    schema_version: str
    case_id: str
    user_input: str
    reference: str
    paper_path: Optional[str] = None
    paper_id: Optional[str] = None
    reference_contexts: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rubrics: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def resolved_paper_path(self) -> Optional[Path]:
        if not self.paper_path:
            return None
        return Path(self.paper_path)


@dataclass
class RagPipelineConfig:
    loader: str = "unstructured"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunker: str = "paragraph"
    min_chunk_size: int = 20
    chunk_size: int = 600
    chunk_overlap: int = 100

    index_backend: str = "file"
    index_dir: str = ".episcope_index"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "episcope"
    retrieval_mode: str = "dense_only"

    top_k: int = 5
    similarity_threshold: float = 0.0

    llm_provider: str = "nollm"
    llm_model: Optional[str] = None
    temperature: float = 0.0

    continue_on_error: bool = True
    local_context_match_threshold: float = 90.0


@dataclass
class RagCaseRunResult:
    case_id: str
    user_input: str
    reference: str
    response: Optional[str]
    paper_path: Optional[str] = None
    paper_id: Optional[str] = None
    reference_contexts: list[str] = field(default_factory=list)
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieval_count: int = 0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    run_error: Optional[str] = None
    exact_match: Optional[float] = None
    reference_context_precision: Optional[float] = None
    reference_context_recall: Optional[float] = None
    reference_context_f1: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RagasEvaluatorConfig:
    metric_names: list[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "context_precision",
            "context_recall",
            "response_relevancy",
            "factual_correctness",
        ]
    )
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    raise_exceptions: bool = False


@dataclass
class GeneratedQueryReviewRecord:
    schema_version: str
    query_id: str
    source_kind: str
    user_input: str
    reference: str
    status: str
    paper_path: Optional[str] = None
    paper_id: Optional[str] = None
    origin_case_id: Optional[str] = None
    reference_contexts: list[str] = field(default_factory=list)
    persona_name: Optional[str] = None
    synthesizer_name: Optional[str] = None
    review_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
