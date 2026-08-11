"""Shared scaffolding for the EpiScope notebooks.

This module is *not* part of the ``episcope`` package — it is notebook plumbing
that lives alongside the notebooks and is imported by them. It holds the two
things every notebook would otherwise copy:

1. **Path bootstrap.** Importing this module puts ``src/`` on ``sys.path`` when
   it detects a local checkout, so the notebooks work from a clone or from an
   installed package without changes.
2. **Deterministic offline stand-ins.** A keyword embedder, fixed-payload
   generators, a static retriever, and two small paper corpora. These keep the
   notebooks runnable with no API key, no model download, and no network — and
   they make every printed score and label reproducible.

None of this is how you would use EpiScope for real. In production you would
call ``EmbedderFactory.get_embedder(...)`` (which defaults to the torch-free
``fastembed`` backend) and ``LLMGenerator`` with a configured provider.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Path bootstrap — must run before any `episcope` import below.
# --------------------------------------------------------------------------


def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Return the repository root of a local checkout, or None if installed."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "episcope").exists():
            return candidate
    return None


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
if PROJECT_ROOT is not None:
    _src = str(PROJECT_ROOT / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


import numpy as np  # noqa: E402

from episcope.db.in_memory_academic_db import InMemoryAcademicDB  # noqa: E402
from episcope.rag.embeddings.base import Embedder  # noqa: E402
from episcope.rag.generation.base import Generator  # noqa: E402
from episcope.rag.indexing.chunking import FixedSizeChunker  # noqa: E402
from episcope.rag.indexing.indexer import Indexer  # noqa: E402
from episcope.rag.provenance import Provenance  # noqa: E402
from episcope.rag.retrieval.candidates import SemanticCandidateRetriever  # noqa: E402
from episcope.rag.retrieval.retriever import Retriever  # noqa: E402
from episcope.schemas import PaperMetadata, SearchResult, StructuredSection  # noqa: E402
from episcope.vectordb.file import FileDB  # noqa: E402

Papers = Dict[str, Dict[str, Any]]


def bootstrap(prefix: str = "episcope-notebook-") -> Path:
    """Return a fresh temporary working directory for a notebook run.

    The ``sys.path`` side of the bootstrap already happened at import time; this
    just hands back scratch space so re-running a notebook never reuses a stale
    index.
    """
    return Path(tempfile.mkdtemp(prefix=prefix))


def pdf_samples_dir() -> Path:
    """Locate ``notebooks/pdf_samples`` from a notebook or the repo root."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "pdf_samples",
        Path.cwd() / "pdf_samples",
        Path.cwd() / "notebooks" / "pdf_samples",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the pdf_samples directory.")


# --------------------------------------------------------------------------
# Deterministic embedder
# --------------------------------------------------------------------------

DATA_VOCABULARY: Tuple[str, ...] = (
    "data",
    "dataset",
    "source",
    "cohort",
    "survey",
    "registry",
    "trial",
    "patient",
    "hospital",
    "mortality",
    "covid",
    "treatment",
    "remdesivir",
    "supplement",
    "table",
    "figure",
    "reference",
    "database",
)

SHARING_VOCABULARY: Tuple[str, ...] = (
    "data",
    "dataset",
    "source",
    "cohort",
    "survey",
    "registry",
    "trial",
    "patient",
    "hospital",
    "mortality",
    "public",
    "repository",
    "available",
    "restricted",
    "consent",
    "records",
    "shared",
    "database",
)

CLASSIFICATION_VOCABULARY: Tuple[str, ...] = (
    "available",
    "repository",
    "dataset",
    "data",
    "registry",
    "public",
    "request",
    "restricted",
    "confidential",
    "hospital",
    "patient",
    "policy",
    "guideline",
    "method",
    "tool",
    "model",
    "workflow",
    "implementation",
)


class TinyKeywordEmbedder(Embedder):
    """A bag-of-keywords embedder: one dimension per vocabulary term.

    Deterministic, instant, and offline — it exists so the notebooks can show
    real indexing and retrieval mechanics without downloading a model. Rankings
    are keyword overlap, not semantics, so treat the scores as illustrative.
    """

    def __init__(
        self,
        vocabulary: Sequence[str] = DATA_VOCABULARY,
        *,
        name: str = "tiny-keyword-demo",
    ) -> None:
        self.vocabulary = tuple(vocabulary)
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return len(self.vocabulary)

    def embed_text(self, text: str) -> List[float]:
        text = text.lower()
        counts = []
        for term in self.vocabulary:
            pattern = rf"\b{re.escape(term)}s?\b"
            counts.append(float(len(re.findall(pattern, text))))

        vector = np.array(counts, dtype="float32")
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return vector.tolist()

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


# --------------------------------------------------------------------------
# Demo corpora
# --------------------------------------------------------------------------


def sample_papers() -> Papers:
    """Two papers that differ in how their data can be accessed.

    Returns fresh objects on every call, so notebooks never share mutable state.
    """
    return {
        "paper_open_data": {
            "metadata": PaperMetadata(
                title="Trial data sharing and reuse",
                abstract=(
                    "A randomized trial reused a public patient-level dataset "
                    "from a hospital registry."
                ),
                keywords=["trial", "registry", "data sharing"],
            ),
            "sections": [
                StructuredSection(
                    title="Methods",
                    section_type="Methods",
                    content=(
                        "The analysis used patient records from the National Hospital Registry. "
                        "The registry stores admission dates, treatment groups, and mortality outcomes."
                    ),
                ),
                StructuredSection(
                    title="Data availability",
                    section_type="Data availability",
                    content=(
                        "De-identified trial data and the analysis code are available from the public repository. "
                        "The dataset can be reused for non-commercial research after registration."
                    ),
                ),
            ],
        },
        "paper_closed_data": {
            "metadata": PaperMetadata(
                title="Hospital cohort study",
                abstract=(
                    "A cohort study collected clinical data directly from "
                    "participating hospitals."
                ),
                keywords=["cohort", "hospital", "mortality"],
            ),
            "sections": [
                StructuredSection(
                    title="Participants",
                    section_type="Methods",
                    content=(
                        "The cohort included adult patients admitted to three hospitals. "
                        "Data were collected by the study team from electronic health records."
                    ),
                ),
                StructuredSection(
                    title="Data sharing",
                    section_type="Data availability",
                    content=(
                        "The patient dataset cannot be shared publicly because the consent agreement "
                        "does not permit redistribution of individual-level hospital records."
                    ),
                ),
            ],
        },
    }


def classification_papers() -> Papers:
    """Two papers that differ in contribution type, for routing/classification."""
    return {
        "paper_open_data": {
            "metadata": PaperMetadata(
                title="Trial data sharing and reuse",
                abstract=(
                    "A randomized trial reused a public patient-level dataset "
                    "from a hospital registry."
                ),
                keywords=["trial", "registry", "data sharing"],
            ),
            "sections": [
                StructuredSection(
                    title="Methods",
                    section_type="Methods",
                    content=(
                        "The analysis used patient records from the National Hospital Registry. "
                        "The registry stores admission dates, treatment groups, and mortality outcomes."
                    ),
                ),
                StructuredSection(
                    title="Data availability",
                    section_type="Data availability",
                    content=(
                        "De-identified trial data and analysis code are available in a public repository. "
                        "The dataset can be reused for non-commercial research after registration."
                    ),
                ),
                StructuredSection(
                    title="Implications",
                    section_type="Discussion",
                    content=(
                        "The findings support a hospital guideline for early patient monitoring. "
                        "Implementation would require local workflow changes."
                    ),
                ),
            ],
        },
        "paper_methods": {
            "metadata": PaperMetadata(
                title="Reusable screening method for outbreak reports",
                abstract=(
                    "The paper develops a text-mining method for classifying "
                    "outbreak reports before manual review."
                ),
                keywords=["method", "screening", "classification"],
            ),
            "sections": [
                StructuredSection(
                    title="Method",
                    section_type="Methods",
                    content=(
                        "We introduce a reusable screening tool and evaluate it on manually annotated abstracts. "
                        "The main contribution is a method for prioritizing review records."
                    ),
                ),
                StructuredSection(
                    title="Data availability",
                    section_type="Data availability",
                    content=(
                        "Annotations are available from the corresponding author "
                        "upon reasonable request."
                    ),
                ),
            ],
        },
    }


# --------------------------------------------------------------------------
# Index / store construction
# --------------------------------------------------------------------------


def build_index(
    index_dir: Path | str,
    papers: Papers,
    embedder: Optional[Embedder] = None,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Tuple[FileDB, Retriever]:
    """Index ``papers`` into a file-backed store and return ``(vdb, retriever)``.

    Notebooks where indexing *is* the lesson build this by hand instead; this
    helper is for the ones that only need a working retriever.
    """
    embedder = embedder or TinyKeywordEmbedder()
    vdb = FileDB(str(index_dir))
    indexer = Indexer(
        vdb,
        embedder=embedder,
        chunker=FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    )
    for paper_id, paper in papers.items():
        indexer.index_paper(paper["sections"], paper["metadata"], paper_id=paper_id)
    vdb.save()

    retriever = Retriever(
        vdb,
        candidate_retrievers=[SemanticCandidateRetriever(vdb, dense_embedder=embedder)],
        use_rerank=False,
    )
    return vdb, retriever


def build_academic_db(
    papers: Papers,
    strategy_name: str,
    *,
    backup_file: Optional[Path | str] = None,
) -> InMemoryAcademicDB:
    """Store ``papers`` as sections/metadata/references under one strategy."""
    db = InMemoryAcademicDB(backup_file=str(backup_file) if backup_file else None)
    for paper_id, paper in papers.items():
        db.insert(
            paper_id,
            "sections",
            strategy_name,
            [section.to_dict() for section in paper["sections"]],
        )
        db.insert(paper_id, "metadata", strategy_name, paper["metadata"].to_dict())
        db.insert(paper_id, "references", strategy_name, [])
    return db


# --------------------------------------------------------------------------
# Offline stand-ins for a real generator / retriever
# --------------------------------------------------------------------------


class FixedJSONGenerator(Generator):
    """Returns a fixed JSON payload, whatever the contexts.

    Swap in ``LLMGenerator`` for real model calls. Note the ``**kwargs``: the
    workflows default to ``structured_output="schema"`` and pass a
    ``response_schema`` down, which a real generator uses to constrain decoding
    and this one simply ignores.
    """

    def __init__(
        self,
        payload: Dict[str, Any],
        *,
        model_id: str = "fixed-json-generator",
    ) -> None:
        self.payload = payload
        self.model_id = model_id

    def generate(
        self,
        contexts: Sequence[Any],
        *,
        question: Optional[str] = None,
        message_builder: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Provenance:
        return Provenance(answer=json.dumps(self.payload), evidences=[])


class StaticRetriever:
    """Returns a fixed evidence list instead of searching an index.

    The workflows only call ``retrieve`` / ``retrieve_by_paper``, so this is
    enough to drive one with evidence you chose by hand — useful when you want
    output changes to come from the prompt, not from retrieval drift.
    """

    def __init__(self, chunks: Sequence[SearchResult]) -> None:
        self.chunks = list(chunks)

    def retrieve(
        self, query: str, top_k: int = 10, **kwargs: Any
    ) -> List[SearchResult]:
        return self.chunks[:top_k]

    def retrieve_by_paper(
        self, query: str, paper_id: str, top_k: int = 10, **kwargs: Any
    ) -> List[SearchResult]:
        return [chunk for chunk in self.chunks if chunk.paper_id == paper_id][:top_k]
