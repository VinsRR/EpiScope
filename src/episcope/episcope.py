"""Command-line interface for EpiScope."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import requests
import typer

from episcope.db import InMemoryAcademicDB, MongoAcademicDB
from episcope.rag.embeddings.factory import EmbedderFactory
from episcope.rag.indexing.chunking import (
    FixedSizeChunker,
    NoChunker,
    ParagraphChunker,
    SentenceChunker,
)
from episcope.rag.indexing.indexer import Indexer
from episcope.rag.ingestion.document_loader import DocumentLoaderFactory
from episcope.rag.generation.nollm_generator import NoLLMGenerator
from episcope.rag.retrieval.candidates import (
    HybridCandidateRetriever,
    SemanticCandidateRetriever,
    SparseCandidateRetriever,
)
from episcope.rag.retrieval.retriever import Retriever
from episcope.schemas import PaperMetadata, Reference, StructuredSection
from episcope.services.errors import humanize_error
from episcope.settings import AppSettings, env
from episcope.utils.logger import setup_logging
from episcope.vectordb.file import FileDB
from episcope.vectordb.qdrant import QdrantDB
from episcope.workspace import (
    WorkspaceConfig,
    create_workspace,
    find_workspace,
    load_workspace,
)
from episcope.workflows import PaperClassifier, PrecisionMiner
from episcope.workflows.classification import (
    GlobalCrossEncoderReranker,
    WithinLabelCrossEncoderReranker,
)
from episcope.workflows.registry import (
    TaskSpec,
    build_classifier_config,
    build_precision_miner_config,
    classifier_catalog,
    load_task_file,
    load_tasks_dir,
    miner_catalog,
    scaffold_task_spec,
    validate_task_file,
)


app = typer.Typer(
    help="EpiScope CLI with local-first defaults for indexing, retrieval, classification, and extraction.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        from episcope import __version__

        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the EpiScope version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """EpiScope CLI with local-first defaults for indexing, retrieval, classification, and extraction."""


_SETTINGS = AppSettings.from_env()
_CLI_ROOT = Path(".episcope")
_DEFAULT_INDEX_DIR = _CLI_ROOT / "index"
_DEFAULT_DB_BACKUP = _CLI_ROOT / "academic_db.json"
_DEFAULT_STRATEGY_NAME = "local-cli"
_DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_MIN_CHUNK_SIZE = 20


class LoaderKind(str, Enum):
    unstructured = "unstructured"
    grobid = "grobid"


class IndexBackend(str, Enum):
    file = "file"
    faiss = "faiss"
    qdrant = "qdrant"


class MetadataBackend(str, Enum):
    memory = "memory"
    mongo = "mongo"


class ChunkerKind(str, Enum):
    none = "none"
    sentence = "sentence"
    paragraph = "paragraph"
    fixed_size = "fixed_size"


class RetrievalMode(str, Enum):
    dense_only = "dense_only"
    hybrid = "hybrid"
    sparse_only = "sparse_only"
    hybrid_candidates_only = "hybrid_candidates_only"


class LLMProvider(str, Enum):
    nollm = "nollm"
    gemini = "gemini"
    openai = "openai"
    openrouter = "openrouter"
    anthropic = "anthropic"
    ollama = "ollama"


class EvidenceRerankerKind(str, Enum):
    none = "none"
    global_cross_encoder = "global_cross_encoder"
    within_label_cross_encoder = "within_label_cross_encoder"


class OutputFormat(str, Enum):
    human = "human"
    json = "json"


class QualityPreset(str, Enum):
    fast = "fast"
    balanced = "balanced"
    accurate = "accurate"


@dataclass(frozen=True)
class _QualityValues:
    """One row of the --quality preset table.

    Deliberately excludes ``loader`` (accurate-but-slower parsing needs a
    running GROBID service, which would make a "quality" knob silently start
    depending on external infrastructure) and ``embed_model`` (changing the
    embedding model changes vector dimensionality, which breaks retrieval
    against an already-built index of a different quality level).
    """

    chunker: ChunkerKind
    min_chunk_size: int
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: RetrievalMode


_QUALITY_PRESETS: Dict[QualityPreset, _QualityValues] = {
    QualityPreset.fast: _QualityValues(
        chunker=ChunkerKind.paragraph,
        min_chunk_size=_DEFAULT_MIN_CHUNK_SIZE,
        chunk_size=800,
        chunk_overlap=80,
        retrieval_mode=RetrievalMode.dense_only,
    ),
    QualityPreset.balanced: _QualityValues(
        chunker=ChunkerKind.paragraph,
        min_chunk_size=_DEFAULT_MIN_CHUNK_SIZE,
        chunk_size=600,
        chunk_overlap=100,
        retrieval_mode=RetrievalMode.hybrid,
    ),
    QualityPreset.accurate: _QualityValues(
        chunker=ChunkerKind.paragraph,
        min_chunk_size=_DEFAULT_MIN_CHUNK_SIZE,
        chunk_size=500,
        chunk_overlap=120,
        retrieval_mode=RetrievalMode.hybrid,
    ),
}


@dataclass
class LoadedPaper:
    paper_id: str
    path: Path
    sections: list[StructuredSection]
    metadata: PaperMetadata
    references: list[Reference]


def main() -> None:
    setup_logging()
    app()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_ready(value.model_dump())
    if hasattr(value, "name") and hasattr(value, "value"):
        return getattr(value, "name")
    return value


def _echo_json(value: Any) -> None:
    typer.echo(json.dumps(_json_ready(value), indent=2, ensure_ascii=False))


def _format_option() -> Any:
    return typer.Option(
        OutputFormat.human,
        "--format",
        "-f",
        envvar="EPISCOPE_OUTPUT_FORMAT",
        help="Output format: 'human' (default) or 'json'. "
        "Falls back to $EPISCOPE_OUTPUT_FORMAT when set.",
    )


def _emit(
    payload: Any,
    output_format: OutputFormat,
    human: Callable[[Any], list[str]],
) -> None:
    """Render a command result as JSON or as a human-readable summary."""
    if output_format == OutputFormat.json:
        _echo_json(payload)
        return
    for line in human(payload):
        typer.echo(line)


def _truncate(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def _human_init(payload: Any) -> list[str]:
    data = _json_ready(payload)
    return [
        f"Created workspace: {data['workspace']}",
        f"  config:   {data['config']}",
        f"  papers:   {data['papers_dir']}",
        f"  index:    {data['index_dir']}",
        f"  metadata: {data['metadata_path']}",
        f"  outputs:  {data['outputs_dir']}",
        f"  logs:     {data['logs_dir']}",
    ]


def _human_inspect(payload: Any) -> list[str]:
    data = _json_ready(payload)
    meta = data.get("metadata") or {}
    lines = [
        f"Paper: {data['paper_id']}",
        f"Title: {meta.get('title') or '(untitled)'}",
        f"Path:  {data['path']}",
        f"Sections: {data['section_count']}   References: {data['reference_count']}",
    ]
    titles = [title for title in data.get("section_titles", []) if title]
    if titles:
        lines.append("Section titles: " + ", ".join(titles))
    return lines


def _human_index(payload: Any) -> list[str]:
    data = _json_ready(payload)
    papers = data.get("papers", [])
    lines = [
        f"Indexed {len(papers)} paper(s) into '{data['index_backend']}' "
        f"(strategy: {data['strategy_name']}, embed: {data['embed_model']})"
    ]
    for paper in papers:
        lines.append(
            f"  - {paper['paper_id']}: {paper.get('title') or '(untitled)'} "
            f"({paper['section_count']} sections, {paper['reference_count']} refs)"
        )
    return lines


def _human_papers(payload: Any) -> list[str]:
    data = _json_ready(payload)
    ids = data.get("paper_ids", [])
    context = f"strategy: {data['strategy_name']}, backend: {data['metadata_backend']}"
    if not ids:
        return [f"No papers found ({context})."]
    lines = [f"{data['count']} paper(s) ({context}):"]
    lines.extend(f"  - {paper_id}" for paper_id in ids)
    return lines


def _human_explore(payload: Any) -> list[str]:
    data = _json_ready(payload)
    chunks = data.get("retrieved_chunks", [])
    lines = [
        f"Query: {data['query']}",
        f"Retrieved {data['retrieval_count']} chunk(s) [mode: {data['retrieval_mode']}]",
    ]
    for rank, chunk in enumerate(chunks, start=1):
        label = chunk.get("section_title") or chunk.get("section_type") or "?"
        lines.append(
            f"  {rank}. [{chunk.get('paper_id', '?')} · {label}] "
            f"(score {_format_score(chunk.get('rank_score'))})"
        )
        text = chunk.get("text") or ""
        if text:
            lines.append(f"     {_truncate(text)}")
    if data.get("answer"):
        lines.extend(["", "Answer:", data["answer"]])
    return lines


def _human_ask(payload: Any) -> list[str]:
    data = _json_ready(payload)
    lines = [f"Q: {data['question']}", "", f"A: {data['answer']}"]
    sources = data.get("sources", [])
    if sources:
        lines.extend(["", f"Sources ({data.get('source_count', len(sources))}):"])
        for source in sources:
            label = source.get("section_title") or source.get("section_type") or ""
            separator = f" · {label}" if label else ""
            lines.append(
                f"  - {source.get('paper_id', '?')}{separator} "
                f"(score {_format_score(source.get('rank_score'))})"
            )
    return lines


def _human_classify(payload: Any) -> list[str]:
    data = _json_ready(payload)
    decision = data.get("decision", data)
    result = decision.get("result") or {}
    labels = result.get("classification") or []
    lines = [
        f"Paper: {decision.get('paper_id', '?')}",
        "Classification: "
        + (", ".join(str(label) for label in labels) if labels else "(none)"),
    ]
    confidence = result.get("confidence")
    if isinstance(confidence, (int, float)):
        lines.append(f"Confidence: {confidence:.2f}")
    extras = result.get("extras") or {}
    if extras:
        lines.append(f"Extras: {json.dumps(extras, ensure_ascii=False)}")
    evidence = decision.get("top_evidence") or []
    if evidence:
        lines.append(f"Evidence chunks: {len(evidence)}")
    return lines


def _human_precision_miner(payload: Any) -> list[str]:
    data = _json_ready(payload)
    result = data.get("result") or {}
    items = result.get("items", [])
    lines = [f"Paper: {data.get('paper_id', '?')}"]
    if result.get("description"):
        lines.append(f"Summary: {result['description']}")
    lines.append(f"Extracted {len(items)} item(s):")
    for item in items:
        url = item.get("url")
        url_str = f" — {url}" if url and url != "N/A" else ""
        lines.append(f"  - {item.get('name', '?')}{url_str}")
        if item.get("explanation"):
            lines.append(f"      {_truncate(item['explanation'], 160)}")
    return lines


def _human_tasks(payload: Any) -> list[str]:
    data = _json_ready(payload)
    lines: list[str] = []
    for family, title in (("classifiers", "Classifiers"), ("miners", "Miners")):
        lines.append(f"{title}:")
        for item in data.get(family, []):
            source = item.get("source", "builtin")
            suffix = "" if source == "builtin" else f"  [{source}]"
            lines.append(f"  - {item['key']}: {item.get('label', '')}{suffix}")
        lines.append("")
    return lines


def _abort(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _abort_exc(exc: Exception) -> None:
    _abort(humanize_error(exc))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _optional_workspace(path: Optional[Path]) -> Optional[WorkspaceConfig]:
    if path is not None:
        return load_workspace(path)
    return find_workspace()


def _workspace_enum(
    workspace: Optional[WorkspaceConfig],
    enum_cls,
    current,
    default,
    workspace_value: str,
):
    if workspace is not None and current == default:
        return enum_cls(workspace_value)
    return current


def _workspace_value(
    workspace: Optional[WorkspaceConfig],
    current,
    default,
    workspace_value,
):
    if workspace is not None and current == default:
        return workspace_value
    return current


def _resolve_value(
    current,
    default,
    quality: Optional[QualityPreset],
    quality_value,
    workspace: Optional[WorkspaceConfig],
    workspace_value,
):
    """Resolve one --quality-eligible field.

    Precedence: an explicitly-typed CLI flag always wins; otherwise
    ``--quality`` wins over ``episcope.toml``, which wins over the
    hardcoded default. Mirrors :func:`_workspace_value`'s
    default-detection, with the preset layer inserted above it.
    """
    if current != default:
        return current
    if quality is not None:
        return quality_value
    if workspace is not None:
        return workspace_value
    return current


def _resolve_enum(
    current,
    default,
    enum_cls,
    quality: Optional[QualityPreset],
    quality_value,
    workspace: Optional[WorkspaceConfig],
    workspace_value: str,
):
    """Enum-typed counterpart of :func:`_resolve_value` (see its docstring)."""
    if current != default:
        return current
    if quality is not None:
        return quality_value
    if workspace is not None:
        return enum_cls(workspace_value)
    return current


def _workspace_path(
    workspace: Optional[WorkspaceConfig],
    current: Path,
    default: Path,
    workspace_value: str,
) -> Path:
    if workspace is not None and current == default:
        return workspace.resolve_path(workspace_value)
    return current


def _prepare_tasks(
    workspace: Optional[WorkspaceConfig],
    task_file: Optional[Path],
    current_kind: str,
    *,
    expected: str,
) -> str:
    """Register workspace + inline declarative tasks; return the kind to run."""
    if workspace is not None:
        load_tasks_dir(workspace.root / "tasks", overwrite=True)
    if task_file is not None:
        spec = load_task_file(task_file, overwrite=True)
        if spec.kind != expected:
            raise ValueError(
                f"--task-file defines a {spec.kind!r} task, "
                f"but this command runs {expected!r} tasks."
            )
        return spec.key
    return current_kind


def _build_chunker(
    chunker_kind: ChunkerKind,
    *,
    min_chunk_size: int,
    chunk_size: int,
    chunk_overlap: int,
):
    if chunker_kind == ChunkerKind.none:
        return NoChunker()
    if chunker_kind == ChunkerKind.sentence:
        return SentenceChunker()
    if chunker_kind == ChunkerKind.paragraph:
        return ParagraphChunker(min_chunk_size=min_chunk_size)
    if chunker_kind == ChunkerKind.fixed_size:
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(f"Unsupported chunker: {chunker_kind.value}")


def _build_vector_db(
    backend: IndexBackend,
    *,
    index_dir: Path,
    qdrant_collection: str,
    qdrant_url: str,
    dense_dim: Optional[int] = None,
):
    if backend == IndexBackend.file:
        index_dir.mkdir(parents=True, exist_ok=True)
        return FileDB(str(index_dir))
    if backend == IndexBackend.faiss:
        from episcope.vectordb.faiss import FaissDB

        index_dir.mkdir(parents=True, exist_ok=True)
        return FaissDB(str(index_dir))
    if backend == IndexBackend.qdrant:
        return QdrantDB(
            collection=qdrant_collection,
            url=qdrant_url,
            dense_dim=dense_dim,
            use_dense=True,
            use_sparse=False,
            use_late=False,
        )
    raise ValueError(f"Unsupported backend: {backend.value}")


def _build_metadata_db(
    backend: MetadataBackend,
    *,
    db_backup: Path,
    mongo_uri: Optional[str],
    mongo_db_name: str,
):
    if backend == MetadataBackend.memory:
        _ensure_parent_dir(db_backup)
        return InMemoryAcademicDB(backup_file=str(db_backup))
    resolved_mongo_uri = mongo_uri or _SETTINGS.mongo_uri
    if not resolved_mongo_uri:
        raise ValueError("Mongo metadata backend requires --mongo-uri or MONGO_URI.")
    return MongoAcademicDB(uri=resolved_mongo_uri, db_name=mongo_db_name)


def _build_generator(provider: LLMProvider, model: Optional[str], temperature: float):
    from episcope.rag.generation.llm_generator import LLMGenerator
    from episcope.clients import (
        AnthropicClient,
        GeminiClient,
        OllamaClient,
        OpenAIClient,
        OpenRouterClient,
    )

    if provider == LLMProvider.nollm:
        return NoLLMGenerator()

    resolved_model = model
    if resolved_model is None:
        if provider == LLMProvider.gemini:
            resolved_model = (
                _SETTINGS.llm_model
                if _SETTINGS.llm_provider == LLMProvider.gemini.value
                and _SETTINGS.llm_model
                else _DEFAULT_GEMINI_MODEL
            )
        else:
            raise ValueError(
                f"--llm-model is required when --llm-provider is {provider.value!r}."
            )

    if provider == LLMProvider.gemini:
        client = GeminiClient()
    elif provider == LLMProvider.openai:
        client = OpenAIClient()
    elif provider == LLMProvider.openrouter:
        client = OpenRouterClient()
    elif provider == LLMProvider.anthropic:
        client = AnthropicClient()
    elif provider == LLMProvider.ollama:
        client = OllamaClient()
    else:
        raise ValueError(f"Unsupported llm provider: {provider.value}")

    return LLMGenerator(client=client, model=resolved_model, temperature=temperature)


# Small model for free-form Q&A; a larger instruct model for the structured
# (JSON) classification/extraction workflows.
_OLLAMA_SMALL_MODEL = "llama3.2:1b"
_OLLAMA_STRONG_MODEL = "qwen2.5:7b"


def _ollama_models() -> Optional[list[str]]:
    """Locally-available Ollama model names, or None if the server is unreachable."""
    try:
        resp = requests.get(f"{_SETTINGS.ollama_host}/api/tags", timeout=3)
        resp.raise_for_status()
        return [model.get("name", "") for model in resp.json().get("models", [])]
    except Exception:
        return None


def _ollama_setup_hint(model: str) -> str:
    return (
        "To run EpiScope without a cloud API key, use Ollama (free, local):\n"
        "  1. Install — macOS: `brew install ollama` (or https://ollama.com/download)\n"
        "              Linux: `curl -fsSL https://ollama.com/install.sh | sh`\n"
        "  2. Start it: `ollama serve`\n"
        f"  3. Pull a model: `ollama pull {model}`\n"
        f"  4. Re-run with: `--llm-provider ollama --llm-model {model}`\n\n"
        "Or set GEMINI_API_KEY (or OPENAI_API_KEY / OPENROUTER_API_KEY / "
        "ANTHROPIC_API_KEY) in your .env."
    )


def _resolve_generator(
    provider: LLMProvider,
    model: Optional[str],
    temperature: float,
    *,
    task: str,
):
    """Build a generator, falling back to a local Ollama model when no key is set.

    Honors an explicit provider choice and the configured Gemini key. Only when
    the default Gemini provider is used *and* no key is present does it route to
    a local Ollama model — a small one for ``ask``, with a notice that the
    classification/extraction workflows need a larger model.
    """
    needs_strong = task in {"classify", "precision-miner"}
    default_model = _OLLAMA_STRONG_MODEL if needs_strong else _OLLAMA_SMALL_MODEL

    if provider != LLMProvider.gemini or env("GEMINI_API_KEY"):
        return _build_generator(provider, model, temperature)

    available = _ollama_models()
    if available is None:
        raise ValueError(
            "No GEMINI_API_KEY found and no local Ollama server is running.\n\n"
            + _ollama_setup_hint(default_model)
        )

    # `model`, if set, was chosen in the Gemini provider's context (either an
    # explicit --llm-model flag or the workspace config's gemini model), so it
    # is never a valid Ollama model name here — always use the Ollama default.
    chosen = default_model
    if chosen not in available:
        hint = (
            f"Ollama is running, but the model '{chosen}' is not pulled.\n"
            f"  Run: `ollama pull {chosen}`"
        )
        if available:
            hint += f"\n  Models available now: {', '.join(available)}"
        raise ValueError(hint)

    typer.secho(
        f"No API key found — using a local Ollama model ('{chosen}').",
        fg=typer.colors.YELLOW,
        err=True,
    )
    if needs_strong:
        typer.secho(
            "Note: classification and extraction need a capable instruct model; "
            "a small model may return incomplete or invalid results "
            f"(recommended: `ollama pull {_OLLAMA_STRONG_MODEL}`).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    if sys.stdin.isatty() and not typer.confirm(
        "Continue with the local model?", default=True
    ):
        raise ValueError("Cancelled.")

    return _build_generator(LLMProvider.ollama, chosen, temperature)


def _build_retriever(
    vectordb: Any,
    *,
    retrieval_mode: RetrievalMode,
    dense_embedder: Any = None,
):
    if retrieval_mode == RetrievalMode.dense_only:
        return Retriever(
            vectordb=vectordb,
            use_rerank=False,
            candidate_retrievers=[
                SemanticCandidateRetriever(vectordb, dense_embedder=dense_embedder)
            ],
        )

    capabilities = vectordb.capabilities()
    if not capabilities.get("sparse"):
        raise ValueError(
            f"Retrieval mode {retrieval_mode.value!r} requires sparse-capable storage. "
            "Use --retrieval-mode dense-only for file/faiss indexes."
        )

    if retrieval_mode == RetrievalMode.sparse_only:
        return Retriever(
            vectordb=vectordb,
            use_rerank=False,
            candidate_retrievers=[SparseCandidateRetriever(vectordb)],
        )
    if retrieval_mode == RetrievalMode.hybrid_candidates_only:
        return Retriever(
            vectordb=vectordb,
            use_rerank=False,
            candidate_retrievers=[HybridCandidateRetriever(vectordb)],
        )
    if retrieval_mode == RetrievalMode.hybrid:
        return Retriever(vectordb=vectordb, use_rerank=False)

    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode.value}")


def _build_evidence_reranker(
    kind: EvidenceRerankerKind,
    *,
    cross_encoder_model: Optional[str],
    cross_encoder_top_k: int,
):
    if kind == EvidenceRerankerKind.none:
        return None
    if not cross_encoder_model:
        raise ValueError(
            "--cross-encoder-model is required when an evidence reranker is enabled."
        )
    if kind == EvidenceRerankerKind.global_cross_encoder:
        return GlobalCrossEncoderReranker.from_huggingface(
            model_name=cross_encoder_model,
            top_k=cross_encoder_top_k,
        )
    if kind == EvidenceRerankerKind.within_label_cross_encoder:
        return WithinLabelCrossEncoderReranker.from_huggingface(
            model_name=cross_encoder_model,
            top_k=cross_encoder_top_k,
        )
    raise ValueError(f"Unsupported evidence reranker: {kind.value}")


def _loader_for(kind: LoaderKind):
    return DocumentLoaderFactory.get_loader(kind.value)


def _is_supported_file(path: Path) -> bool:
    return path.suffix.lower() in {".pdf", ".txt", ".md", ".text"}


def _iter_supported_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if not _is_supported_file(path):
            raise ValueError(f"Unsupported file type: {path}")
        yield path
        return

    if not path.is_dir():
        raise ValueError(f"Expected a file or directory, got: {path}")

    for child in sorted(path.rglob("*")):
        if child.is_file() and _is_supported_file(child):
            yield child


def _load_paper(
    path: Path, *, loader_kind: LoaderKind, paper_id: Optional[str] = None
) -> LoadedPaper:
    loader = _loader_for(loader_kind)
    sections, metadata, references = loader.load(path)
    if not metadata.title:
        metadata.title = path.stem
    metadata.file_path = str(path)
    return LoadedPaper(
        paper_id=paper_id or path.stem,
        path=path,
        sections=sections,
        metadata=metadata,
        references=references,
    )


def _load_path(
    path: Path,
    *,
    loader_kind: LoaderKind,
    paper_id: Optional[str] = None,
) -> list[LoadedPaper]:
    files = list(_iter_supported_files(path))
    if not files:
        raise ValueError(f"No supported documents found under {path}")
    if paper_id and len(files) != 1:
        raise ValueError("--paper-id can only be used when indexing a single file.")

    seen_ids: set[str] = set()
    papers: list[LoadedPaper] = []
    for file_path in files:
        loaded = _load_paper(file_path, loader_kind=loader_kind, paper_id=paper_id)
        if loaded.paper_id in seen_ids:
            raise ValueError(
                f"Duplicate paper id {loaded.paper_id!r}. Rename files or index them separately."
            )
        seen_ids.add(loaded.paper_id)
        papers.append(loaded)
    return papers


def _persist_papers(papers: list[LoadedPaper], *, db: Any, strategy_name: str) -> None:
    for paper in papers:
        db.insert(
            paper.paper_id,
            "sections",
            strategy_name,
            [section.to_dict() for section in paper.sections],
        )
        db.insert(
            paper.paper_id,
            "metadata",
            strategy_name,
            paper.metadata.to_dict(),
        )
        db.insert(
            paper.paper_id,
            "references",
            strategy_name,
            [reference.to_dict() for reference in paper.references],
        )


def _index_papers(
    papers: list[LoadedPaper],
    *,
    vectordb: Any,
    embed_model: str,
    chunker_kind: ChunkerKind,
    min_chunk_size: int,
    chunk_size: int,
    chunk_overlap: int,
):
    chunker = _build_chunker(
        chunker_kind,
        min_chunk_size=min_chunk_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedder = EmbedderFactory.get_embedder(embed_model)
    indexer = Indexer(vectordb, embedder=embedder, chunker=chunker)
    for paper in papers:
        indexer.index_paper(paper.sections, paper.metadata, paper.paper_id)
    if hasattr(vectordb, "save"):
        vectordb.save()
    return embedder


def _paper_summaries(papers: list[LoadedPaper]) -> list[dict[str, Any]]:
    return [
        {
            "paper_id": paper.paper_id,
            "path": str(paper.path),
            "title": paper.metadata.title,
            "section_count": len(paper.sections),
            "reference_count": len(paper.references),
        }
        for paper in papers
    ]


def _transient_retriever_for_path(
    path: Path,
    *,
    loader_kind: LoaderKind,
    embed_model: str,
    chunker_kind: ChunkerKind,
    min_chunk_size: int,
    chunk_size: int,
    chunk_overlap: int,
):
    papers = _load_path(path, loader_kind=loader_kind)
    tempdir = tempfile.TemporaryDirectory(prefix="episcope-cli-")
    vectordb = FileDB(tempdir.name)
    embedder = _index_papers(
        papers,
        vectordb=vectordb,
        embed_model=embed_model,
        chunker_kind=chunker_kind,
        min_chunk_size=min_chunk_size,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    retriever = _build_retriever(
        vectordb,
        retrieval_mode=RetrievalMode.dense_only,
        dense_embedder=embedder,
    )
    return tempdir, papers, retriever


def _resolve_paper_from_store(
    paper_id: str,
    *,
    loader_kind: LoaderKind,
    file_path: Optional[Path],
    strategy_name: str,
    metadata_backend: MetadataBackend,
    db_backup: Path,
    mongo_uri: Optional[str],
    mongo_db_name: str,
    index_backend: IndexBackend,
    index_dir: Path,
    qdrant_collection: str,
    qdrant_url: str,
    retrieval_mode: RetrievalMode,
    embed_model: str,
    chunker_kind: ChunkerKind,
    min_chunk_size: int,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_mode_explicit: bool = False,
):
    if file_path is not None:
        if retrieval_mode != RetrievalMode.dense_only:
            if not retrieval_mode_explicit:
                # A --quality preset (not the user) asked for hybrid/sparse
                # retrieval; transient --file mode only ever builds a
                # dense-only local index, so silently use it rather than
                # erroring on a choice the user never made.
                retrieval_mode = RetrievalMode.dense_only
            else:
                raise ValueError(
                    "Transient --file mode only supports --retrieval-mode dense-only."
                )
        tempdir, papers, retriever = _transient_retriever_for_path(
            file_path,
            loader_kind=loader_kind,
            embed_model=embed_model,
            chunker_kind=chunker_kind,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if len(papers) != 1:
            raise ValueError("--file mode expects a single document.")
        return {
            "paper_id": papers[0].paper_id,
            "metadata": papers[0].metadata,
            "retriever": retriever,
            "academic_db": None,
            "tempdir": tempdir,
        }

    if not paper_id:
        raise ValueError("Provide either a paper_id argument or --file.")

    academic_db = _build_metadata_db(
        metadata_backend,
        db_backup=db_backup,
        mongo_uri=mongo_uri,
        mongo_db_name=mongo_db_name,
    )
    metadata = academic_db.get_paper_metadata(paper_id, strategy_name)
    if metadata is None:
        raise ValueError(
            f"Paper {paper_id!r} was not found under strategy {strategy_name!r}. "
            "Run `episcope papers` to inspect the available ids."
        )

    vectordb = _build_vector_db(
        index_backend,
        index_dir=index_dir,
        qdrant_collection=qdrant_collection,
        qdrant_url=qdrant_url,
    )
    retriever = _build_retriever(vectordb, retrieval_mode=retrieval_mode)
    return {
        "paper_id": paper_id,
        "metadata": metadata,
        "retriever": retriever,
        "academic_db": academic_db,
        "tempdir": None,
    }


_DOCTOR_STATUS_STYLES = {
    "ok": ("✓", typer.colors.GREEN),
    "warn": ("!", typer.colors.YELLOW),
    "fail": ("✗", typer.colors.RED),
    "skip": ("·", typer.colors.BRIGHT_BLACK),
}


def _probe_http(url: str, *, timeout: float = 3.0) -> tuple[bool, Optional[int], str]:
    """Best-effort HTTP GET used by `doctor`; never raises."""
    try:
        resp = requests.get(url, timeout=timeout)
        return True, resp.status_code, ""
    except Exception as exc:
        return False, None, type(exc).__name__


def _check_mongo(uri: str) -> tuple[str, str, str]:
    """Return (status, detail, hint) for a MongoDB connectivity probe."""
    try:
        from pymongo import MongoClient
    except ImportError:
        return "skip", "configured", "Install epi-scope[server] to verify connectivity."
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        try:
            client.admin.command("ping")
        finally:
            client.close()
        return "ok", "reachable", ""
    except Exception as exc:
        return (
            "warn",
            f"unreachable ({type(exc).__name__})",
            "Only needed for the API/UI corpus path.",
        )


def _print_doctor_check(status: str, label: str, detail: str, hint: str) -> None:
    symbol, color = _DOCTOR_STATUS_STYLES.get(status, ("?", None))
    line = f"  {typer.style(symbol, fg=color)} {label}"
    if detail:
        line += f": {detail}"
    typer.echo(line)
    if hint:
        typer.secho(f"      {hint}", fg=typer.colors.BRIGHT_BLACK)


def _collect_doctor_checks(
    probe: bool, workspace: Optional[Path]
) -> tuple[list[dict[str, Any]], bool]:
    """Build the same environment/LLM/services/workspace checks `doctor`
    reports, for reuse by other commands (e.g. `quickstart`).

    Returns ``(checks, any_fail)``.
    """
    import platform
    import sys

    from episcope import __version__

    settings = AppSettings.from_env()
    checks: list[dict[str, Any]] = []

    def add(
        status: str,
        label: str,
        detail: str = "",
        hint: str = "",
        *,
        section: str,
    ) -> None:
        checks.append(
            {
                "section": section,
                "status": status,
                "label": label,
                "detail": detail,
                "hint": hint,
            }
        )

    # Environment ----------------------------------------------------------
    py_ok = sys.version_info >= (3, 10)
    add("ok", "EpiScope version", __version__, section="Environment")
    add(
        "ok" if py_ok else "fail",
        "Python",
        platform.python_version(),
        "" if py_ok else "EpiScope requires Python 3.10 or newer.",
        section="Environment",
    )
    from episcope.rag.device import resolve_device

    add(
        "ok",
        "Compute device",
        resolve_device(),
        "Override with EPISCOPE_DEVICE=cpu|cuda|mps.",
        section="Environment",
    )

    # LLM ------------------------------------------------------------------
    provider = settings.llm_provider
    add("ok", "Provider", provider, section="LLM")
    add("ok", "Model", settings.llm_model, section="LLM")
    add("ok", "Embedding provider", settings.embed_provider, section="LLM")
    key_env = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider)
    if provider == "ollama":
        if probe:
            ok, code, err = _probe_http(f"{settings.ollama_host}/api/tags")
            add(
                "ok" if ok else "fail",
                "Ollama server",
                f"{settings.ollama_host} (HTTP {code})"
                if ok
                else f"{settings.ollama_host} ({err})",
                "" if ok else "Start Ollama or set OLLAMA_HOST.",
                section="LLM",
            )
        else:
            add(
                "skip",
                "Ollama server",
                settings.ollama_host,
                "Drop --no-probe to test connectivity.",
                section="LLM",
            )
    elif key_env is not None:
        has_key = bool(env(key_env))
        add(
            "ok" if has_key else "warn",
            key_env,
            "set" if has_key else "not set",
            ""
            if has_key
            else (
                f"No {key_env} set. Set it for provider {provider!r}, or run "
                "locally without a key via Ollama (see the README)."
            ),
            section="LLM",
        )
    else:
        add(
            "warn",
            "Credentials",
            f"unknown provider {provider!r}",
            "Set EPISCOPE_LLM_PROVIDER to one of: gemini, openai, openrouter, anthropic, ollama.",
            section="LLM",
        )

    # Services (optional for the local CLI path) ---------------------------
    if probe:
        ok, code, err = _probe_http(f"{settings.grobid_url}/api/isalive")
        add(
            "ok" if ok else "warn",
            "GROBID",
            f"{settings.grobid_url} (HTTP {code})"
            if ok
            else f"{settings.grobid_url} ({err})",
            ""
            if ok
            else (
                "Only needed for --loader grobid; the default "
                "'unstructured' loader works without it."
            ),
            section="Services",
        )

        ok, code, err = _probe_http(
            f"{settings.qdrant_url}/collections/{settings.qdrant_collection}"
        )
        if ok and code == 200:
            add(
                "ok",
                "Qdrant",
                f"{settings.qdrant_url} "
                f"(collection {settings.qdrant_collection!r} present)",
                section="Services",
            )
        elif ok:
            add(
                "warn",
                "Qdrant",
                f"{settings.qdrant_url} "
                f"(collection {settings.qdrant_collection!r} missing, HTTP {code})",
                "Only needed for the API/UI corpus path; the local CLI uses a file index.",
                section="Services",
            )
        else:
            add(
                "warn",
                "Qdrant",
                f"{settings.qdrant_url} ({err})",
                "Only needed for the API/UI corpus path; the local CLI uses a file index.",
                section="Services",
            )

        if settings.mongo_uri:
            status, detail, hint = _check_mongo(settings.mongo_uri)
            add(status, "MongoDB", detail, hint, section="Services")
        else:
            add(
                "warn",
                "MongoDB",
                "MONGO_URI not set",
                "Only needed for the API/UI corpus path; the local CLI uses an in-memory store.",
                section="Services",
            )
    else:
        add("skip", "Service probes", "skipped (--no-probe)", section="Services")

    # Workspace ------------------------------------------------------------
    ws = None
    ws_error: Optional[Exception] = None
    try:
        ws = _optional_workspace(workspace)
    except Exception as exc:
        ws_error = exc
    if ws is not None:
        add("ok", "Workspace", str(ws.root), section="Workspace")
    elif ws_error is not None:
        add(
            "warn",
            "Workspace",
            f"could not load ({type(ws_error).__name__})",
            section="Workspace",
        )
    else:
        add(
            "skip",
            "Workspace",
            "none discovered",
            "Optional. Run `episcope init <name>` to create one.",
            section="Workspace",
        )

    any_fail = any(check["status"] == "fail" for check in checks)
    return checks, any_fail


def _print_doctor_report(checks: list[dict[str, Any]], any_fail: bool) -> None:
    current_section: Optional[str] = None
    for check in checks:
        if check["section"] != current_section:
            current_section = check["section"]
            typer.echo("")
            typer.secho(current_section, bold=True)
        _print_doctor_check(
            check["status"], check["label"], check["detail"], check["hint"]
        )

    typer.echo("")
    if any_fail:
        typer.secho(
            "Some required checks failed. See the hints above.", fg=typer.colors.RED
        )
    else:
        typer.secho("All required checks passed.", fg=typer.colors.GREEN)


@app.command()
def doctor(
    probe: bool = typer.Option(
        True,
        "--probe/--no-probe",
        help="Probe optional services (Qdrant, GROBID, MongoDB, Ollama) over the network.",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Check the local environment and report what is and is not ready to use."""
    checks, any_fail = _collect_doctor_checks(probe, workspace)

    if output_format == OutputFormat.json:
        _echo_json({"ok": not any_fail, "checks": checks})
        raise typer.Exit(code=1 if any_fail else 0)

    _print_doctor_report(checks, any_fail)
    if any_fail:
        raise typer.Exit(code=1)


_PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@app.command()
def quickstart(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory to check at the end (see `episcope init`).",
    ),
) -> None:
    """Interactively choose an LLM provider, write a .env file, and check the result.

    Only touches your LLM provider configuration - it does not index anything
    or make any model/API calls itself. Run `episcope doctor` any time
    afterward to re-check your environment.
    """
    import dotenv

    typer.secho("EpiScope quickstart", bold=True)
    typer.echo("Let's configure an LLM provider and write a .env file.\n")

    provider_choices = list(_PROVIDER_KEY_ENV) + ["ollama"]
    typer.echo("Choose an LLM provider:")
    for i, choice in enumerate(provider_choices, start=1):
        note = "  (local, no API key needed)" if choice == "ollama" else ""
        typer.echo(f"  {i}. {choice}{note}")
    raw_choice = typer.prompt("Enter a number", default="1")
    try:
        provider = provider_choices[int(raw_choice) - 1]
    except (ValueError, IndexError):
        _abort(f"Invalid choice {raw_choice!r}. Run `episcope quickstart` again.")

    env_path = Path(".env")
    if not env_path.exists():
        example_path = Path(".env.example")
        env_path.write_text(
            example_path.read_text(encoding="utf-8") if example_path.exists() else "",
            encoding="utf-8",
        )

    dotenv.set_key(str(env_path), "EPISCOPE_LLM_PROVIDER", provider)
    typer.echo(f"\nSet EPISCOPE_LLM_PROVIDER={provider} in {env_path}.")

    if provider == "ollama":
        ok, _code, _err = _probe_http(f"{_SETTINGS.ollama_host}/api/tags")
        if ok:
            typer.secho(
                f"Ollama is reachable at {_SETTINGS.ollama_host}.",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho("Ollama is not reachable yet.", fg=typer.colors.YELLOW)
            typer.echo(_ollama_setup_hint(_OLLAMA_SMALL_MODEL))
    else:
        key_env = _PROVIDER_KEY_ENV[provider]
        has_key = bool(env(key_env))
        prompt_label = f"Enter your {key_env}"
        if has_key:
            prompt_label += " (already set - leave blank to keep it)"
        key_value = typer.prompt(
            prompt_label, default="", hide_input=True, show_default=False
        )
        if key_value:
            dotenv.set_key(str(env_path), key_env, key_value)
            typer.echo(f"Set {key_env} in {env_path}.")
        elif not has_key:
            typer.secho(
                f"No {key_env} provided. Add it to {env_path} before using "
                f"provider {provider!r}.",
                fg=typer.colors.YELLOW,
            )

    dotenv.load_dotenv(str(env_path), override=True)

    typer.echo()
    typer.secho("Checking your setup...", bold=True)
    checks, any_fail = _collect_doctor_checks(probe=True, workspace=workspace)
    _print_doctor_report(checks, any_fail)

    typer.echo()
    if any_fail:
        typer.secho(
            "Some checks still need attention - see the hints above.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.secho("You're ready. Try:", fg=typer.colors.GREEN)
        typer.echo(
            '  episcope ask "What data sources were used?" --path /path/to/paper.pdf'
        )


@app.command("init")
def init_workspace(
    path: Path = typer.Argument(
        Path("."),
        help="Workspace directory to create. Defaults to the current directory.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Human-readable workspace name. Defaults to the directory name.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite an existing episcope.toml in the workspace directory.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Create an EpiScope workspace with local-first defaults."""
    try:
        workspace = create_workspace(path, name=name, force=force)
    except Exception as exc:
        _abort_exc(exc)

    _emit(
        {
            "status": "ok",
            "workspace": str(workspace.root),
            "config": str(workspace.config_path),
            "papers_dir": str(workspace.resolve_path(workspace.papers_dir)),
            "index_dir": str(workspace.resolve_path(workspace.index_dir)),
            "metadata_path": str(workspace.resolve_path(workspace.metadata_path)),
            "outputs_dir": str(workspace.resolve_path(workspace.outputs_dir)),
            "logs_dir": str(workspace.resolve_path(workspace.logs_dir)),
        },
        output_format,
        _human_init,
    )


@app.command("inspect")
def inspect_document(
    file_path: Path = typer.Argument(..., exists=True, help="Document file to parse."),
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        help="Parsing backend. Unstructured is the local-first default.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Parse a file and print its extracted metadata and section counts."""
    try:
        paper = _load_paper(file_path, loader_kind=loader)
    except Exception as exc:
        _abort_exc(exc)

    _emit(
        {
            "paper_id": paper.paper_id,
            "path": str(paper.path),
            "metadata": paper.metadata,
            "section_count": len(paper.sections),
            "reference_count": len(paper.references),
            "section_titles": [section.title for section in paper.sections[:10]],
        },
        output_format,
        _human_inspect,
    )


@app.command()
def index(
    path: Path = typer.Argument(..., exists=True, help="File or directory to index."),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    paper_id: Optional[str] = typer.Option(
        None,
        "--paper-id",
        help="Override the paper id when indexing a single file.",
    ),
    strategy_name: str = typer.Option(
        _DEFAULT_STRATEGY_NAME,
        "--strategy-name",
        help="Namespace used for stored metadata and extracted content.",
    ),
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        help="Parsing backend. Unstructured avoids requiring a separate GROBID service.",
    ),
    index_backend: IndexBackend = typer.Option(
        IndexBackend.file,
        "--index-backend",
        help="Vector backend. File-backed dense search is the local-first default.",
    ),
    index_dir: Path = typer.Option(
        _DEFAULT_INDEX_DIR,
        "--index-dir",
        help="Directory for file/faiss indexes.",
    ),
    qdrant_url: str = typer.Option(
        _SETTINGS.qdrant_url,
        "--qdrant-url",
        help="Qdrant URL when --index-backend=qdrant.",
    ),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection,
        "--qdrant-collection",
        help="Qdrant collection when --index-backend=qdrant.",
    ),
    metadata_backend: MetadataBackend = typer.Option(
        MetadataBackend.memory,
        "--metadata-backend",
        help="Metadata store. JSON-backed in-memory storage is the local-first default.",
    ),
    db_backup: Path = typer.Option(
        _DEFAULT_DB_BACKUP,
        "--db-backup",
        help="JSON file used by the local metadata store.",
    ),
    mongo_uri: Optional[str] = typer.Option(
        None,
        "--mongo-uri",
        help="Mongo URI when --metadata-backend=mongo. Defaults to the MONGO_URI environment variable.",
        show_default=False,
    ),
    mongo_db_name: str = typer.Option(
        _SETTINGS.mongo_db_name,
        "--mongo-db-name",
        help="Mongo database name when --metadata-backend=mongo.",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Dense embedding model used for indexing. The default is local-first.",
    ),
    embed_provider: str = typer.Option(
        _SETTINGS.embed_provider,
        "--embed-provider",
        help="Embedding provider: auto (default), huggingface, openai, gemini, or ollama.",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        help="Chunking strategy used before indexing.",
    ),
    min_chunk_size: int = typer.Option(
        _DEFAULT_MIN_CHUNK_SIZE,
        "--min-chunk-size",
        help="Minimum paragraph length when --chunker=paragraph.",
    ),
    chunk_size: int = typer.Option(
        600,
        "--chunk-size",
        help="Chunk size when --chunker=fixed-size.",
    ),
    chunk_overlap: int = typer.Option(
        100,
        "--chunk-overlap",
        help="Chunk overlap when --chunker=fixed-size.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Index documents into local or Qdrant-backed storage and persist metadata."""
    try:
        workspace_config = _optional_workspace(workspace)
        strategy_name = _workspace_value(
            workspace_config,
            strategy_name,
            _DEFAULT_STRATEGY_NAME,
            workspace_config.strategy_name if workspace_config else None,
        )
        loader = _workspace_enum(
            workspace_config,
            LoaderKind,
            loader,
            LoaderKind.unstructured,
            workspace_config.loader if workspace_config else "",
        )
        index_backend = _workspace_enum(
            workspace_config,
            IndexBackend,
            index_backend,
            IndexBackend.file,
            workspace_config.index_backend if workspace_config else "",
        )
        index_dir = _workspace_path(
            workspace_config,
            index_dir,
            _DEFAULT_INDEX_DIR,
            workspace_config.index_dir if workspace_config else "",
        )
        metadata_backend = _workspace_enum(
            workspace_config,
            MetadataBackend,
            metadata_backend,
            MetadataBackend.memory,
            workspace_config.metadata_backend if workspace_config else "",
        )
        db_backup = _workspace_path(
            workspace_config,
            db_backup,
            _DEFAULT_DB_BACKUP,
            workspace_config.metadata_path if workspace_config else "",
        )
        mongo_db_name = _workspace_value(
            workspace_config,
            mongo_db_name,
            _SETTINGS.mongo_db_name,
            workspace_config.mongo_db_name if workspace_config else None,
        )
        qdrant_url = _workspace_value(
            workspace_config,
            qdrant_url,
            _SETTINGS.qdrant_url,
            workspace_config.qdrant_url if workspace_config else None,
        )
        qdrant_collection = _workspace_value(
            workspace_config,
            qdrant_collection,
            _SETTINGS.qdrant_collection,
            workspace_config.qdrant_collection if workspace_config else None,
        )
        embed_model = _workspace_value(
            workspace_config,
            embed_model,
            _DEFAULT_EMBED_MODEL,
            workspace_config.embed_model if workspace_config else None,
        )
        chunker = _workspace_enum(
            workspace_config,
            ChunkerKind,
            chunker,
            ChunkerKind.paragraph,
            workspace_config.chunker if workspace_config else "",
        )
        min_chunk_size = _workspace_value(
            workspace_config,
            min_chunk_size,
            _DEFAULT_MIN_CHUNK_SIZE,
            workspace_config.min_chunk_size if workspace_config else None,
        )
        chunk_size = _workspace_value(
            workspace_config,
            chunk_size,
            600,
            workspace_config.chunk_size if workspace_config else None,
        )
        chunk_overlap = _workspace_value(
            workspace_config,
            chunk_overlap,
            100,
            workspace_config.chunk_overlap if workspace_config else None,
        )
        papers = _load_path(path, loader_kind=loader, paper_id=paper_id)
        embedder = EmbedderFactory.get_embedder(embed_model, provider=embed_provider)
        vectordb = _build_vector_db(
            index_backend,
            index_dir=index_dir,
            qdrant_collection=qdrant_collection,
            qdrant_url=qdrant_url,
            dense_dim=getattr(embedder, "dim", None),
        )
        academic_db = _build_metadata_db(
            metadata_backend,
            db_backup=db_backup,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
        )
        chunker_instance = _build_chunker(
            chunker,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        indexer = Indexer(vectordb, embedder=embedder, chunker=chunker_instance)
        for paper in papers:
            indexer.index_paper(paper.sections, paper.metadata, paper.paper_id)
        if hasattr(vectordb, "save"):
            vectordb.save()
        _persist_papers(papers, db=academic_db, strategy_name=strategy_name)
    except Exception as exc:
        _abort_exc(exc)

    _emit(
        {
            "status": "ok",
            "workspace": str(workspace_config.root) if workspace_config else None,
            "strategy_name": strategy_name,
            "index_backend": index_backend,
            "metadata_backend": metadata_backend,
            "embed_model": embed_model,
            "papers": _paper_summaries(papers),
        },
        output_format,
        _human_index,
    )


@app.command()
def papers(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    strategy_name: str = typer.Option(
        _DEFAULT_STRATEGY_NAME,
        "--strategy-name",
        help="Namespace to inspect.",
    ),
    metadata_backend: MetadataBackend = typer.Option(
        MetadataBackend.memory,
        "--metadata-backend",
        help="Metadata store to inspect.",
    ),
    db_backup: Path = typer.Option(
        _DEFAULT_DB_BACKUP,
        "--db-backup",
        help="JSON file used by the local metadata store.",
    ),
    mongo_uri: Optional[str] = typer.Option(
        None,
        "--mongo-uri",
        help="Mongo URI when --metadata-backend=mongo. Defaults to the MONGO_URI environment variable.",
        show_default=False,
    ),
    mongo_db_name: str = typer.Option(
        _SETTINGS.mongo_db_name,
        "--mongo-db-name",
        help="Mongo database name when --metadata-backend=mongo.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """List available paper ids in the configured metadata store."""
    try:
        workspace_config = _optional_workspace(workspace)
        strategy_name = _workspace_value(
            workspace_config,
            strategy_name,
            _DEFAULT_STRATEGY_NAME,
            workspace_config.strategy_name if workspace_config else None,
        )
        metadata_backend = _workspace_enum(
            workspace_config,
            MetadataBackend,
            metadata_backend,
            MetadataBackend.memory,
            workspace_config.metadata_backend if workspace_config else "",
        )
        db_backup = _workspace_path(
            workspace_config,
            db_backup,
            _DEFAULT_DB_BACKUP,
            workspace_config.metadata_path if workspace_config else "",
        )
        mongo_db_name = _workspace_value(
            workspace_config,
            mongo_db_name,
            _SETTINGS.mongo_db_name,
            workspace_config.mongo_db_name if workspace_config else None,
        )
        academic_db = _build_metadata_db(
            metadata_backend,
            db_backup=db_backup,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
        )
        doc_ids = academic_db.list_docs(strategy_name)
    except Exception as exc:
        _abort_exc(exc)

    _emit(
        {
            "workspace": str(workspace_config.root) if workspace_config else None,
            "strategy_name": strategy_name,
            "metadata_backend": metadata_backend,
            "count": len(doc_ids),
            "paper_ids": doc_ids,
        },
        output_format,
        _human_papers,
    )


@app.command()
def explore(
    query: str = typer.Argument(..., help="Question or search query."),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        exists=True,
        help="Index a file or directory on the fly for this query.",
    ),
    paper_id: Optional[str] = typer.Option(
        None,
        "--paper-id",
        help="Restrict retrieval to a single indexed paper id.",
    ),
    quality: Optional[QualityPreset] = typer.Option(
        None,
        "--quality",
        help=(
            "Preset for chunking/retrieval: fast (fewer, larger chunks), "
            "balanced, or accurate (smaller chunks, hybrid retrieval). "
            "Any of the advanced flags below still overrides its part of "
            "the preset when set explicitly."
        ),
    ),
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        help="Parsing backend used only with --path.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Dense embedding model used only with --path.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        help="Chunking strategy used only with --path.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    min_chunk_size: int = typer.Option(
        _DEFAULT_MIN_CHUNK_SIZE,
        "--min-chunk-size",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunk_size: int = typer.Option(
        600, "--chunk-size", rich_help_panel="Advanced (retrieval internals)"
    ),
    chunk_overlap: int = typer.Option(
        100, "--chunk-overlap", rich_help_panel="Advanced (retrieval internals)"
    ),
    index_backend: IndexBackend = typer.Option(
        IndexBackend.file,
        "--index-backend",
        help="Backend used for previously indexed corpora.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    index_dir: Path = typer.Option(
        _DEFAULT_INDEX_DIR,
        "--index-dir",
        help="Directory for previously built file/faiss indexes.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    qdrant_url: str = typer.Option(
        _SETTINGS.qdrant_url,
        "--qdrant-url",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection,
        "--qdrant-collection",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    retrieval_mode: RetrievalMode = typer.Option(
        RetrievalMode.dense_only,
        "--retrieval-mode",
        help="Dense-only is the most local-friendly mode.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    top_k: int = typer.Option(5, "--top-k", min=1),
    similarity_threshold: float = typer.Option(0.0, "--similarity-threshold"),
    generate_answer: bool = typer.Option(
        False,
        "--generate-answer/--retrieval-only",
        help="Retrieval-only is the default so the command can run without an LLM.",
    ),
    llm_provider: LLMProvider = typer.Option(
        LLMProvider.nollm,
        "--llm-provider",
        help="Used only when --generate-answer is enabled. `nollm` is the least-assumptive default.",
    ),
    llm_model: Optional[str] = typer.Option(
        None,
        "--llm-model",
        help="Optional explicit generation model.",
    ),
    temperature: float = typer.Option(0.0, "--temperature"),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Retrieve relevant chunks from a transient or previously indexed corpus."""
    tempdir = None
    try:
        workspace_config = _optional_workspace(workspace)
        preset = _QUALITY_PRESETS[quality] if quality is not None else None
        retrieval_mode_explicit = retrieval_mode != RetrievalMode.dense_only
        loader = _workspace_enum(
            workspace_config,
            LoaderKind,
            loader,
            LoaderKind.unstructured,
            workspace_config.loader if workspace_config else "",
        )
        embed_model = _workspace_value(
            workspace_config,
            embed_model,
            _DEFAULT_EMBED_MODEL,
            workspace_config.embed_model if workspace_config else None,
        )
        chunker = _resolve_enum(
            chunker,
            ChunkerKind.paragraph,
            ChunkerKind,
            quality,
            preset.chunker if preset else None,
            workspace_config,
            workspace_config.chunker if workspace_config else "",
        )
        min_chunk_size = _resolve_value(
            min_chunk_size,
            _DEFAULT_MIN_CHUNK_SIZE,
            quality,
            preset.min_chunk_size if preset else None,
            workspace_config,
            workspace_config.min_chunk_size if workspace_config else None,
        )
        chunk_size = _resolve_value(
            chunk_size,
            600,
            quality,
            preset.chunk_size if preset else None,
            workspace_config,
            workspace_config.chunk_size if workspace_config else None,
        )
        chunk_overlap = _resolve_value(
            chunk_overlap,
            100,
            quality,
            preset.chunk_overlap if preset else None,
            workspace_config,
            workspace_config.chunk_overlap if workspace_config else None,
        )
        index_backend = _workspace_enum(
            workspace_config,
            IndexBackend,
            index_backend,
            IndexBackend.file,
            workspace_config.index_backend if workspace_config else "",
        )
        index_dir = _workspace_path(
            workspace_config,
            index_dir,
            _DEFAULT_INDEX_DIR,
            workspace_config.index_dir if workspace_config else "",
        )
        qdrant_url = _workspace_value(
            workspace_config,
            qdrant_url,
            _SETTINGS.qdrant_url,
            workspace_config.qdrant_url if workspace_config else None,
        )
        qdrant_collection = _workspace_value(
            workspace_config,
            qdrant_collection,
            _SETTINGS.qdrant_collection,
            workspace_config.qdrant_collection if workspace_config else None,
        )
        retrieval_mode = _resolve_enum(
            retrieval_mode,
            RetrievalMode.dense_only,
            RetrievalMode,
            quality,
            preset.retrieval_mode if preset else None,
            workspace_config,
            workspace_config.retrieval_mode if workspace_config else "",
        )
        llm_provider = _workspace_enum(
            workspace_config,
            LLMProvider,
            llm_provider,
            LLMProvider.nollm,
            workspace_config.llm_provider if workspace_config else "",
        )
        if workspace_config is not None and llm_model is None:
            llm_model = workspace_config.llm_model
        if path is not None:
            if retrieval_mode != RetrievalMode.dense_only:
                if not retrieval_mode_explicit:
                    # A --quality preset (not the user) asked for
                    # hybrid/sparse retrieval; --path mode only ever builds
                    # a dense-only local index, so use it silently rather
                    # than erroring on a choice the user never made.
                    retrieval_mode = RetrievalMode.dense_only
                else:
                    raise ValueError(
                        "--path mode only supports --retrieval-mode dense-only."
                    )
            tempdir, papers, retriever = _transient_retriever_for_path(
                path,
                loader_kind=loader,
                embed_model=embed_model,
                chunker_kind=chunker,
                min_chunk_size=min_chunk_size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            if paper_id is not None and paper_id not in {
                paper.paper_id for paper in papers
            }:
                raise ValueError(
                    f"Paper {paper_id!r} was not found under transient --path input."
                )
        else:
            vectordb = _build_vector_db(
                index_backend,
                index_dir=index_dir,
                qdrant_collection=qdrant_collection,
                qdrant_url=qdrant_url,
            )
            retriever = _build_retriever(vectordb, retrieval_mode=retrieval_mode)

        filter_payload = {"paper_id": paper_id} if paper_id else None
        results = list(
            retriever.retrieve(
                query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filter=filter_payload,
            )
        )
        payload: dict[str, Any] = {
            "query": query,
            "workspace": str(workspace_config.root) if workspace_config else None,
            "paper_id": paper_id,
            "retrieval_mode": retrieval_mode,
            "retrieval_count": len(results),
            "retrieved_chunks": results,
            "answer": None,
            "provenance": None,
        }
        if generate_answer:
            generator = _resolve_generator(
                llm_provider, llm_model, temperature, task="ask"
            )
            provenance = generator.generate(results, question=query)
            payload["answer"] = provenance.answer
            payload["provenance"] = provenance
        _emit(payload, output_format, _human_explore)
    except Exception as exc:
        _abort_exc(exc)
    finally:
        if tempdir is not None:
            tempdir.cleanup()


@app.command()
def ask(
    question: str = typer.Argument(
        ..., help="Question to answer from a local document or folder."
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        exists=True,
        help="File or directory to index temporarily for this question.",
    ),
    quality: Optional[QualityPreset] = typer.Option(
        None,
        "--quality",
        help=(
            "Preset for chunking/retrieval: fast (fewer, larger chunks), "
            "balanced, or accurate (smaller chunks, hybrid retrieval). "
            "Any of the advanced flags below still overrides its part of "
            "the preset when set explicitly."
        ),
    ),
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        help="Parsing backend. Use `grobid` when a running GROBID service is available.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Dense embedding model used for the temporary local index.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    min_chunk_size: int = typer.Option(
        _DEFAULT_MIN_CHUNK_SIZE,
        "--min-chunk-size",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunk_size: int = typer.Option(
        600, "--chunk-size", rich_help_panel="Advanced (retrieval internals)"
    ),
    chunk_overlap: int = typer.Option(
        100, "--chunk-overlap", rich_help_panel="Advanced (retrieval internals)"
    ),
    top_k: int = typer.Option(5, "--top-k", min=1),
    llm_provider: LLMProvider = typer.Option(
        LLMProvider.gemini,
        "--llm-provider",
        help="Generation provider. Use `ollama` with --llm-model for local generation.",
    ),
    llm_model: Optional[str] = typer.Option(None, "--llm-model"),
    temperature: float = typer.Option(0.0, "--temperature"),
    show_context: bool = typer.Option(
        False,
        "--show-context/--answer-only",
        help="Include full retrieved source chunks in the JSON output.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Ask a question over local papers without requiring MongoDB or Qdrant."""
    tempdir = None
    try:
        workspace_config = _optional_workspace(workspace)
        preset = _QUALITY_PRESETS[quality] if quality is not None else None
        loader = _workspace_enum(
            workspace_config,
            LoaderKind,
            loader,
            LoaderKind.unstructured,
            workspace_config.loader if workspace_config else "",
        )
        embed_model = _workspace_value(
            workspace_config,
            embed_model,
            _DEFAULT_EMBED_MODEL,
            workspace_config.embed_model if workspace_config else None,
        )
        chunker = _resolve_enum(
            chunker,
            ChunkerKind.paragraph,
            ChunkerKind,
            quality,
            preset.chunker if preset else None,
            workspace_config,
            workspace_config.chunker if workspace_config else "",
        )
        min_chunk_size = _resolve_value(
            min_chunk_size,
            _DEFAULT_MIN_CHUNK_SIZE,
            quality,
            preset.min_chunk_size if preset else None,
            workspace_config,
            workspace_config.min_chunk_size if workspace_config else None,
        )
        chunk_size = _resolve_value(
            chunk_size,
            600,
            quality,
            preset.chunk_size if preset else None,
            workspace_config,
            workspace_config.chunk_size if workspace_config else None,
        )
        chunk_overlap = _resolve_value(
            chunk_overlap,
            100,
            quality,
            preset.chunk_overlap if preset else None,
            workspace_config,
            workspace_config.chunk_overlap if workspace_config else None,
        )
        llm_provider = _workspace_enum(
            workspace_config,
            LLMProvider,
            llm_provider,
            LLMProvider.gemini,
            workspace_config.llm_provider if workspace_config else "",
        )
        if workspace_config is not None and llm_model is None:
            llm_model = workspace_config.llm_model

        papers: list[LoadedPaper] = []
        if path is not None:
            tempdir, papers, retriever = _transient_retriever_for_path(
                path,
                loader_kind=loader,
                embed_model=embed_model,
                chunker_kind=chunker,
                min_chunk_size=min_chunk_size,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif workspace_config is not None:
            vectordb = _build_vector_db(
                IndexBackend(workspace_config.index_backend),
                index_dir=workspace_config.resolve_path(workspace_config.index_dir),
                qdrant_collection=workspace_config.qdrant_collection,
                qdrant_url=workspace_config.qdrant_url,
            )
            retriever = _build_retriever(
                vectordb,
                retrieval_mode=RetrievalMode(workspace_config.retrieval_mode),
            )
        else:
            raise ValueError("Provide --path or run inside/pass --workspace.")

        results = list(retriever.retrieve(question, top_k=top_k))
        generator = _resolve_generator(
            llm_provider, llm_model, temperature, task="ask"
        )
        provenance = generator.generate(results, question=question)

        source_summaries = [
            {
                "paper_id": chunk.paper_id,
                "section_type": chunk.section_type,
                "section_title": chunk.section_title,
                "rank_score": chunk.rank_score,
            }
            for chunk in results
        ]
        payload: dict[str, Any] = {
            "question": question,
            "workspace": str(workspace_config.root) if workspace_config else None,
            "answer": provenance.answer,
            "papers": _paper_summaries(papers),
            "source_count": len(results),
            "sources": results if show_context else source_summaries,
        }
        _emit(payload, output_format, _human_ask)
    except Exception as exc:
        _abort_exc(exc)
    finally:
        if tempdir is not None:
            tempdir.cleanup()


@app.command()
def classify(
    paper_id: Optional[str] = typer.Argument(
        None,
        help="Indexed paper id. Omit this when using --file for transient local classification.",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    file_path: Optional[Path] = typer.Option(
        None,
        "--file",
        exists=True,
        help="Single file to classify without requiring a prebuilt index or metadata store.",
    ),
    classifier_kind: str = typer.Option(
        "data_accessibility",
        "--classifier-kind",
        help="Built-in or workspace/declarative task key (see `episcope tasks`).",
    ),
    task_file: Optional[Path] = typer.Option(
        None,
        "--task-file",
        exists=True,
        help="JSON task spec to run for this call (overrides --classifier-kind).",
    ),
    quality: Optional[QualityPreset] = typer.Option(
        None,
        "--quality",
        help=(
            "Preset for chunking/retrieval: fast (fewer, larger chunks), "
            "balanced, or accurate (smaller chunks, hybrid retrieval). "
            "Any of the advanced flags below still overrides its part of "
            "the preset when set explicitly."
        ),
    ),
    strategy_name: str = typer.Option(_DEFAULT_STRATEGY_NAME, "--strategy-name"),
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Used only with --file.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        help="Used only with --file.",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    min_chunk_size: int = typer.Option(
        _DEFAULT_MIN_CHUNK_SIZE,
        "--min-chunk-size",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunk_size: int = typer.Option(
        600, "--chunk-size", rich_help_panel="Advanced (retrieval internals)"
    ),
    chunk_overlap: int = typer.Option(
        100, "--chunk-overlap", rich_help_panel="Advanced (retrieval internals)"
    ),
    index_backend: IndexBackend = typer.Option(
        IndexBackend.file,
        "--index-backend",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    index_dir: Path = typer.Option(
        _DEFAULT_INDEX_DIR,
        "--index-dir",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    qdrant_url: str = typer.Option(
        _SETTINGS.qdrant_url,
        "--qdrant-url",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection,
        "--qdrant-collection",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    metadata_backend: MetadataBackend = typer.Option(
        MetadataBackend.memory,
        "--metadata-backend",
    ),
    db_backup: Path = typer.Option(_DEFAULT_DB_BACKUP, "--db-backup"),
    mongo_uri: Optional[str] = typer.Option(
        None,
        "--mongo-uri",
        help="Mongo URI when --metadata-backend=mongo. Defaults to the MONGO_URI environment variable.",
        show_default=False,
    ),
    mongo_db_name: str = typer.Option(_SETTINGS.mongo_db_name, "--mongo-db-name"),
    retrieval_mode: RetrievalMode = typer.Option(
        RetrievalMode.dense_only,
        "--retrieval-mode",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    workflow_top_k: int = typer.Option(10, "--workflow-top-k", min=1),
    llm_provider: LLMProvider = typer.Option(LLMProvider.gemini, "--llm-provider"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model"),
    temperature: float = typer.Option(0.0, "--temperature"),
    evidence_reranker: EvidenceRerankerKind = typer.Option(
        EvidenceRerankerKind.none,
        "--evidence-reranker",
    ),
    cross_encoder_model: Optional[str] = typer.Option(None, "--cross-encoder-model"),
    cross_encoder_top_k: int = typer.Option(15, "--cross-encoder-top-k", min=1),
    detailed: bool = typer.Option(
        False,
        "--detailed/--compact",
        help="Emit the full trace/training payload instead of only the compact decision.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Run a paper classification workflow with local-first defaults."""
    tempdir = None
    try:
        workspace_config = _optional_workspace(workspace)
        preset = _QUALITY_PRESETS[quality] if quality is not None else None
        retrieval_mode_explicit = retrieval_mode != RetrievalMode.dense_only
        strategy_name = _workspace_value(
            workspace_config,
            strategy_name,
            _DEFAULT_STRATEGY_NAME,
            workspace_config.strategy_name if workspace_config else None,
        )
        loader = _workspace_enum(
            workspace_config,
            LoaderKind,
            loader,
            LoaderKind.unstructured,
            workspace_config.loader if workspace_config else "",
        )
        embed_model = _workspace_value(
            workspace_config,
            embed_model,
            _DEFAULT_EMBED_MODEL,
            workspace_config.embed_model if workspace_config else None,
        )
        chunker = _resolve_enum(
            chunker,
            ChunkerKind.paragraph,
            ChunkerKind,
            quality,
            preset.chunker if preset else None,
            workspace_config,
            workspace_config.chunker if workspace_config else "",
        )
        min_chunk_size = _resolve_value(
            min_chunk_size,
            _DEFAULT_MIN_CHUNK_SIZE,
            quality,
            preset.min_chunk_size if preset else None,
            workspace_config,
            workspace_config.min_chunk_size if workspace_config else None,
        )
        chunk_size = _resolve_value(
            chunk_size,
            600,
            quality,
            preset.chunk_size if preset else None,
            workspace_config,
            workspace_config.chunk_size if workspace_config else None,
        )
        chunk_overlap = _resolve_value(
            chunk_overlap,
            100,
            quality,
            preset.chunk_overlap if preset else None,
            workspace_config,
            workspace_config.chunk_overlap if workspace_config else None,
        )
        index_backend = _workspace_enum(
            workspace_config,
            IndexBackend,
            index_backend,
            IndexBackend.file,
            workspace_config.index_backend if workspace_config else "",
        )
        index_dir = _workspace_path(
            workspace_config,
            index_dir,
            _DEFAULT_INDEX_DIR,
            workspace_config.index_dir if workspace_config else "",
        )
        metadata_backend = _workspace_enum(
            workspace_config,
            MetadataBackend,
            metadata_backend,
            MetadataBackend.memory,
            workspace_config.metadata_backend if workspace_config else "",
        )
        db_backup = _workspace_path(
            workspace_config,
            db_backup,
            _DEFAULT_DB_BACKUP,
            workspace_config.metadata_path if workspace_config else "",
        )
        mongo_db_name = _workspace_value(
            workspace_config,
            mongo_db_name,
            _SETTINGS.mongo_db_name,
            workspace_config.mongo_db_name if workspace_config else None,
        )
        qdrant_url = _workspace_value(
            workspace_config,
            qdrant_url,
            _SETTINGS.qdrant_url,
            workspace_config.qdrant_url if workspace_config else None,
        )
        qdrant_collection = _workspace_value(
            workspace_config,
            qdrant_collection,
            _SETTINGS.qdrant_collection,
            workspace_config.qdrant_collection if workspace_config else None,
        )
        retrieval_mode = _resolve_enum(
            retrieval_mode,
            RetrievalMode.dense_only,
            RetrievalMode,
            quality,
            preset.retrieval_mode if preset else None,
            workspace_config,
            workspace_config.retrieval_mode if workspace_config else "",
        )
        llm_provider = _workspace_enum(
            workspace_config,
            LLMProvider,
            llm_provider,
            LLMProvider.gemini,
            workspace_config.llm_provider if workspace_config else "",
        )
        if workspace_config is not None and llm_model is None:
            llm_model = workspace_config.llm_model
        resolved = _resolve_paper_from_store(
            paper_id or "",
            loader_kind=loader,
            file_path=file_path,
            strategy_name=strategy_name,
            metadata_backend=metadata_backend,
            db_backup=db_backup,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
            index_backend=index_backend,
            index_dir=index_dir,
            qdrant_collection=qdrant_collection,
            qdrant_url=qdrant_url,
            retrieval_mode=retrieval_mode,
            retrieval_mode_explicit=retrieval_mode_explicit,
            embed_model=embed_model,
            chunker_kind=chunker,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        tempdir = resolved["tempdir"]
        generator = _resolve_generator(
            llm_provider, llm_model, temperature, task="classify"
        )
        classifier_kind = _prepare_tasks(
            workspace_config, task_file, classifier_kind, expected="classifier"
        )
        classifier = PaperClassifier(
            retriever=resolved["retriever"],
            generator=generator,
            strategy_name=strategy_name,
            academic_db=resolved["academic_db"],
            config=build_classifier_config(classifier_kind, workflow_top_k),
            evidence_reranker=_build_evidence_reranker(
                evidence_reranker,
                cross_encoder_model=cross_encoder_model,
                cross_encoder_top_k=cross_encoder_top_k,
            ),
        )
        if detailed:
            result = classifier.run_detailed(
                resolved["paper_id"],
                metadata=resolved["metadata"],
            )
        else:
            result = classifier.run(
                resolved["paper_id"],
                metadata=resolved["metadata"],
            )
        _emit(result, output_format, _human_classify)
    except Exception as exc:
        _abort_exc(exc)
    finally:
        if tempdir is not None:
            tempdir.cleanup()


@app.command("precision-miner")
def precision_miner(
    paper_id: Optional[str] = typer.Argument(
        None,
        help="Indexed paper id. Omit this when using --file for transient local extraction.",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml.",
    ),
    file_path: Optional[Path] = typer.Option(
        None,
        "--file",
        exists=True,
        help="Single file to analyze without requiring Mongo or Qdrant.",
    ),
    miner_kind: str = typer.Option(
        "find_data_sources",
        "--miner-kind",
        help="Built-in or workspace/declarative task key (see `episcope tasks`).",
    ),
    task_file: Optional[Path] = typer.Option(
        None,
        "--task-file",
        exists=True,
        help="JSON task spec to run for this call (overrides --miner-kind).",
    ),
    quality: Optional[QualityPreset] = typer.Option(
        None,
        "--quality",
        help=(
            "Preset for chunking/retrieval: fast (fewer, larger chunks), "
            "balanced, or accurate (smaller chunks, hybrid retrieval). "
            "Any of the advanced flags below still overrides its part of "
            "the preset when set explicitly."
        ),
    ),
    strategy_name: str = typer.Option(_DEFAULT_STRATEGY_NAME, "--strategy-name"),
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    min_chunk_size: int = typer.Option(
        _DEFAULT_MIN_CHUNK_SIZE,
        "--min-chunk-size",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    chunk_size: int = typer.Option(
        600, "--chunk-size", rich_help_panel="Advanced (retrieval internals)"
    ),
    chunk_overlap: int = typer.Option(
        100, "--chunk-overlap", rich_help_panel="Advanced (retrieval internals)"
    ),
    index_backend: IndexBackend = typer.Option(
        IndexBackend.file,
        "--index-backend",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    index_dir: Path = typer.Option(
        _DEFAULT_INDEX_DIR,
        "--index-dir",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    qdrant_url: str = typer.Option(
        _SETTINGS.qdrant_url,
        "--qdrant-url",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection,
        "--qdrant-collection",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    metadata_backend: MetadataBackend = typer.Option(
        MetadataBackend.memory,
        "--metadata-backend",
    ),
    db_backup: Path = typer.Option(_DEFAULT_DB_BACKUP, "--db-backup"),
    mongo_uri: Optional[str] = typer.Option(
        None,
        "--mongo-uri",
        help="Mongo URI when --metadata-backend=mongo. Defaults to the MONGO_URI environment variable.",
        show_default=False,
    ),
    mongo_db_name: str = typer.Option(_SETTINGS.mongo_db_name, "--mongo-db-name"),
    retrieval_mode: RetrievalMode = typer.Option(
        RetrievalMode.dense_only,
        "--retrieval-mode",
        rich_help_panel="Advanced (retrieval internals)",
    ),
    workflow_top_k: int = typer.Option(10, "--workflow-top-k", min=1),
    llm_provider: LLMProvider = typer.Option(LLMProvider.gemini, "--llm-provider"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model"),
    temperature: float = typer.Option(0.0, "--temperature"),
    detailed: bool = typer.Option(
        False,
        "--detailed/--compact",
        help="Emit provenance/trace/chunks instead of only the extraction result.",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Run a precision-miner workflow with local-first defaults."""
    tempdir = None
    try:
        workspace_config = _optional_workspace(workspace)
        preset = _QUALITY_PRESETS[quality] if quality is not None else None
        retrieval_mode_explicit = retrieval_mode != RetrievalMode.dense_only
        strategy_name = _workspace_value(
            workspace_config,
            strategy_name,
            _DEFAULT_STRATEGY_NAME,
            workspace_config.strategy_name if workspace_config else None,
        )
        loader = _workspace_enum(
            workspace_config,
            LoaderKind,
            loader,
            LoaderKind.unstructured,
            workspace_config.loader if workspace_config else "",
        )
        embed_model = _workspace_value(
            workspace_config,
            embed_model,
            _DEFAULT_EMBED_MODEL,
            workspace_config.embed_model if workspace_config else None,
        )
        chunker = _resolve_enum(
            chunker,
            ChunkerKind.paragraph,
            ChunkerKind,
            quality,
            preset.chunker if preset else None,
            workspace_config,
            workspace_config.chunker if workspace_config else "",
        )
        min_chunk_size = _resolve_value(
            min_chunk_size,
            _DEFAULT_MIN_CHUNK_SIZE,
            quality,
            preset.min_chunk_size if preset else None,
            workspace_config,
            workspace_config.min_chunk_size if workspace_config else None,
        )
        chunk_size = _resolve_value(
            chunk_size,
            600,
            quality,
            preset.chunk_size if preset else None,
            workspace_config,
            workspace_config.chunk_size if workspace_config else None,
        )
        chunk_overlap = _resolve_value(
            chunk_overlap,
            100,
            quality,
            preset.chunk_overlap if preset else None,
            workspace_config,
            workspace_config.chunk_overlap if workspace_config else None,
        )
        index_backend = _workspace_enum(
            workspace_config,
            IndexBackend,
            index_backend,
            IndexBackend.file,
            workspace_config.index_backend if workspace_config else "",
        )
        index_dir = _workspace_path(
            workspace_config,
            index_dir,
            _DEFAULT_INDEX_DIR,
            workspace_config.index_dir if workspace_config else "",
        )
        metadata_backend = _workspace_enum(
            workspace_config,
            MetadataBackend,
            metadata_backend,
            MetadataBackend.memory,
            workspace_config.metadata_backend if workspace_config else "",
        )
        db_backup = _workspace_path(
            workspace_config,
            db_backup,
            _DEFAULT_DB_BACKUP,
            workspace_config.metadata_path if workspace_config else "",
        )
        mongo_db_name = _workspace_value(
            workspace_config,
            mongo_db_name,
            _SETTINGS.mongo_db_name,
            workspace_config.mongo_db_name if workspace_config else None,
        )
        qdrant_url = _workspace_value(
            workspace_config,
            qdrant_url,
            _SETTINGS.qdrant_url,
            workspace_config.qdrant_url if workspace_config else None,
        )
        qdrant_collection = _workspace_value(
            workspace_config,
            qdrant_collection,
            _SETTINGS.qdrant_collection,
            workspace_config.qdrant_collection if workspace_config else None,
        )
        retrieval_mode = _resolve_enum(
            retrieval_mode,
            RetrievalMode.dense_only,
            RetrievalMode,
            quality,
            preset.retrieval_mode if preset else None,
            workspace_config,
            workspace_config.retrieval_mode if workspace_config else "",
        )
        llm_provider = _workspace_enum(
            workspace_config,
            LLMProvider,
            llm_provider,
            LLMProvider.gemini,
            workspace_config.llm_provider if workspace_config else "",
        )
        if workspace_config is not None and llm_model is None:
            llm_model = workspace_config.llm_model
        resolved = _resolve_paper_from_store(
            paper_id or "",
            loader_kind=loader,
            file_path=file_path,
            strategy_name=strategy_name,
            metadata_backend=metadata_backend,
            db_backup=db_backup,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
            index_backend=index_backend,
            index_dir=index_dir,
            qdrant_collection=qdrant_collection,
            qdrant_url=qdrant_url,
            retrieval_mode=retrieval_mode,
            retrieval_mode_explicit=retrieval_mode_explicit,
            embed_model=embed_model,
            chunker_kind=chunker,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        tempdir = resolved["tempdir"]
        generator = _resolve_generator(
            llm_provider, llm_model, temperature, task="precision-miner"
        )
        miner_kind = _prepare_tasks(
            workspace_config, task_file, miner_kind, expected="miner"
        )
        miner = PrecisionMiner(
            retriever=resolved["retriever"],
            generator=generator,
            strategy_name=strategy_name,
            academic_db=resolved["academic_db"],
            config=build_precision_miner_config(miner_kind, workflow_top_k),
        )
        if detailed:
            result = miner.run_detailed(
                resolved["paper_id"],
                metadata=resolved["metadata"],
            )
        else:
            result = {
                "paper_id": resolved["paper_id"],
                "result": miner.run(
                    resolved["paper_id"],
                    metadata=resolved["metadata"],
                ),
            }
        _emit(result, output_format, _human_precision_miner)
    except Exception as exc:
        _abort_exc(exc)
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def _prompt_task_key() -> str:
    while True:
        key = typer.prompt("Task key (lowercase letters, digits, underscores)")
        if re.fullmatch(r"[a-z0-9_]+", key):
            return key
        typer.secho(
            "Key must contain only lowercase letters, digits, and underscores.",
            fg=typer.colors.RED,
        )


def _prompt_optional_int(message: str) -> Optional[int]:
    while True:
        raw = typer.prompt(message, default="", show_default=False).strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            typer.secho(f"{raw!r} is not a whole number.", fg=typer.colors.RED)


def _prompt_classifier_labels() -> list[Dict[str, Any]]:
    labels: list[Dict[str, Any]] = []
    typer.echo("\nNow define at least one label. Leave the code blank when done.")
    while True:
        prompt = f"  Label {len(labels) + 1} code" + ("" if not labels else " (blank to finish)")
        code = typer.prompt(prompt, default="", show_default=False).strip()
        if not code:
            if labels:
                return labels
            typer.secho("At least one label is required.", fg=typer.colors.RED)
            continue
        if any(existing["code"] == code for existing in labels):
            typer.secho(f"Code {code!r} is already used.", fg=typer.colors.RED)
            continue
        name = typer.prompt("    Name", default=code)
        definition = typer.prompt(
            "    Definition (injected into the prompt)", default="", show_default=False
        )
        examples_raw = typer.prompt(
            "    Example sentences (comma-separated, optional)",
            default="",
            show_default=False,
        )
        examples = [item.strip() for item in examples_raw.split(",") if item.strip()]
        labels.append(
            {"code": code, "name": name, "definition": definition, "examples": examples}
        )


def _prompt_default_label(codes: list[str]) -> Optional[str]:
    typer.echo("\nChoose a default label for when classification is unclear:")
    typer.echo("  0. (none)")
    for index, code in enumerate(codes, start=1):
        typer.echo(f"  {index}. {code}")
    while True:
        raw = typer.prompt("Enter a number", default="0")
        try:
            choice = int(raw)
        except ValueError:
            typer.secho("Enter a number from the list.", fg=typer.colors.RED)
            continue
        if choice == 0:
            return None
        if 1 <= choice <= len(codes):
            return codes[choice - 1]
        typer.secho("Out of range.", fg=typer.colors.RED)


def _prompt_miner_templates() -> list[str]:
    templates: list[str] = []
    typer.echo(
        "\nNow enter retrieval query templates (one per line). Leave blank to finish."
    )
    while True:
        prompt = f"  Template {len(templates) + 1}" + ("" if not templates else " (blank to finish)")
        raw = typer.prompt(prompt, default="", show_default=False).strip()
        if not raw:
            if templates:
                return templates
            typer.secho("At least one retrieval template is required.", fg=typer.colors.RED)
            continue
        templates.append(raw)


def _run_task_wizard(kind: str, workspace_config: Optional[WorkspaceConfig]) -> None:
    if kind not in ("classifier", "miner"):
        _abort(f"--kind must be 'classifier' or 'miner', got {kind!r}.")
        return  # unreachable; satisfies the type checker

    typer.secho(f"Creating a new {kind} task", bold=True)
    key = _prompt_task_key()
    label = typer.prompt("Human-readable name (optional)", default="", show_default=False)
    description = typer.prompt("Description (optional)", default="", show_default=False)
    top_k = _prompt_optional_int("Default retrieval depth (top_k, optional)")

    data: Dict[str, Any] = {
        "key": key,
        "kind": kind,
        "label": label,
        "description": description,
        "top_k": top_k,
    }

    if kind == "classifier":
        labels = _prompt_classifier_labels()
        multi_label = typer.confirm("\nAllow multiple labels per paper?", default=True)
        default_label = _prompt_default_label([lbl["code"] for lbl in labels])
        data.update(
            {"labels": labels, "multi_label": multi_label, "default_label": default_label}
        )
    else:
        templates = _prompt_miner_templates()
        section_filters_raw = typer.prompt(
            "\nSection filters (comma-separated, optional)", default="", show_default=False
        )
        section_filters = [
            item.strip() for item in section_filters_raw.split(",") if item.strip()
        ] or None
        data.update({"retrieval_templates": templates, "section_filters": section_filters})

    spec = TaskSpec.model_validate(data)

    if workspace_config is not None:
        default_dir = workspace_config.resolve_path("tasks")
    else:
        default_dir = Path(".")
    default_path = default_dir / f"{spec.key}.json"
    output_path = Path(typer.prompt("\nSave to", default=str(default_path)))
    _ensure_parent_dir(output_path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    typer.secho(
        f"\nSaved {spec.kind} task {spec.key!r} to {output_path}", fg=typer.colors.GREEN
    )
    if workspace_config is not None and output_path.is_relative_to(default_dir):
        typer.echo(
            "This is a workspace task - it will be auto-loaded next time you run "
            "a command in this workspace."
        )
    else:
        example_cmd = "classify" if kind == "classifier" else "precision-miner"
        typer.echo(f"Run it with: episcope {example_cmd} --task-file {output_path}")


@app.command("tasks")
def tasks(
    action: str = typer.Argument(
        "list",
        help=(
            "Action to perform: "
            "'list' (default) — show available kinds; "
            "'new' — print a filled-in JSON scaffold for a new task; "
            "'validate' — check a task file without running anything."
        ),
    ),
    kind: Optional[str] = typer.Option(
        None,
        "--kind",
        help="Task kind for 'new': 'classifier' or 'miner'.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        help="For 'new': build the task by answering prompts instead of "
        "printing a scaffold to edit by hand.",
    ),
    task_file: Optional[Path] = typer.Option(
        None,
        "--task-file",
        exists=True,
        help="JSON task spec for 'validate'.",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory containing episcope.toml (used by 'list' and "
        "'new --interactive').",
    ),
    output_format: OutputFormat = _format_option(),
) -> None:
    """Manage declarative task specs (list / new / validate).

    \b
    Examples:
      episcope tasks                                  # list all kinds
      episcope tasks new --kind classifier            # print a scaffold
      episcope tasks new --kind miner > my.json       # save it
      episcope tasks new --kind classifier --interactive  # answer prompts instead
      episcope tasks validate --task-file my.json     # lint without running
    """
    try:
        if action == "list":
            workspace_config = _optional_workspace(workspace)
            if workspace_config is not None:
                load_tasks_dir(workspace_config.root / "tasks", overwrite=True)
            _emit(
                {"classifiers": classifier_catalog(), "miners": miner_catalog()},
                output_format,
                _human_tasks,
            )

        elif action == "new":
            if kind is None:
                _abort("--kind is required for 'tasks new'. Choose 'classifier' or 'miner'.")
                return  # unreachable; satisfies the type checker
            if interactive:
                workspace_config = _optional_workspace(workspace)
                _run_task_wizard(kind, workspace_config)
            else:
                typer.echo(scaffold_task_spec(kind))

        elif action == "validate":
            if task_file is None:
                _abort("--task-file is required for 'tasks validate'.")
                return  # unreachable; satisfies the type checker
            spec = validate_task_file(task_file)
            msg = (
                f"OK  {task_file.name}\n"
                f"    kind={spec.kind}  key={spec.key!r}  label={spec.label!r}"
            )
            if spec.kind == "classifier":
                msg += f"\n    labels ({len(spec.labels)}): {', '.join(lbl.code for lbl in spec.labels)}"
                msg += f"\n    multi_label={spec.multi_label}  default_label={spec.default_label!r}"
            else:
                msg += f"\n    retrieval_templates: {len(spec.retrieval_templates)}"
            typer.secho(msg, fg=typer.colors.GREEN)

        else:
            _abort(
                f"Unknown action {action!r}. Choose 'list', 'new', or 'validate'."
            )
    except Exception as exc:
        _abort_exc(exc)


@app.command()
def serve(
    host: str = typer.Option(_SETTINGS.api_host, "--host"),
    port: int = typer.Option(_SETTINGS.api_port, "--port"),
    reload: bool = typer.Option(False, "--reload/--no-reload"),
) -> None:
    """Run the FastAPI app."""
    try:
        import uvicorn
    except ImportError as exc:
        _abort(f"uvicorn is required to run the API server: {exc}")
    uvicorn.run("episcope.api:app", host=host, port=port, reload=reload)


def _index_dir_is_empty(index_dir: Path) -> bool:
    try:
        return FileDB(str(index_dir)).get_embedding_model() is None
    except Exception:
        return True


@app.command()
def studio(
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace whose index/metadata to serve. Without one, uses the "
        "current directory's local `.episcope/` store.",
    ),
    host: str = typer.Option(_SETTINGS.api_host, "--host"),
    api_port: int = typer.Option(_SETTINGS.api_port, "--api-port"),
    ui_port: int = typer.Option(8501, "--ui-port"),
) -> None:
    """Run the API and Streamlit UI together against a local, file-backed
    index and in-memory metadata store - no Qdrant or MongoDB required.

    Does not index anything itself; run `episcope index ... --workspace ...`
    first. If QDRANT_URL or MONGO_URI are already set, those take precedence
    over the local store, same as everywhere else in the CLI.
    """
    import os
    import subprocess
    import time
    from importlib import resources

    try:
        import uvicorn  # noqa: F401
    except ImportError as exc:
        _abort(f"uvicorn is required for `studio`: {exc}. Install epi-scope[server].")
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        _abort(f"streamlit is required for `studio`: {exc}. Install epi-scope[ui].")

    workspace_config = _optional_workspace(workspace)
    child_env = dict(os.environ)

    if workspace_config is not None:
        index_dir = workspace_config.resolve_path(workspace_config.index_dir)
        metadata_path = workspace_config.resolve_path(workspace_config.metadata_path)
        if not env("QDRANT_URL"):
            child_env["EPISCOPE_LOCAL_INDEX_DIR"] = str(index_dir)
        if not env("MONGO_URI"):
            child_env["EPISCOPE_LOCAL_METADATA_BACKUP"] = str(metadata_path)
    else:
        index_dir = _CLI_ROOT / "index"

    if not env("QDRANT_URL") and _index_dir_is_empty(index_dir):
        typer.secho(
            "No papers are indexed yet at "
            f"{index_dir} - the Explorer/Classification/Precision Miner tabs "
            "will have nothing to retrieve. Run `episcope index ...` "
            f"{'--workspace ' + str(workspace_config.root) if workspace_config else ''} "
            "first.",
            fg=typer.colors.YELLOW,
        )

    api_url = f"http://localhost:{api_port}"
    ui_url = f"http://localhost:{ui_port}"
    typer.secho("Starting EpiScope studio", bold=True)
    typer.echo(f"  API: {api_url}")
    typer.echo(f"  UI:  {ui_url}")
    typer.echo("Press Ctrl-C to stop both.\n")

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "episcope.api:app",
            "--host",
            host,
            "--port",
            str(api_port),
        ],
        env=child_env,
    )
    ui_app_path = resources.files("episcope.ui").joinpath("streamlit_app.py")
    ui_env = {**child_env, "EPISCOPE_API_BASE_URL": api_url}
    ui_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ui_app_path),
            "--server.port",
            str(ui_port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        env=ui_env,
    )

    try:
        while True:
            if api_proc.poll() is not None:
                typer.secho(
                    f"API process exited unexpectedly (code {api_proc.returncode}).",
                    fg=typer.colors.RED,
                )
                break
            if ui_proc.poll() is not None:
                typer.secho(
                    f"UI process exited unexpectedly (code {ui_proc.returncode}).",
                    fg=typer.colors.RED,
                )
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        typer.echo("\nStopping...")
    finally:
        for proc in (api_proc, ui_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (api_proc, ui_proc):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
