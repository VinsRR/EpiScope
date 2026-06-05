"""Command-line interface for EpiScope."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

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
from episcope.settings import AppSettings, env
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
    DataAccessibilityClassifierConfig,
    DataTypeClassifierConfig,
    GeoClassifierConfig,
    GlobalCrossEncoderReranker,
    PaperTypeClassifierConfig,
    WithinLabelCrossEncoderReranker,
)
from episcope.workflows.precision_miner import (
    FindDataSourcesConfig,
    FindSupplementaryLinksConfig,
    IdentifyKeyReferencesConfig,
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
    ollama = "ollama"


class ClassifierKind(str, Enum):
    paper_type = "paper_type"
    data_accessibility = "data_accessibility"
    data_type = "data_type"
    geo = "geo"


class PrecisionMinerKind(str, Enum):
    find_data_sources = "find_data_sources"
    find_supplementary_links = "find_supplementary_links"
    identify_key_references = "identify_key_references"


class EvidenceRerankerKind(str, Enum):
    none = "none"
    global_cross_encoder = "global_cross_encoder"
    within_label_cross_encoder = "within_label_cross_encoder"


class OutputFormat(str, Enum):
    human = "human"
    json = "json"


@dataclass
class LoadedPaper:
    paper_id: str
    path: Path
    sections: list[StructuredSection]
    metadata: PaperMetadata
    references: list[Reference]


def main() -> None:
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


def _abort(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


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


def _workspace_path(
    workspace: Optional[WorkspaceConfig],
    current: Path,
    default: Path,
    workspace_value: str,
) -> Path:
    if workspace is not None and current == default:
        return workspace.resolve_path(workspace_value)
    return current


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
    elif provider == LLMProvider.ollama:
        client = OllamaClient()
    else:
        raise ValueError(f"Unsupported llm provider: {provider.value}")

    return LLMGenerator(client=client, model=resolved_model, temperature=temperature)


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


def _build_classifier_config(kind: ClassifierKind, top_k: int):
    config_map = {
        ClassifierKind.paper_type: PaperTypeClassifierConfig,
        ClassifierKind.data_accessibility: DataAccessibilityClassifierConfig,
        ClassifierKind.data_type: DataTypeClassifierConfig,
        ClassifierKind.geo: GeoClassifierConfig,
    }
    config = config_map[kind]()
    config.top_k = top_k
    return config


def _build_precision_miner_config(kind: PrecisionMinerKind, top_k: int):
    config_map = {
        PrecisionMinerKind.find_data_sources: FindDataSourcesConfig,
        PrecisionMinerKind.find_supplementary_links: FindSupplementaryLinksConfig,
        PrecisionMinerKind.identify_key_references: IdentifyKeyReferencesConfig,
    }
    config = config_map[kind]()
    config.top_k = top_k
    return config


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
):
    if file_path is not None:
        if retrieval_mode != RetrievalMode.dense_only:
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

    # LLM ------------------------------------------------------------------
    provider = settings.llm_provider
    add("ok", "Provider", provider, section="LLM")
    add("ok", "Model", settings.llm_model, section="LLM")
    key_env = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
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
            "ok" if has_key else "fail",
            key_env,
            "set" if has_key else "not set",
            ""
            if has_key
            else (
                f"Export {key_env} to use ask/classify/precision-miner "
                f"with provider {provider!r}."
            ),
            section="LLM",
        )
    else:
        add(
            "warn",
            "Credentials",
            f"unknown provider {provider!r}",
            "Set EPISCOPE_LLM_PROVIDER to one of: gemini, openai, openrouter, ollama.",
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

    # Report ---------------------------------------------------------------
    any_fail = any(check["status"] == "fail" for check in checks)

    if output_format == OutputFormat.json:
        _echo_json({"ok": not any_fail, "checks": checks})
        raise typer.Exit(code=1 if any_fail else 0)

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
        raise typer.Exit(code=1)
    typer.secho("All required checks passed.", fg=typer.colors.GREEN)


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
        _abort(str(exc))

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
        _abort(str(exc))

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
        embedder = EmbedderFactory.get_embedder(embed_model)
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
        _abort(str(exc))

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
        _abort(str(exc))

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
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        help="Parsing backend used only with --path.",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Dense embedding model used only with --path.",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        help="Chunking strategy used only with --path.",
    ),
    min_chunk_size: int = typer.Option(_DEFAULT_MIN_CHUNK_SIZE, "--min-chunk-size"),
    chunk_size: int = typer.Option(600, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
    index_backend: IndexBackend = typer.Option(
        IndexBackend.file,
        "--index-backend",
        help="Backend used for previously indexed corpora.",
    ),
    index_dir: Path = typer.Option(
        _DEFAULT_INDEX_DIR,
        "--index-dir",
        help="Directory for previously built file/faiss indexes.",
    ),
    qdrant_url: str = typer.Option(_SETTINGS.qdrant_url, "--qdrant-url"),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection,
        "--qdrant-collection",
    ),
    retrieval_mode: RetrievalMode = typer.Option(
        RetrievalMode.dense_only,
        "--retrieval-mode",
        help="Dense-only is the most local-friendly mode.",
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
        retrieval_mode = _workspace_enum(
            workspace_config,
            RetrievalMode,
            retrieval_mode,
            RetrievalMode.dense_only,
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
            generator = _build_generator(llm_provider, llm_model, temperature)
            provenance = generator.generate(results, question=query)
            payload["answer"] = provenance.answer
            payload["provenance"] = provenance
        _emit(payload, output_format, _human_explore)
    except Exception as exc:
        _abort(str(exc))
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
    loader: LoaderKind = typer.Option(
        LoaderKind.unstructured,
        "--loader",
        help="Parsing backend. Use `grobid` when a running GROBID service is available.",
    ),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Dense embedding model used for the temporary local index.",
    ),
    chunker: ChunkerKind = typer.Option(ChunkerKind.paragraph, "--chunker"),
    min_chunk_size: int = typer.Option(_DEFAULT_MIN_CHUNK_SIZE, "--min-chunk-size"),
    chunk_size: int = typer.Option(600, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
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
        generator = _build_generator(llm_provider, llm_model, temperature)
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
        _abort(str(exc))
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
    classifier_kind: ClassifierKind = typer.Option(
        ClassifierKind.data_accessibility,
        "--classifier-kind",
        help="Data accessibility is the most lightweight default workflow.",
    ),
    strategy_name: str = typer.Option(_DEFAULT_STRATEGY_NAME, "--strategy-name"),
    loader: LoaderKind = typer.Option(LoaderKind.unstructured, "--loader"),
    embed_model: str = typer.Option(
        _DEFAULT_EMBED_MODEL,
        "--embed-model",
        help="Used only with --file.",
    ),
    chunker: ChunkerKind = typer.Option(
        ChunkerKind.paragraph,
        "--chunker",
        help="Used only with --file.",
    ),
    min_chunk_size: int = typer.Option(_DEFAULT_MIN_CHUNK_SIZE, "--min-chunk-size"),
    chunk_size: int = typer.Option(600, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
    index_backend: IndexBackend = typer.Option(IndexBackend.file, "--index-backend"),
    index_dir: Path = typer.Option(_DEFAULT_INDEX_DIR, "--index-dir"),
    qdrant_url: str = typer.Option(_SETTINGS.qdrant_url, "--qdrant-url"),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection, "--qdrant-collection"
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
        retrieval_mode = _workspace_enum(
            workspace_config,
            RetrievalMode,
            retrieval_mode,
            RetrievalMode.dense_only,
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
            embed_model=embed_model,
            chunker_kind=chunker,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        tempdir = resolved["tempdir"]
        generator = _build_generator(llm_provider, llm_model, temperature)
        classifier = PaperClassifier(
            retriever=resolved["retriever"],
            generator=generator,
            strategy_name=strategy_name,
            academic_db=resolved["academic_db"],
            config=_build_classifier_config(classifier_kind, workflow_top_k),
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
        _abort(str(exc))
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
    miner_kind: PrecisionMinerKind = typer.Option(
        PrecisionMinerKind.find_data_sources,
        "--miner-kind",
        help="Find data sources is the lowest-friction default extraction workflow.",
    ),
    strategy_name: str = typer.Option(_DEFAULT_STRATEGY_NAME, "--strategy-name"),
    loader: LoaderKind = typer.Option(LoaderKind.unstructured, "--loader"),
    embed_model: str = typer.Option(_DEFAULT_EMBED_MODEL, "--embed-model"),
    chunker: ChunkerKind = typer.Option(ChunkerKind.paragraph, "--chunker"),
    min_chunk_size: int = typer.Option(_DEFAULT_MIN_CHUNK_SIZE, "--min-chunk-size"),
    chunk_size: int = typer.Option(600, "--chunk-size"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap"),
    index_backend: IndexBackend = typer.Option(IndexBackend.file, "--index-backend"),
    index_dir: Path = typer.Option(_DEFAULT_INDEX_DIR, "--index-dir"),
    qdrant_url: str = typer.Option(_SETTINGS.qdrant_url, "--qdrant-url"),
    qdrant_collection: str = typer.Option(
        _SETTINGS.qdrant_collection, "--qdrant-collection"
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
        retrieval_mode = _workspace_enum(
            workspace_config,
            RetrievalMode,
            retrieval_mode,
            RetrievalMode.dense_only,
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
            embed_model=embed_model,
            chunker_kind=chunker,
            min_chunk_size=min_chunk_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        tempdir = resolved["tempdir"]
        generator = _build_generator(llm_provider, llm_model, temperature)
        miner = PrecisionMiner(
            retriever=resolved["retriever"],
            generator=generator,
            strategy_name=strategy_name,
            academic_db=resolved["academic_db"],
            config=_build_precision_miner_config(miner_kind, workflow_top_k),
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
        _abort(str(exc))
    finally:
        if tempdir is not None:
            tempdir.cleanup()


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


if __name__ == "__main__":
    main()
