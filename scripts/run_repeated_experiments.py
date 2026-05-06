from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import dotenv
dotenv.load_dotenv()


@dataclass(frozen=True)
class Settings:
    strategy_name: str = "grobid"
    paper_source: str = "csv_subset" # one of ["all_db", "csv_subset"]
    subset_papers_csv_path: Optional[str] = "sampled_papers_full.csv" # None
    subset_papers_csv_sep: str = "\t"
    ground_truth_csv_path: Optional[str] = "sampled_papers_full.csv"
    ground_truth_csv_sep: str = "\t"
    mongo_uri_or_env: str = os.environ.get("MONGO_URI", "")  #"MONGO_URI"
    mongo_db_name: str = "episcope_academic_db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "episcope_academic"
    llm_provider: str = "openrouter"  # one of ["gemini", "openrouter", "openai", "ollama"]
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"  # "minimax/minimax-m2.5:free"  #  "deepseek/deepseek-v3.2"  # "gemini-2.5-flash"
    llm_temperature: float = 0.0
    classifier_kind: str = "data_accessibility" # one of ["paper_type", "data_accessibility", "data_type", "geo"]
    workflow_top_k: int = 20
    retrieval_mode: str = "dense_only" # one of ["dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"]
    # hybris uses both dense and sparse retrievers and does the reranking
    # while hybrid_candidates_only uses both retrievers and RRF
    #
    evidence_reranker_kind: str = "none" # one of ["none", "global_cross_encoder", "within_label_cross_encoder"]
    cross_encoder_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2" # e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_k: Optional[int] = 15
    base_output_dir: str = "outputs/nemotron_dense"#"output"
    explicit_run_dir: Optional[str] = None
    checkpoint_every: int = 1
    fail_fast: bool = False
    record_failures: bool = True
    allow_delete_on_errors: bool = False


# =========================
# FIXED Tee Class
# =========================
class Tee:
    """
    Stream splitter - FIXED to handle closed files gracefully.

    The bug: when file handles close (e.g., at end of with block),
    logging systems may still try to write to them, causing
    "ValueError: I/O operation on closed file"

    The fix: check if streams are closed before writing.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                # Skip if stream is closed
                if hasattr(s, 'closed') and s.closed:
                    continue
                s.write(data)
                s.flush()
            except (ValueError, AttributeError, OSError):
                # Silently skip streams that error
                # (they're probably closed or broken)
                continue

    def flush(self):
        for s in self.streams:
            try:
                if hasattr(s, 'closed') and not s.closed:
                    s.flush()
            except (ValueError, AttributeError, OSError):
                continue


def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "run"


def stable_hash_dict(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(df: pd.DataFrame, path: Path, sep: str = "\t") -> None:
    """Write TSV atomically with verification"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, sep=sep, index=False)

        if not tmp.exists():
            raise IOError(f"Temp TSV not created: {tmp}")

        tmp_size = tmp.stat().st_size
        if tmp_size == 0:
            raise IOError(f"Temp TSV is empty: {tmp}")

        tmp.replace(path)

        if not path.exists():
            raise IOError(f"Final TSV missing after replace: {path}")

        final_size = path.stat().st_size
        msg = f"[TSV_OK] {path.name}: {len(df)} rows, {final_size:,} bytes"
        print(msg)
        print(msg, file=sys.stderr)

    except Exception as e:
        err = f"[TSV_FAIL] {path}: {e}"
        print(err)
        print(err, file=sys.stderr)
        raise


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    tmp.replace(path)


def append_log(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def resolve_mongo_uri(mongo_uri_or_env: str) -> str:
    s = (mongo_uri_or_env or "").strip()
    if s.lower().startswith("mongodb"):
        return s
    uri = os.environ.get(s)
    if not uri:
        raise RuntimeError(
            f"Mongo URI not found. Settings.mongo_uri_or_env='{s}' looks like an env var name, "
            f"but os.environ['{s}'] is not set."
        )
    return uri


def build_run_dir(settings: Settings) -> Path:
    if settings.explicit_run_dir:
        return Path(settings.explicit_run_dir)

    identity = {
        "classifier_kind": settings.classifier_kind,
        "workflow_top_k": settings.workflow_top_k,
        "retrieval_mode": settings.retrieval_mode,
        "evidence_reranker_kind": settings.evidence_reranker_kind,
        "cross_encoder_model": settings.cross_encoder_model,
        "cross_encoder_top_k": settings.cross_encoder_top_k,
        "strategy_name": settings.strategy_name,
        "qdrant_collection": settings.qdrant_collection,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "paper_source": settings.paper_source,
        "subset_papers_csv_path": settings.subset_papers_csv_path,
        "subset_papers_csv_sep": settings.subset_papers_csv_sep,
        "ground_truth_csv_path": settings.ground_truth_csv_path,
        "ground_truth_csv_sep": settings.ground_truth_csv_sep,
    }
    h = stable_hash_dict(identity)

    run_dir = (
        Path(settings.base_output_dir)
        / slugify(settings.classifier_kind)
        / slugify(settings.llm_model)
        / f"temperature_{settings.llm_temperature}"
        / slugify(settings.strategy_name)
        / f"{slugify(settings.qdrant_collection)}-{h}"
    )
    return run_dir


def paths_for_repeat(run_dir: Path, repeat_idx: int) -> Dict[str, Path]:
    return {
        "run_dir": run_dir,
        "checkpoint_tsv": run_dir / f"checkpoint_{repeat_idx}.tsv",
        "checkpoint_meta": run_dir / f"checkpoint_meta_{repeat_idx}.json",
        "checkpoint_provenance_jsonl": run_dir / f"checkpoint_provenance_{repeat_idx}.jsonl",
        "checkpoint_token_summary": run_dir / f"checkpoint_token_summary_{repeat_idx}.json",
        "final_tsv": run_dir / f"final_{repeat_idx}.tsv",
        "final_provenance_jsonl": run_dir / f"provenance_{repeat_idx}.jsonl",
        "final_token_summary": run_dir / f"token_summary_{repeat_idx}.json",
        "discrepancy_csv": run_dir / f"discrepancy_{repeat_idx}.csv",
        "log_txt": run_dir / f"run_{repeat_idx}.log",
    }


def describe_paper_source(settings: Settings) -> str:
    if settings.paper_source == "csv_subset":
        return f"csv_subset:{settings.subset_papers_csv_path}"
    return f"all_db:{settings.strategy_name}"


def load_paper_ids(settings: Settings) -> List[str]:
    if settings.paper_source == "csv_subset":
        if not settings.subset_papers_csv_path:
            raise ValueError("subset_papers_csv_path must be provided when paper_source='csv_subset'.")
        df = pd.read_csv(settings.subset_papers_csv_path, sep=settings.subset_papers_csv_sep)
        if "paper_id" not in df.columns:
            raise ValueError(f"CSV must contain a 'paper_id' column. Columns: {list(df.columns)}")
        return df["paper_id"].astype(str).tolist()

    if settings.paper_source != "all_db":
        raise ValueError(
            f"Unknown paper_source={settings.paper_source}. Use one of ['all_db', 'csv_subset']."
        )

    from episcope.db.mongo_academic_db import MongoAcademicDB
    uri = resolve_mongo_uri(settings.mongo_uri_or_env)
    db = MongoAcademicDB(uri=uri, db_name=settings.mongo_db_name)

    return [i for i in db.list_docs(strategy_name=settings.strategy_name) if "DS_Store_" not in i]
    # return [str(x) for x in db.list_docs(strategy_name=settings.strategy_name)]


def build_classifier(settings: Settings):
    from episcope.db.mongo_academic_db import MongoAcademicDB
    from episcope.vectordb.qdrant import QdrantDB
    from episcope.clients import GeminiClient, OllamaClient, OpenAIClient, OpenRouterClient
    from episcope.rag.generation.llm_generator import LLMGenerator
    from episcope.workflows import PaperClassifier
    from episcope.workflows.classification import (
        PaperTypeClassifierConfig,
        DataAccessibilityClassifierConfig,
        DataTypeClassifierConfig,
        GeoClassifierConfig,
        GlobalCrossEncoderReranker,
        WithinLabelCrossEncoderReranker,
    )
    from episcope.rag.retrieval.retriever import Retriever
    from episcope.rag.retrieval.candidates import (
        HybridCandidateRetriever,
        SemanticCandidateRetriever,
        SparseCandidateRetriever,
    )

    config_map = {
        "paper_type": PaperTypeClassifierConfig,
        "data_accessibility": DataAccessibilityClassifierConfig,
        "data_type": DataTypeClassifierConfig,
        "geo": GeoClassifierConfig,
    }
    if settings.classifier_kind not in config_map:
        raise ValueError(f"Unknown classifier_kind={settings.classifier_kind}. Use one of {list(config_map)}")

    uri = resolve_mongo_uri(settings.mongo_uri_or_env)
    db = MongoAcademicDB(uri=uri, db_name=settings.mongo_db_name)

    needs_dense = settings.retrieval_mode in {"dense_only", "hybrid", "hybrid_candidates_only"}
    needs_sparse = settings.retrieval_mode in {"sparse_only", "hybrid", "hybrid_candidates_only"}

    try:
        vdb = QdrantDB(
            collection=settings.qdrant_collection,
            url=settings.qdrant_url,
            use_dense=needs_dense,
            use_sparse=needs_sparse,
        )
    except Exception as exc:
        message = str(exc)
        if "connection refused" in message.lower():
            raise ValueError(
                f"Qdrant is not reachable at {settings.qdrant_url}. "
                "Start Qdrant or point qdrant_url at a running instance."
            ) from exc
        if "dense_dim must be provided" in message:
            raise ValueError(
                f"Qdrant collection {settings.qdrant_collection!r} was not found at {settings.qdrant_url}. "
                "This experiment runner expects an existing indexed collection for dense or hybrid retrieval."
            ) from exc
        raise
    # Notice: rerankers here always set to false because we are offloading the reranking to the classifier
    # i.e., rereanking here would rerank the chunks of the individual queries, rather than across queries
    if settings.retrieval_mode == "dense_only":
        retriever = Retriever(
            vectordb=vdb,
            candidate_retrievers=[SemanticCandidateRetriever(vdb)],
            use_rerank=False,
        )
    elif settings.retrieval_mode == "hybrid":
        retriever = Retriever(
            vectordb=vdb,
            use_rerank=False
            )
    elif settings.retrieval_mode == "sparse_only":
        retriever = Retriever(
            vectordb=vdb,
            candidate_retrievers=[SparseCandidateRetriever(vdb)],
            use_rerank=False,
        )
    elif settings.retrieval_mode == "hybrid_candidates_only":
        retriever = Retriever(
            vectordb=vdb,
            candidate_retrievers=[HybridCandidateRetriever(vdb)],
            use_rerank=False,
        )
    else:
        raise ValueError(
            f"Unknown retrieval_mode={settings.retrieval_mode}. "
            "Use one of ['dense_only', 'hybrid', 'sparse_only', 'hybrid_candidates_only']."
        )

    if settings.llm_provider == "gemini":
        client = GeminiClient()
    elif settings.llm_provider == "openrouter":
        client = OpenRouterClient()
    elif settings.llm_provider == "openai":
        client = OpenAIClient()
    elif settings.llm_provider == "ollama":
        client = OllamaClient()
    else:
        raise ValueError(
            f"Unknown llm_provider={settings.llm_provider}. "
            "Use one of ['gemini', 'openrouter', 'openai', 'ollama']."
        )

    try:
        generator = LLMGenerator(client=client, model=settings.llm_model, temperature=settings.llm_temperature)
    except TypeError:
        generator = LLMGenerator(client=client, model=settings.llm_model)

    config = config_map[settings.classifier_kind]()
    config.top_k = settings.workflow_top_k

    evidence_reranker = None
    if settings.evidence_reranker_kind != "none":
        if not settings.cross_encoder_model:
            raise ValueError("cross_encoder_model must be provided when evidence_reranker_kind is not 'none'.")

        if settings.evidence_reranker_kind == "global_cross_encoder":
            evidence_reranker = GlobalCrossEncoderReranker.from_huggingface(
                model_name=settings.cross_encoder_model,
                top_k=settings.cross_encoder_top_k,
            )
        elif settings.evidence_reranker_kind == "within_label_cross_encoder":
            evidence_reranker = WithinLabelCrossEncoderReranker.from_huggingface(
                model_name=settings.cross_encoder_model,
                top_k=settings.cross_encoder_top_k,
            )
        else:
            raise ValueError(
                f"Unknown evidence_reranker_kind={settings.evidence_reranker_kind}. "
                "Use one of ['none', 'global_cross_encoder', 'within_label_cross_encoder']."
            )

    classifier = PaperClassifier(
        retriever=retriever,
        generator=generator,
        academic_db=db,
        strategy_name=settings.strategy_name,
        config=config,
        evidence_reranker=evidence_reranker,
    )
    return classifier


def result_row_from_output(paper_id: str, c_res, classifier_kind: str) -> Dict[str, Any]:
    detailed = c_res if hasattr(c_res, "decision") else None
    decision = detailed.decision if detailed is not None else c_res
    if hasattr(decision, "result"):
        c_res = decision.result
    else:
        c_res = decision

    evidence = None
    if hasattr(c_res, "evidence"):
        if isinstance(c_res.evidence, dict):
            evidence = c_res.evidence.get("reasoning") or c_res.evidence
        else:
            evidence = c_res.evidence

    base = {
        "paper_id": str(paper_id),
        "classification": [c.name for c in getattr(c_res, "classification", [])],
        "class_probabilities": getattr(c_res, "class_probabilities", None),
        "confidence": getattr(c_res, "confidence", None),
        "evidence": evidence,
    }

    if detailed is not None:
        base.update(
            {
                "provenance_answer": getattr(detailed.provenance, "answer", None),
                "provenance_evidence_count": len(getattr(detailed.provenance, "evidences", []) or []),
                "trace_raw_llm_response": getattr(detailed.trace, "raw_llm_response", None),
                "trace_prompt_message_count": len(getattr(detailed.trace, "prompt_messages", []) or []),
                "top_evidence_count": len(getattr(detailed.decision, "top_evidence", []) or []),
                "training_reward": getattr(detailed.training, "reward", None),
                "training_sample_count": len(getattr(detailed.training, "all_samples", []) or []),
                "top_evidence_json": json.dumps(
                    [_search_result_to_dict(chunk) for chunk in getattr(detailed.decision, "top_evidence", [])],
                    ensure_ascii=False,
                ),
                "provenance_evidences_json": json.dumps(
                    [_evidence_to_dict(item) for item in getattr(detailed.provenance, "evidences", [])],
                    ensure_ascii=False,
                ),
                "trace_prompt_messages_json": json.dumps(
                    getattr(detailed.trace, "prompt_messages", []) or [],
                    ensure_ascii=False,
                ),
                "training_samples_json": json.dumps(
                    [_completion_sample_to_dict(sample) for sample in getattr(detailed.training, "all_samples", [])],
                    ensure_ascii=False,
                ),
            }
        )

    extras = getattr(c_res, "extras", {}) or {}

    if classifier_kind == "paper_type":
        sec = extras.get("secondary_labels", [])
        base["secondary_labels"] = [c.name for c in sec] if isinstance(sec, list) else []
        base["extras"] = extras
    elif classifier_kind == "geo":
        base["countries"] = extras.get("countries", [])
        base["cities"] = extras.get("cities", [])
        base["extras"] = extras
    else:
        base["extras"] = extras

    return base


def compute_run_signature(settings: Settings) -> Dict[str, Any]:
    return {
        "llm_provider": settings.llm_provider,
        "classifier_kind": settings.classifier_kind,
        "workflow_top_k": settings.workflow_top_k,
        "retrieval_mode": settings.retrieval_mode,
        "evidence_reranker_kind": settings.evidence_reranker_kind,
        "cross_encoder_model": settings.cross_encoder_model,
        "cross_encoder_top_k": settings.cross_encoder_top_k,
        "strategy_name": settings.strategy_name,
        "mongo_db_name": settings.mongo_db_name,
        "qdrant_url": settings.qdrant_url,
        "qdrant_collection": settings.qdrant_collection,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature,
        "paper_source": settings.paper_source,
        "subset_papers_csv_path": settings.subset_papers_csv_path,
        "subset_papers_csv_sep": settings.subset_papers_csv_sep,
        "ground_truth_csv_path": settings.ground_truth_csv_path,
        "ground_truth_csv_sep": settings.ground_truth_csv_sep,
    }


def load_checkpoint_if_any(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, sep="\t")
    return pd.DataFrame()


def load_meta_if_any(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_jsonl_if_any(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _enum_name(value: Any) -> Any:
    return getattr(value, "name", value)


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return value


def _search_result_to_dict(chunk: Any) -> Dict[str, Any]:
    return {
        "id": getattr(chunk, "id", None),
        "paper_id": getattr(chunk, "paper_id", None),
        "text": getattr(chunk, "text", None),
        "section_type": getattr(chunk, "section_type", None),
        "title": getattr(chunk, "title", None),
        "similarity_score": getattr(chunk, "similarity_score", None),
        "rank_score": getattr(chunk, "rank_score", None),
        "source": getattr(chunk, "source", None),
        "artifacts": _json_ready(getattr(chunk, "artifacts", {}) or {}),
    }


def _evidence_to_dict(item: Any) -> Dict[str, Any]:
    return {
        "paper_id": getattr(item, "paper_id", None),
        "snippet": getattr(item, "snippet", None),
        "section": getattr(item, "section", None),
        "index_version": getattr(item, "index_version", None),
        "model_id": getattr(item, "model_id", None),
        "prompt_id": getattr(item, "prompt_id", None),
    }


def _completion_sample_to_dict(sample: Any) -> Dict[str, Any]:
    return {
        "messages": _json_ready(getattr(sample, "messages", []) or []),
        "completion": getattr(sample, "completion", None),
        "parsed_ok": getattr(sample, "parsed_ok", None),
        "reward": getattr(sample, "reward", None),
    }


def _usage_delta(after: Any, before: Any) -> Dict[str, int]:
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


def _usage_snapshot_from_classifier(classifier: Any) -> Optional[Any]:
    generator = getattr(classifier, "generator", None)
    client = getattr(generator, "client", None)
    usage_snapshot = getattr(client, "usage_snapshot", None)
    if callable(usage_snapshot):
        return usage_snapshot()
    return None


def _usage_payload_for_row(settings: Settings, usage_delta: Optional[Dict[str, int]]) -> Dict[str, Any]:
    usage = usage_delta or {}
    return {
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "llm_completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "llm_total_tokens": int(usage.get("total_tokens", 0) or 0),
        "llm_cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        "llm_reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
        "llm_call_count": int(usage.get("call_count", 0) or 0),
    }


def build_token_summary(results_df: pd.DataFrame, settings: Settings, *, repeat_idx: int) -> Dict[str, Any]:
    if results_df.empty:
        return {
            "repeat_idx": repeat_idx,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "paper_count": 0,
            "papers_with_llm_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "call_count": 0,
        }

    def _sum_column(name: str) -> int:
        if name not in results_df.columns:
            return 0
        series = pd.to_numeric(results_df[name], errors="coerce").fillna(0)
        return int(series.sum())

    if "llm_total_tokens" in results_df.columns:
        papers_with_usage = int((pd.to_numeric(results_df["llm_total_tokens"], errors="coerce").fillna(0) > 0).sum())
    else:
        papers_with_usage = 0

    return {
        "repeat_idx": repeat_idx,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "paper_count": int(len(results_df)),
        "papers_with_llm_usage": papers_with_usage,
        "prompt_tokens": _sum_column("llm_prompt_tokens"),
        "completion_tokens": _sum_column("llm_completion_tokens"),
        "total_tokens": _sum_column("llm_total_tokens"),
        "cached_tokens": _sum_column("llm_cached_tokens"),
        "reasoning_tokens": _sum_column("llm_reasoning_tokens"),
        "call_count": _sum_column("llm_call_count"),
    }


def _looks_like_credit_exhaustion(error: Exception) -> bool:
    text = repr(error).lower()
    needles = (
        "insufficient credits",
        "insufficient credit",
        "insufficient balance",
        "quota exceeded",
        "out of credits",
        "payment required",
        "402",
    )
    return any(needle in text for needle in needles)


def detailed_record_from_output(
    paper_id: str,
    detailed_output: Any,
    *,
    token_usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "paper_id": str(paper_id),
        "token_usage": token_usage or {},
        "decision": _json_ready(getattr(detailed_output, "decision", None)),
        "provenance": {
            "answer": getattr(getattr(detailed_output, "provenance", None), "answer", None),
            "evidences": [
                _evidence_to_dict(item)
                for item in getattr(getattr(detailed_output, "provenance", None), "evidences", []) or []
            ],
        },
        "trace": _json_ready(getattr(detailed_output, "trace", None)),
        "training": _json_ready(getattr(detailed_output, "training", None)),
    }


def failure_record(paper_id: str, error: Exception) -> Dict[str, Any]:
    return {
        "paper_id": str(paper_id),
        "token_usage": {},
        "error": repr(error),
        "decision": None,
        "provenance": None,
        "trace": None,
        "training": None,
    }


def parse_label_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(_enum_name(item)) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return [str(_enum_name(item)) for item in parsed]
        return [text]
    return [str(_enum_name(value))]


def ground_truth_column(classifier_kind: str) -> str:
    mapping = {
        "paper_type": "ptype_classification",
        "data_accessibility": "availability_classification",
        "data_type": "data_type_classification",
        "geo": "geo_classification",
    }
    if classifier_kind not in mapping:
        raise ValueError(f"No ground-truth mapping configured for classifier_kind={classifier_kind!r}")
    return mapping[classifier_kind]


def build_discrepancy_dataframe(results_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if not settings.ground_truth_csv_path:
        raise ValueError("ground_truth_csv_path must be set to build a discrepancy report.")

    ground_truth = pd.read_csv(settings.ground_truth_csv_path, sep=settings.ground_truth_csv_sep)
    gt_col = ground_truth_column(settings.classifier_kind)
    extra_gt_cols = ["title", "journal", "year"]
    if settings.classifier_kind == "paper_type":
        extra_gt_cols.append("ptype_secondary")
    if settings.classifier_kind == "geo":
        extra_gt_cols.extend(["geo_countries", "geo_cities_regions"])

    available_extra_cols = [col for col in extra_gt_cols if col in ground_truth.columns]
    merge_cols = ["paper_id", gt_col, *available_extra_cols]
    merged = results_df.merge(ground_truth[merge_cols], on="paper_id", how="left")

    rows: List[Dict[str, Any]] = []
    for record in merged.to_dict(orient="records"):
        predicted = sorted(parse_label_list(record.get("classification")))
        expected = sorted(parse_label_list(record.get(gt_col)))
        predicted_set = set(predicted)
        expected_set = set(expected)
        missing = sorted(expected_set - predicted_set)
        extra = sorted(predicted_set - expected_set)

        if record.get(gt_col) is None or (isinstance(record.get(gt_col), float) and pd.isna(record.get(gt_col))):
            discrepancy_kind = "missing_ground_truth"
            exact_match = None
        elif not predicted:
            discrepancy_kind = "missing_prediction"
            exact_match = False
        elif not missing and not extra:
            discrepancy_kind = "exact_match"
            exact_match = True
        elif missing and extra:
            discrepancy_kind = "missing_and_extra"
            exact_match = False
        elif missing:
            discrepancy_kind = "missing_labels"
            exact_match = False
        else:
            discrepancy_kind = "extra_labels"
            exact_match = False

        row = {
            "paper_id": record.get("paper_id"),
            "predicted_classification": predicted,
            "ground_truth_classification": expected,
            "exact_match": exact_match,
            "discrepancy_kind": discrepancy_kind,
            "missing_labels": missing,
            "extra_labels": extra,
            "predicted_count": len(predicted),
            "ground_truth_count": len(expected),
            "confidence": record.get("confidence"),
            "error": ((record.get("extras") or {}).get("error") if isinstance(record.get("extras"), dict) else None),
        }
        for col in available_extra_cols:
            row[f"ground_truth_{col}"] = record.get(col)
        if settings.classifier_kind == "paper_type" and "ptype_secondary" in record:
            row["ground_truth_secondary_labels"] = parse_label_list(record.get("ptype_secondary"))
        rows.append(row)

    discrepancy_df = pd.DataFrame(rows)
    discrepancy_df["sort_exact_match"] = discrepancy_df["exact_match"].map({False: 0, True: 1}).fillna(2)
    discrepancy_df["symmetric_difference_count"] = discrepancy_df["missing_labels"].map(len) + discrepancy_df["extra_labels"].map(len)
    discrepancy_df = discrepancy_df.sort_values(
        by=["sort_exact_match", "symmetric_difference_count", "paper_id"],
        ascending=[True, False, True],
    ).drop(columns=["sort_exact_match"])
    return discrepancy_df


def safe_to_resume(existing_meta: Dict[str, Any], expected_n: int, run_sig: Dict[str, Any]) -> Tuple[bool, str]:
    if not existing_meta:
        return True, "No metadata found; resume based on checkpoint only."

    if existing_meta.get("run_signature") != run_sig:
        return False, "Run signature differs from existing checkpoint metadata; refusing auto-resume."

    old_expected = existing_meta.get("expected_n")
    if isinstance(old_expected, int) and old_expected != expected_n:
        return False, f"Expected N differs (meta={old_expected} vs current={expected_n}); refusing auto-resume."

    return True, "Checkpoint metadata matches; safe to resume."


def run_once(settings: Settings, repeat_idx: int) -> None:
    """
    FIXED VERSION - Tee class now handles closed files gracefully
    """
    run_dir = build_run_dir(settings)

    print(f"\n{'='*70}")
    print(f"RUN DIRECTORY: {run_dir.absolute()}")
    print(f"REPEAT INDEX:  {repeat_idx}")
    print(f"{'='*70}\n")

    run_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise IOError(f"CRITICAL: Failed to create run directory: {run_dir.absolute()}")

    p = paths_for_repeat(run_dir, repeat_idx)

    stdout_path = run_dir / f"stdout_{repeat_idx}.txt"
    stderr_path = run_dir / f"stderr_{repeat_idx}.txt"

    orig_out, orig_err = sys.stdout, sys.stderr

    # Open files and create Tee objects
    with stdout_path.open("a", encoding="utf-8") as out_f, \
         stderr_path.open("a", encoding="utf-8") as err_f:

        sys.stdout = Tee(orig_out, out_f)
        sys.stderr = Tee(orig_err, err_f)

        try:
            if p["final_tsv"].exists():
                print(f"[OK] final already exists: {p['final_tsv']}")
                return

            paper_ids = load_paper_ids(settings)
            expected_n = len(paper_ids)
            run_sig = compute_run_signature(settings)
            source_desc = describe_paper_source(settings)

            df_ckpt = load_checkpoint_if_any(p["checkpoint_tsv"])
            meta = load_meta_if_any(p["checkpoint_meta"])
            provenance_records = {
                str(record["paper_id"]): record
                for record in load_jsonl_if_any(p["checkpoint_provenance_jsonl"])
                if "paper_id" in record
            }

            ok_resume, reason = safe_to_resume(meta, expected_n, run_sig)
            append_log(p["log_txt"], f"Startup repeat={repeat_idx}. source={source_desc}. {reason}")
            print(f"[INFO] Startup repeat={repeat_idx}. source={source_desc}. {reason}")

            if not ok_resume:
                raise RuntimeError(
                    reason
                    + "\nTo override, delete the checkpoint/meta for this repeat or change parameters."
                )

            done_ids = (
                set(df_ckpt["paper_id"].astype(str).tolist())
                if not df_ckpt.empty and "paper_id" in df_ckpt
                else set()
            )

            meta_out = {
                "repeat_idx": repeat_idx,
                "expected_n": expected_n,
                "run_signature": run_sig,
                "created_at": meta.get("created_at") or time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            atomic_write_text(p["checkpoint_meta"], json.dumps(meta_out, indent=2, sort_keys=True))

            classifier = build_classifier(settings)

            list_rows: List[Dict[str, Any]] = []
            list_provenance: List[Dict[str, Any]] = []
            processed_since_flush = 0
            n_errors = 0
            stop_due_to_credits = False

            for idx, paper_id in enumerate(paper_ids, start=1):

                if paper_id in done_ids:
                    print(f"[{idx}/{expected_n}] skip {paper_id} (checkpoint)")
                    continue

                # if idx % 15 == 0:
                #     # rest for 10 minutes every 10 papers to avoid rate limits
                #     print(f"[INFO] Resting for 3 minutes to avoid rate limits...")
                #     time.sleep(180)
                paper_id = str(paper_id)

                print(f"[{idx}/{expected_n}] analyze {paper_id}")
                usage_before = _usage_snapshot_from_classifier(classifier)
                caught_error: Optional[Exception] = None
                try:
                    c_res = classifier.run_detailed(paper_id)
                    row = result_row_from_output(paper_id, c_res, settings.classifier_kind)
                except Exception as e:
                    caught_error = e
                    n_errors += 1
                    append_log(p["log_txt"], f"ERROR paper_id={paper_id} err={repr(e)}")
                    print(f"[ERROR] paper_id={paper_id} err={repr(e)}")

                    if _looks_like_credit_exhaustion(e):
                        append_log(
                            p["log_txt"],
                            f"Stopping repeat={repeat_idx} after provider credit exhaustion at paper_id={paper_id}.",
                        )
                        print(
                            f"[WARN] stopping after provider credit exhaustion at paper_id={paper_id}. "
                            "Checkpoint data will be preserved for resume."
                        )
                        stop_due_to_credits = True
                        break

                    if settings.fail_fast:
                        raise

                    if settings.record_failures:
                        row = {
                            "paper_id": paper_id,
                            "classification": [],
                            "class_probabilities": None,
                            "confidence": None,
                            "evidence": None,
                            "extras": {"error": repr(e)},
                        }
                    else:
                        continue
                    c_res = None

                usage_after = _usage_snapshot_from_classifier(classifier)
                usage_delta = (
                    _usage_delta(usage_after, usage_before)
                    if usage_before is not None and usage_after is not None
                    else None
                )
                row.update(_usage_payload_for_row(settings, usage_delta))
                if c_res is not None:
                    provenance_record = detailed_record_from_output(
                        paper_id,
                        c_res,
                        token_usage=usage_delta,
                    )
                else:
                    provenance_record = failure_record(paper_id, caught_error or RuntimeError("unknown error"))
                    provenance_record["token_usage"] = usage_delta or {}

                list_rows.append(row)
                list_provenance.append(provenance_record)
                processed_since_flush += 1

                if processed_since_flush >= settings.checkpoint_every:
                    df_new = pd.DataFrame(list_rows)
                    df_merged = pd.concat([df_ckpt, df_new], ignore_index=True)

                    if "paper_id" in df_merged.columns:
                        df_merged = df_merged.drop_duplicates(subset=["paper_id"], keep="first")

                    atomic_write_tsv(df_merged, p["checkpoint_tsv"])
                    for record in list_provenance:
                        provenance_records[str(record["paper_id"])] = record
                    atomic_write_jsonl(list(provenance_records.values()), p["checkpoint_provenance_jsonl"])
                    atomic_write_text(
                        p["checkpoint_token_summary"],
                        json.dumps(
                            build_token_summary(df_merged, settings, repeat_idx=repeat_idx),
                            indent=2,
                            sort_keys=True,
                        ),
                    )

                    append_log(
                        p["log_txt"],
                        f"Checkpoint flush: +{len(df_new)} rows (total={len(df_merged)}/{expected_n}), errors={n_errors}",
                    )
                    print(
                        f"[INFO] checkpoint flush: +{len(df_new)} rows "
                        f"(total={len(df_merged)}/{expected_n}) errors={n_errors}"
                    )

                    df_ckpt = df_merged
                    done_ids = set(df_ckpt["paper_id"].astype(str).tolist())
                    list_rows.clear()
                    list_provenance.clear()
                    processed_since_flush = 0

            # Final flush
            if list_rows:
                df_new = pd.DataFrame(list_rows)
                df_merged = pd.concat([df_ckpt, df_new], ignore_index=True)

                if "paper_id" in df_merged.columns:
                    df_merged = df_merged.drop_duplicates(subset=["paper_id"], keep="first")

                atomic_write_tsv(df_merged, p["checkpoint_tsv"])
                for record in list_provenance:
                    provenance_records[str(record["paper_id"])] = record
                atomic_write_jsonl(list(provenance_records.values()), p["checkpoint_provenance_jsonl"])
                atomic_write_text(
                    p["checkpoint_token_summary"],
                    json.dumps(
                        build_token_summary(df_merged, settings, repeat_idx=repeat_idx),
                        indent=2,
                        sort_keys=True,
                    ),
                )
                df_ckpt = df_merged

            n_done = len(set(df_ckpt["paper_id"].astype(str).tolist())) if not df_ckpt.empty else 0
            append_log(p["log_txt"], f"Run end: done={n_done}/{expected_n}, errors={n_errors}")
            print(f"[INFO] run end: done={n_done}/{expected_n}, errors={n_errors}")

            if stop_due_to_credits:
                print(
                    f"[WARN] Provider credits exhausted before completion: {n_done}/{expected_n}. "
                    "Resume later after restoring credits or switching provider/model."
                )
                return

            if n_done != expected_n:
                print(f"[WARN] Not complete: {n_done}/{expected_n}. Resume will continue.")
                return

            # Write final
            atomic_write_tsv(df_ckpt, p["final_tsv"])
            atomic_write_jsonl(list(provenance_records.values()), p["final_provenance_jsonl"])
            atomic_write_text(
                p["final_token_summary"],
                json.dumps(
                    build_token_summary(df_ckpt, settings, repeat_idx=repeat_idx),
                    indent=2,
                    sort_keys=True,
                ),
            )

            if settings.ground_truth_csv_path:
                discrepancy_df = build_discrepancy_dataframe(df_ckpt, settings)
                atomic_write_csv(discrepancy_df, p["discrepancy_csv"])

            # Verify
            if not p["final_tsv"].exists():
                raise IOError(f"CRITICAL: Final TSV missing: {p['final_tsv']}")

            verify_msg = f"SUCCESS: Final TSV at {p['final_tsv']} ({p['final_tsv'].stat().st_size:,} bytes, {len(df_ckpt)} rows)"
            print(verify_msg)
            append_log(p["log_txt"], verify_msg)

            # Cleanup
            if n_errors == 0 or settings.allow_delete_on_errors:
                try:
                    p["checkpoint_tsv"].unlink(missing_ok=True)
                    p["checkpoint_meta"].unlink(missing_ok=True)
                    p["checkpoint_provenance_jsonl"].unlink(missing_ok=True)
                    print("[INFO] cleanup: deleted checkpoint files")
                except Exception as e:
                    print(f"[WARN] cleanup error: {repr(e)}")

            print(f"\n{'='*70}")
            print(f"[FINAL] Repeat {repeat_idx} complete")
            print(f"  Final: {p['final_tsv'].name} ({len(df_ckpt)} rows, {n_errors} errors)")
            print(f"{'='*70}\n")

        finally:
            # Restore original streams
            sys.stdout, sys.stderr = orig_out, orig_err
            # Files will auto-close when exiting the with block


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Restartable PaperClassifier runner")
    ap.add_argument(
        "--paper-source",
        type=str,
        default=None,
        choices=["all_db", "csv_subset"],
    )
    ap.add_argument("--subset-papers-csv", type=str, default=None)
    ap.add_argument("--subset-papers-csv-sep", type=str, default=None)
    ap.add_argument("--ground-truth-csv", type=str, default=None)
    ap.add_argument("--ground-truth-csv-sep", type=str, default=None)
    ap.add_argument("--strategy-name", type=str, default=None)
    ap.add_argument("--qdrant-url", type=str, default=None)
    ap.add_argument("--qdrant-collection", type=str, default=None)
    ap.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        choices=["gemini", "openrouter", "openai", "ollama"],
    )
    ap.add_argument("--mongo-uri-or-env", type=str, default=None)
    ap.add_argument("--mongo-db-name", type=str, default=None)
    ap.add_argument("--base-output-dir", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--fail-fast", action="store_true", default=False)
    ap.add_argument("--no-record-failures", action="store_true", default=False)
    ap.add_argument("--classifier-kind", type=str, default=None)
    ap.add_argument("--llm-model", type=str, default=None)
    ap.add_argument("--llm-temperature", type=float, default=None)
    ap.add_argument("--workflow-top-k", type=int, default=None)
    ap.add_argument(
        "--retrieval-mode",
        type=str,
        default=None,
        choices=["dense_only", "hybrid", "sparse_only", "hybrid_candidates_only"],
    )
    ap.add_argument(
        "--evidence-reranker-kind",
        type=str,
        default=None,
        choices=["none", "global_cross_encoder", "within_label_cross_encoder"],
    )
    ap.add_argument("--cross-encoder-model", type=str, default=None)
    ap.add_argument("--cross-encoder-top-k", type=int, default=None)
    return ap.parse_args()


def merge_settings(settings: Settings, args: argparse.Namespace) -> Settings:
    d = dataclasses.asdict(settings)
    if isinstance(d.get("llm_model"), tuple) and len(d["llm_model"]) == 1:
        d["llm_model"] = d["llm_model"][0]
    if args.paper_source is not None:
        d["paper_source"] = args.paper_source
    if args.subset_papers_csv is not None:
        d["subset_papers_csv_path"] = args.subset_papers_csv
        d["paper_source"] = "csv_subset"
    if args.subset_papers_csv_sep is not None:
        d["subset_papers_csv_sep"] = args.subset_papers_csv_sep
    if args.ground_truth_csv is not None:
        d["ground_truth_csv_path"] = args.ground_truth_csv
    if args.ground_truth_csv_sep is not None:
        d["ground_truth_csv_sep"] = args.ground_truth_csv_sep
    if args.strategy_name is not None:
        d["strategy_name"] = args.strategy_name
    if args.qdrant_url is not None:
        d["qdrant_url"] = args.qdrant_url
    if args.qdrant_collection is not None:
        d["qdrant_collection"] = args.qdrant_collection
    if args.llm_provider is not None:
        d["llm_provider"] = args.llm_provider
    if args.mongo_uri_or_env is not None:
        d["mongo_uri_or_env"] = args.mongo_uri_or_env
    if args.mongo_db_name is not None:
        d["mongo_db_name"] = args.mongo_db_name
    if args.base_output_dir is not None:
        d["base_output_dir"] = args.base_output_dir
    if args.run_dir is not None:
        d["explicit_run_dir"] = args.run_dir
    if args.checkpoint_every is not None:
        d["checkpoint_every"] = args.checkpoint_every
    if args.fail_fast:
        d["fail_fast"] = True
    if args.no_record_failures:
        d["record_failures"] = False
    if args.classifier_kind is not None:
        d["classifier_kind"] = args.classifier_kind
    if args.llm_model is not None:
        d["llm_model"] = args.llm_model
    if args.llm_temperature is not None:
        d["llm_temperature"] = args.llm_temperature
    if args.workflow_top_k is not None:
        d["workflow_top_k"] = args.workflow_top_k
    if args.retrieval_mode is not None:
        d["retrieval_mode"] = args.retrieval_mode
    if args.evidence_reranker_kind is not None:
        d["evidence_reranker_kind"] = args.evidence_reranker_kind
    if args.cross_encoder_model is not None:
        d["cross_encoder_model"] = args.cross_encoder_model
    if args.cross_encoder_top_k is not None:
        d["cross_encoder_top_k"] = args.cross_encoder_top_k
    return Settings(**d)


def main():
    base = Settings()
    args = parse_args()
    base = merge_settings(base, args)

    if any(
        value is not None
        for value in [
            args.classifier_kind,
            args.llm_provider,
            args.llm_model,
            args.llm_temperature,
            args.paper_source,
            args.subset_papers_csv,
            args.subset_papers_csv_sep,
            args.ground_truth_csv,
            args.ground_truth_csv_sep,
            args.workflow_top_k,
            args.retrieval_mode,
            args.evidence_reranker_kind,
            args.cross_encoder_model,
            args.cross_encoder_top_k,
        ]
    ):
        print("[INFO] Single-run mode")
        run_once(base, repeat_idx=1)
        return

    # Grid
    CLASSIFIER_KINDS = [
        "paper_type",
        "data_accessibility",
        "data_type",
        "geo"
        ]
    LLM_PROVIDERS = [
        base.llm_provider,
    ]
    LLM_MODELS = [
        base.llm_model,
    ]
    TEMPERATURES = [
        0.0,
        # 1.0
        ]
    REPEATS = 1

    def apply_overrides(s: Settings, overrides: dict) -> Settings:
        d = dataclasses.asdict(s)
        d.update(overrides)
        return Settings(**d)

    had_failure = False

    for kind in CLASSIFIER_KINDS:
        for provider in LLM_PROVIDERS:
            for model in LLM_MODELS:
                for temp in TEMPERATURES:
                    combo = {
                        "classifier_kind": kind,
                        "llm_provider": provider,
                        "llm_model": model,
                        "llm_temperature": float(temp),
                    }
                    settings = apply_overrides(base, combo)
                    run_dir = build_run_dir(settings)

                    print(
                        f"\n=== PARAMS kind={kind} | provider={provider} | model={model} | "
                        f"temp={temp} | dir={run_dir} ==="
                    )

                    for rep in range(1, REPEATS + 1):
                        try:
                            print(f"\n--- repeat {rep}/{REPEATS} ---")
                            run_once(settings, repeat_idx=rep)
                        except Exception as e:
                            had_failure = True
                            print(
                                f"[FATAL] kind={kind} provider={provider} model={model} "
                                f"temp={temp} rep={rep} err={repr(e)}",
                                file=sys.stderr,
                            )
                            if settings.fail_fast:
                                raise

    if had_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
