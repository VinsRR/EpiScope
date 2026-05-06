"""retry_unclear.py

Scans TSV output files produced by run_repeated_experiments.py and re-runs the
classifier for rows whose classification is empty or is the classifier's
current fallback/unclear label.

The patched TSV is written back atomically so that no data is lost if the
script is interrupted mid-way.

Usage examples
--------------
# Retry everything under the default output/ tree:
python retry_unclear.py

# Retry a specific run directory (all repeats inside it):
python retry_unclear.py --run-dir output/data-type/gemini-2-5-pro/temperature_0.0/grobid/episcope-abc123

# Override model / temperature for the retry:
python retry_unclear.py --llm-model gemini-2.5-flash --llm-temperature 0.0

# Dry-run: show what would be retried without changing anything:
python retry_unclear.py --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import dotenv

dotenv.load_dotenv()

import run_repeated_experiments as runner

# ---------------------------------------------------------------------------
# Re-use helpers from run_repeated_experiments (copy-pasted to keep the
# retry script self-contained; adjust the import path if you prefer).
# ---------------------------------------------------------------------------

# -- inline copies of the shared helpers -----------------------------------

import hashlib
import os


def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "run"


def stable_hash_dict(d: Dict[str, Any]) -> str:
    blob = json.dumps(
        d, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def atomic_write_tsv(df: pd.DataFrame, path: Path, sep: str = "\t") -> None:
    """Write a DataFrame as TSV atomically (temp-file + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep=sep, index=False)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise IOError(f"Temp TSV not created or empty: {tmp}")
    tmp.replace(path)
    print(f"[TSV_OK] {path.name}: {len(df)} rows, {path.stat().st_size:,} bytes")


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
            f"Mongo URI not found. '{s}' looks like an env-var name but is not set."
        )
    return uri


# ---------------------------------------------------------------------------
# Settings (mirrors the original)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    strategy_name: str = "grobid"
    paper_source: str = "csv_subset"
    subset_papers_csv_path: Optional[str] = "sampled_papers_full.csv"
    subset_papers_csv_sep: str = "\t"
    ground_truth_csv_path: Optional[str] = "sampled_papers_full.csv"
    ground_truth_csv_sep: str = "\t"
    mongo_uri_or_env: str = "MONGO_URI"
    mongo_db_name: str = "episcope_academic_db"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "episcope_academic"
    llm_provider: str = "openrouter"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    llm_temperature: float = 0.0
    classifier_kind: str = "data_accessibility"
    workflow_top_k: int = 20
    retrieval_mode: str = "dense_only"
    evidence_reranker_kind: str = "none"
    cross_encoder_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_k: Optional[int] = 15
    base_output_dir: str = "outputs/nemotron_dense"
    explicit_run_dir: Optional[str] = None
    checkpoint_every: int = 1
    fail_fast: bool = False
    record_failures: bool = True
    allow_delete_on_errors: bool = False


# ---------------------------------------------------------------------------
# Classifier construction (identical to original)
# ---------------------------------------------------------------------------

def build_classifier(settings: Settings):
    return runner.build_classifier(settings)


def result_row_from_output(
    paper_id: str, c_res: Any, classifier_kind: str
) -> Dict[str, Any]:
    return runner.result_row_from_output(paper_id, c_res, classifier_kind)


# ---------------------------------------------------------------------------
# Helpers: detect UNCLEAR / failed rows
# ---------------------------------------------------------------------------

def _parse_classification(raw: Any) -> List[str]:
    """Parse the 'classification' cell regardless of its serialised form."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return []
    raw = raw.strip()
    if not raw or raw in ("[]", "nan", "None"):
        return []
    # Stored as Python-repr list: "['UNCLEAR']" or "['A', 'B']"
    if raw.startswith("["):
        try:
            import ast
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [raw]


UNCLEAR_TOKENS_BY_KIND = {
    "paper_type": {"UNCLEAR", "unclear"},
    "data_accessibility": {"UNCLEAR", "unclear"},
    "data_type": {"UNCLEAR", "unclear", "H"},
    "geo": {"UNCLEAR", "unclear", "H"},
}


def is_unclear_row(row: pd.Series, classifier_kind: str = "data_accessibility") -> bool:
    """Return True if this row should be retried."""
    labels = _parse_classification(row.get("classification"))
    if not labels:
        return True
    unclear_tokens = UNCLEAR_TOKENS_BY_KIND.get(
        classifier_kind, {"UNCLEAR", "unclear"}
    )
    return all(str(lbl).strip() in unclear_tokens for lbl in labels)


# ---------------------------------------------------------------------------
# Infer classifier_kind from the run directory path
# ---------------------------------------------------------------------------

KNOWN_KINDS = ["paper_type", "data_accessibility", "data_type", "geo"]


def infer_classifier_kind(run_dir: Path) -> Optional[str]:
    """
    The run directory hierarchy is:
      <base>/<classifier_kind_slug>/<model_slug>/temperature_X/<strategy>/<collection-hash>
    Walk up the parts and try to match a known kind (after un-slugifying).
    """
    for part in run_dir.parts:
        # slugified versions: paper-type, data-accessibility, data-type, geo
        candidate = part.replace("-", "_")
        if candidate in KNOWN_KINDS:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Discover TSV files to process
# ---------------------------------------------------------------------------

def find_tsv_files(base_dir: Path) -> List[Path]:
    """
    Return all checkpoint_*.tsv and final_*.tsv files under base_dir,
    preferring final_ over checkpoint_ when both exist for the same repeat.
    """
    all_tsvs: List[Path] = sorted(base_dir.rglob("*.tsv"))
    # Filter to only files matching our naming convention
    matched = [
        p for p in all_tsvs
        if re.match(r"(checkpoint|final)_\d+\.tsv$", p.name)
    ]
    # For each run_dir + repeat_idx, prefer final over checkpoint
    seen: Dict[Tuple[Path, str], Path] = {}
    for p in matched:
        m = re.match(r"(checkpoint|final)_(\d+)\.tsv$", p.name)
        if not m:
            continue
        kind, idx = m.group(1), m.group(2)
        key = (p.parent, idx)
        existing = seen.get(key)
        # final > checkpoint
        if existing is None or kind == "final":
            seen[key] = p
    return sorted(seen.values())


# ---------------------------------------------------------------------------
# Core retry logic for a single TSV file
# ---------------------------------------------------------------------------

def retry_tsv(
    tsv_path: Path,
    settings: Settings,
    *,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Retry UNCLEAR rows in *tsv_path*.

    Returns (n_retried, n_still_unclear).
    """
    df = pd.read_csv(tsv_path, sep="\t")

    if "paper_id" not in df.columns or "classification" not in df.columns:
        print(f"  [SKIP] Missing required columns in {tsv_path}")
        return 0, 0

    unclear_mask = df.apply(
        lambda row: is_unclear_row(row, settings.classifier_kind), axis=1
    )
    unclear_ids: List[str] = df.loc[unclear_mask, "paper_id"].astype(str).tolist()

    if not unclear_ids:
        print(f"  [OK]   No UNCLEAR rows in {tsv_path.name}")
        return 0, 0

    print(f"  [RETRY] {len(unclear_ids)} UNCLEAR row(s) in {tsv_path.name}: {unclear_ids}")

    if dry_run:
        print("  [DRY-RUN] Skipping actual classification.")
        return len(unclear_ids), len(unclear_ids)

    # Determine the log path for this repeat
    m = re.match(r"(?:checkpoint|final)_(\d+)\.tsv$", tsv_path.name)
    repeat_idx = int(m.group(1)) if m else 0
    log_path = tsv_path.parent / f"retry_unclear_{repeat_idx}.log"

    classifier = build_classifier(settings)

    n_retried = 0
    n_still_unclear = 0
    p = runner.paths_for_repeat(tsv_path.parent, repeat_idx)
    if tsv_path.name.startswith("final_"):
        provenance_path = p["final_provenance_jsonl"]
        token_summary_path = p["final_token_summary"]
    else:
        provenance_path = p["checkpoint_provenance_jsonl"]
        token_summary_path = p["checkpoint_token_summary"]

    provenance_records = {
        str(record["paper_id"]): record
        for record in runner.load_jsonl_if_any(provenance_path)
        if "paper_id" in record
    }

    pending_rows: List[Dict[str, Any]] = []
    pending_provenance: List[Dict[str, Any]] = []

    for i, paper_id in enumerate(unclear_ids, start=1):
        print(f"    [{i}/{len(unclear_ids)}] retrying paper_id={paper_id} …")
        usage_before = runner._usage_snapshot_from_classifier(classifier)
        caught_error: Optional[Exception] = None
        try:
            c_res = classifier.run_detailed(paper_id)
            new_row = result_row_from_output(paper_id, c_res, settings.classifier_kind)
            append_log(log_path, f"OK paper_id={paper_id} classification={new_row['classification']}")
        except Exception as exc:
            caught_error = exc
            append_log(log_path, f"ERROR paper_id={paper_id} err={repr(exc)}")
            print(f"    [ERROR] paper_id={paper_id}: {repr(exc)}")
            if settings.fail_fast:
                raise
            new_row = {
                "paper_id": paper_id,
                "classification": [],
                "class_probabilities": None,
                "confidence": None,
                "evidence": None,
                "extras": {"error": repr(exc)},
            }
            c_res = None
            n_still_unclear += 1

        usage_after = runner._usage_snapshot_from_classifier(classifier)
        usage_delta = (
            runner._usage_delta(usage_after, usage_before)
            if usage_before is not None and usage_after is not None
            else None
        )
        new_row.update(runner._usage_payload_for_row(settings, usage_delta))

        if c_res is not None:
            provenance_record = runner.detailed_record_from_output(
                paper_id,
                c_res,
                token_usage=usage_delta,
            )
            if is_unclear_row(pd.Series(new_row), settings.classifier_kind):
                n_still_unclear += 1
        else:
            provenance_record = runner.failure_record(
                paper_id, caught_error or RuntimeError("unknown error")
            )
            provenance_record["token_usage"] = usage_delta or {}

        pending_rows.append(new_row)
        pending_provenance.append(provenance_record)
        n_retried += 1

        # Periodic checkpoint write
        if len(pending_rows) % settings.checkpoint_every == 0:
            df = _apply_updates(df, pending_rows)
            atomic_write_tsv(df, tsv_path)
            for record in pending_provenance:
                provenance_records[str(record["paper_id"])] = record
            runner.atomic_write_jsonl(list(provenance_records.values()), provenance_path)
            runner.atomic_write_text(
                token_summary_path,
                json.dumps(
                    runner.build_token_summary(df, settings, repeat_idx=repeat_idx),
                    indent=2,
                    sort_keys=True,
                ),
            )
            pending_rows.clear()
            pending_provenance.clear()

    # Final flush
    if pending_rows:
        df = _apply_updates(df, pending_rows)
        atomic_write_tsv(df, tsv_path)
        for record in pending_provenance:
            provenance_records[str(record["paper_id"])] = record
        runner.atomic_write_jsonl(list(provenance_records.values()), provenance_path)
        runner.atomic_write_text(
            token_summary_path,
            json.dumps(
                runner.build_token_summary(df, settings, repeat_idx=repeat_idx),
                indent=2,
                sort_keys=True,
            ),
        )

    return n_retried, n_still_unclear


def _apply_updates(df: pd.DataFrame, updated_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Overwrite rows in *df* for each paper_id in *updated_rows*.

    All existing columns for the target row are reset to None first so that
    stale values (e.g. a previous error in 'extras') never bleed through.
    New columns introduced by the fresh result are then added to the frame.
    """
    df = df.copy()
    df = df.set_index("paper_id")
    for row in updated_rows:
        pid = str(row["paper_id"])
        # 1. Blank out every existing column for this row
        for col in df.columns:
            df.at[pid, col] = None
        # 2. Write all values from the new result
        for col, val in row.items():
            if col == "paper_id":
                continue
            if col not in df.columns:
                df[col] = None
            df.at[pid, col] = val
    return df.reset_index()


# ---------------------------------------------------------------------------
# Settings inference from run directory path + meta files
# ---------------------------------------------------------------------------

def _parse_temperature_from_path(run_dir: Path) -> Optional[float]:
    """
    The directory hierarchy produced by build_run_dir is:
      <base>/<classifier_kind>/<llm_model>/temperature_<T>/<strategy>/<collection-hash>
    Extract <T> from the 'temperature_X' part.
    """
    for part in run_dir.parts:
        m = re.match(r"^temperature_([0-9]+(?:\.[0-9]+)?)$", part)
        if m:
            return float(m.group(1))
    return None


def _load_run_signature_from_meta(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Read any checkpoint_meta_*.json file present in run_dir and return its
    'run_signature' dict, which contains the original (non-slugified) values
    for llm_model, llm_temperature, classifier_kind, strategy_name, etc.
    """
    for meta_path in sorted(run_dir.glob("checkpoint_meta_*.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            sig = data.get("run_signature")
            if isinstance(sig, dict):
                return sig
        except Exception:
            continue
    return None


def settings_from_run_dir(run_dir: Path, base_settings: Settings) -> Settings:
    """
    Infer classifier_kind, llm_model, and llm_temperature from the run
    directory, overriding only the values that can be reliably recovered.

    Priority:
      1. checkpoint_meta_*.json → run_signature  (exact original values)
      2. Path parsing            (temperature is exact; model slug is lossy
                                  so it is only used as a last resort)
    """
    d = dataclasses.asdict(base_settings)
    d["explicit_run_dir"] = str(run_dir)

    # --- 1. Best source: run_signature stored in the meta JSON --------------
    sig = _load_run_signature_from_meta(run_dir)
    if sig:
        if "classifier_kind" in sig:
            d["classifier_kind"] = sig["classifier_kind"]
        if "llm_provider" in sig:
            d["llm_provider"] = sig["llm_provider"]
        if "llm_model" in sig:
            d["llm_model"] = sig["llm_model"]
        if "llm_temperature" in sig:
            d["llm_temperature"] = float(sig["llm_temperature"])
        if "workflow_top_k" in sig:
            d["workflow_top_k"] = int(sig["workflow_top_k"])
        if "retrieval_mode" in sig:
            d["retrieval_mode"] = sig["retrieval_mode"]
        if "evidence_reranker_kind" in sig:
            d["evidence_reranker_kind"] = sig["evidence_reranker_kind"]
        if "cross_encoder_model" in sig:
            d["cross_encoder_model"] = sig["cross_encoder_model"]
        if "cross_encoder_top_k" in sig:
            d["cross_encoder_top_k"] = sig["cross_encoder_top_k"]
        if "strategy_name" in sig:
            d["strategy_name"] = sig["strategy_name"]
        if "qdrant_url" in sig:
            d["qdrant_url"] = sig["qdrant_url"]
        if "qdrant_collection" in sig:
            d["qdrant_collection"] = sig["qdrant_collection"]
        if "mongo_db_name" in sig:
            d["mongo_db_name"] = sig["mongo_db_name"]
        if "paper_source" in sig:
            d["paper_source"] = sig["paper_source"]
        if "subset_papers_csv_path" in sig:
            d["subset_papers_csv_path"] = sig["subset_papers_csv_path"]
        if "subset_papers_csv_sep" in sig:
            d["subset_papers_csv_sep"] = sig["subset_papers_csv_sep"]
        if "ground_truth_csv_path" in sig:
            d["ground_truth_csv_path"] = sig["ground_truth_csv_path"]
        if "ground_truth_csv_sep" in sig:
            d["ground_truth_csv_sep"] = sig["ground_truth_csv_sep"]
        return Settings(**d)

    # --- 2. Fallback: parse what we can from the path -----------------------
    kind = infer_classifier_kind(run_dir)
    if kind:
        d["classifier_kind"] = kind

    temp = _parse_temperature_from_path(run_dir)
    if temp is not None:
        d["llm_temperature"] = temp

    # Model slug is lossy (dots → dashes via slugify) so we only use it when
    # no meta file was found and the caller did not supply --llm-model.
    # We leave d["llm_model"] as the base_settings value in that case so the
    # user gets a clear warning rather than a silently wrong model name.

    return Settings(**d)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Re-classify UNCLEAR rows in existing experiment TSV outputs."
    )
    ap.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Root directory to scan for TSV files (default: Settings.base_output_dir).",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Scan only this specific run directory (all repeats inside it).",
    )
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
    ap.add_argument("--classifier-kind", type=str, default=None,
                    help="Override classifier kind (otherwise inferred from path).")
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--fail-fast", action="store_true", default=False)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only report which rows would be retried; do not call the classifier.",
    )
    return ap.parse_args()


def merge_settings(base: Settings, args: argparse.Namespace) -> Settings:
    d = dataclasses.asdict(base)
    if args.strategy_name:       d["strategy_name"]       = args.strategy_name
    if args.qdrant_url:          d["qdrant_url"]           = args.qdrant_url
    if args.qdrant_collection:   d["qdrant_collection"]    = args.qdrant_collection
    if args.llm_provider:        d["llm_provider"]         = args.llm_provider
    if args.mongo_uri_or_env:    d["mongo_uri_or_env"]     = args.mongo_uri_or_env
    if args.mongo_db_name:       d["mongo_db_name"]        = args.mongo_db_name
    if args.llm_model:           d["llm_model"]            = args.llm_model
    if args.llm_temperature is not None:
                                 d["llm_temperature"]      = args.llm_temperature
    if args.workflow_top_k is not None:
                                 d["workflow_top_k"]       = args.workflow_top_k
    if args.retrieval_mode:      d["retrieval_mode"]       = args.retrieval_mode
    if args.evidence_reranker_kind:
                                 d["evidence_reranker_kind"] = args.evidence_reranker_kind
    if args.cross_encoder_model is not None:
                                 d["cross_encoder_model"]  = args.cross_encoder_model
    if args.cross_encoder_top_k is not None:
                                 d["cross_encoder_top_k"]  = args.cross_encoder_top_k
    if args.classifier_kind:     d["classifier_kind"]      = args.classifier_kind
    if args.checkpoint_every:    d["checkpoint_every"]     = args.checkpoint_every
    if args.fail_fast:           d["fail_fast"]            = True
    return Settings(**d)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    base_settings = merge_settings(Settings(), args)

    # Determine the scan root
    if args.run_dir:
        scan_root = Path(args.run_dir)
    elif args.base_output_dir:
        scan_root = Path(args.base_output_dir)
    else:
        scan_root = Path(base_settings.base_output_dir)

    if not scan_root.exists():
        print(f"[ERROR] Scan root does not exist: {scan_root}")
        sys.exit(1)

    tsv_files = find_tsv_files(scan_root)

    if not tsv_files:
        print(f"[INFO] No checkpoint_*.tsv / final_*.tsv files found under {scan_root}")
        return

    print(f"[INFO] Found {len(tsv_files)} TSV file(s) to inspect under {scan_root}\n")

    total_retried = 0
    total_still_unclear = 0
    had_failure = False

    # Group TSVs by their run directory so we only build the classifier once
    # per unique (run_dir, classifier_kind) combination.
    from itertools import groupby

    tsv_by_run: Dict[Path, List[Path]] = {}
    for tsv in tsv_files:
        tsv_by_run.setdefault(tsv.parent, []).append(tsv)

    for run_dir, run_tsvs in tsv_by_run.items():
        # Infer / override settings for this run directory
        run_settings = settings_from_run_dir(run_dir, base_settings)
        # If the user explicitly passed --classifier-kind, honour it always
        if args.classifier_kind:
            run_settings = dataclasses.replace(run_settings, classifier_kind=args.classifier_kind)

        # Warn when we had to fall back to path-based inference (lossy for
        # model name) rather than reading from the checkpoint meta JSON.
        meta_found = bool(_load_run_signature_from_meta(run_dir))
        model_source = "meta" if meta_found else "fallback/override"

        print(f"\n{'='*70}")
        print(f"Run dir  : {run_dir}")
        print(f"Kind     : {run_settings.classifier_kind}")
        print(f"Model    : {run_settings.llm_model}  T={run_settings.llm_temperature}  (source: {model_source})")
        if not meta_found:
            print(
                f"  [WARN] No checkpoint_meta_*.json found — model name inferred from "
                f"Settings default or --llm-model override. Pass --llm-model if incorrect."
            )
        print(f"Files    : {[p.name for p in run_tsvs]}")
        print(f"{'='*70}")

        for tsv_path in run_tsvs:
            print(f"\n-> {tsv_path.name}")
            try:
                n_retried, n_still_unclear = retry_tsv(
                    tsv_path,
                    run_settings,
                    dry_run=args.dry_run,
                )
                total_retried += n_retried
                total_still_unclear += n_still_unclear
            except Exception as exc:
                had_failure = True
                print(f"  [FATAL] {tsv_path}: {repr(exc)}")
                if base_settings.fail_fast:
                    raise

    print(f"\n{'='*70}")
    print(f"[SUMMARY] TSVs inspected  : {len(tsv_files)}")
    print(f"[SUMMARY] Rows retried    : {total_retried}")
    print(f"[SUMMARY] Still UNCLEAR   : {total_still_unclear}")
    if args.dry_run:
        print("[SUMMARY] DRY-RUN — no files were modified.")
    print(f"{'='*70}")

    if had_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
