# -*- coding: utf-8 -*-
"""
Core pipeline for review expansion.

This script defines the main logic for processing a review paper, extracting
references to other studies, matching them against the review's bibliography,
and storing the results in a structured database.
"""
from __future__ import annotations

import json
import hashlib
import datetime as dt
from dataclasses import is_dataclass, asdict
import logging

from sqlalchemy import create_engine, MetaData
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.schema import Table

# Note: These imports assume that the necessary libraries (episcope, tableref)
# are installed in the environment.
from episcope.db.in_memory_academic_db import InMemoryAcademicDB
from tableref.pipeline import FileRefPipeline
from tableref.candidates import GeminiFullFileGenerator
from tableref.matching import ReferenceMatcher, SimpleSurnameMatcherConfig
from tableref.config import GeminiConfig, GeminiCredentials, IncludedStudiesPayload

# --- Setup ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def class_path(obj_or_cls: object) -> str:
    """Returns the full import path for a class or object's class."""
    cls = obj_or_cls if isinstance(obj_or_cls, type) else obj_or_cls.__class__
    return f"{cls.__module__}.{cls.__name__}"

def dataclass_to_dict(x: object) -> dict:
    """Converts a dataclass instance to a dictionary."""
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return x
    return {k: v for k, v in vars(x).items() if not k.startswith("_")}

def redact_secrets(d: dict, secret_keys: tuple = ("api_key", "token", "secret", "key", "password")) -> dict:
    """Recursively redacts secret keys from a dictionary."""
    def _r(v):
        if isinstance(v, dict):
            return {k: ("<REDACTED>" if k.lower() in secret_keys else _r(vv)) for k, vv in v.items()}
        if isinstance(v, list):
            return [_r(i) for i in v]
        return v
    return _r(json.loads(json.dumps(d)))

def stable_hash(payload: dict) -> str:
    """Creates a stable SHA256 hash of a dictionary."""
    s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def none_if_null(v):
    """Returns None if a value is an empty or null-like string."""
    return None if (isinstance(v, str) and v.strip().lower() in ("null", "none", "")) or v in ("",) else v

# --- Database Functions ---

def setup_database(path: str = "bibliography.db") -> tuple[Engine, dict[str, Table]]:
    """
    Initializes the database engine and reflects tables.
    
    Args:
        path: Path to the SQLite database file.

    Returns:
        A tuple containing the SQLAlchemy engine and a dictionary of table objects.
    """
    engine = create_engine(f"sqlite:///{path}", future=True)
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
    
    meta = MetaData()
    meta.reflect(bind=engine)
    
    tables = {
        "reviews": meta.tables["reviews"],
        "runs": meta.tables["review_expansion_runs"],
        "papers": meta.tables["review_expansion_papers"],
        "supplements": meta.tables["review_expansion_supplements"],
        "notes": meta.tables["review_expansion_notes"],
    }
    return engine, tables

# --- Pipeline and Provenance Setup ---

def setup_pipeline_and_provenance(academic_db_path: str) -> tuple[FileRefPipeline, InMemoryAcademicDB, dict]:
    """
    Initializes and configures the pipeline and provenance data.

    Args:
        academic_db_path: Path to the academic database JSON file.

    Returns:
        A tuple containing the configured pipeline, the academic database object,
        and a dictionary with provenance information.
    """
    db = InMemoryAcademicDB(academic_db_path)
    strategy_name = "grobid"
    gemini_config = GeminiConfig.build_prompts(IncludedStudiesPayload)

    pipeline = FileRefPipeline(
        candidate_generator=GeminiFullFileGenerator(config=gemini_config, creds=GeminiCredentials()),
        matcher=ReferenceMatcher(SimpleSurnameMatcherConfig()),
        output_root="table_output/",
    )

    candidate_config_dict = redact_secrets(dataclass_to_dict(gemini_config))
    matcher_config_dict = redact_secrets(dataclass_to_dict(SimpleSurnameMatcherConfig()))

    provenance = {
        "strategy_name": strategy_name,
        "candidate_generator_class": class_path(GeminiFullFileGenerator),
        "candidate_generator_config": candidate_config_dict,
        "matcher_class": class_path(ReferenceMatcher),
        "matcher_config": matcher_config_dict,
        "config_hash": stable_hash({
            "strategy_name": strategy_name,
            "candidate_generator_class": class_path(GeminiFullFileGenerator),
            "candidate_generator_config": candidate_config_dict,
            "matcher_class": class_path(ReferenceMatcher),
            "matcher_config": matcher_config_dict,
        }),
        "library_versions": {},
        "prompts_snapshot": {
            "system_prompt": gemini_config.system_prompt,
            "user_prompt": gemini_config.user_prompt,
        }
    }
    return pipeline, db, provenance

# --- Main Processing Logic ---

def process_review(paper_id: str, db: InMemoryAcademicDB, pipeline: FileRefPipeline, engine: Engine, tables: dict[str, Table], provenance: dict):
    """
    Runs the pipeline for a single paper and saves the results to the database.

    Args:
        paper_id: The ID of the paper to process.
        db: The academic database object.
        pipeline: The configured FileRefPipeline instance.
        engine: The SQLAlchemy database engine.
        tables: A dictionary of SQLAlchemy table objects.
        provenance: A dictionary containing provenance information for the run.
    """
    logger.info(f"Processing review: {paper_id}")

    # Handle known ID inconsistencies
    if paper_id == "Bente_2013":
        paper_id = "crimean_congo_hemorrhagic_fever__crimean-congo_hemorrhagic_fever_orthonairovirus"
    
    references = db.retrieve(paper_id, "references", provenance["strategy_name"])
    metadata = db.retrieve(paper_id, "metadata", provenance["strategy_name"])

    if not metadata or not metadata.file_path:
        logger.warning(f"Skipping {paper_id}: No metadata or file_path found.")
        return

    file_path = metadata.file_path
    # Normalize path for different OS and data source inconsistencies
    # This part might need adjustment based on the actual file paths
    replacements = {
        "zika_virus_disease__zika_virus_/": "",
        "covd-22_Variant_sars-cov-2/": "",
        "influenza___influenza-a-h1n1pdm09/": "",
        "human_avian_influenza__avian_influenza-a-h5n1/": "",
        "rift_valley_fever__rift_valley_fever_phlebovirus/": "",
        "crimean_congo_hemorrhagic_fever__crimean-congo_hemorrhagic_fever_orthonairovirus/": "",
        "mpox__monkeypox_virus/": "",
        "covid-19_Variant_sars-cov-2/": "",
        "covd-19_Variant_sars-cov-2/": "",
        "ebola_virus_disease__ebola_virus_/": ""
    }
    for old, new in replacements.items():
        file_path = file_path.replace(old, new)

    if paper_id == "crimean_congo_hemorrhagic_fever__crimean-congo_hemorrhagic_fever_orthonairovirus":
        file_path = "papersWP2/bunyavirales/2013_Bente.pdf"

    results = pipeline.run(
        pdf_path=file_path,
        references=list(references)
    )

    # --- Data Extraction ---
    summary = results.get("summary", {})
    payload = results.get("extraction_payload") or {}
    
    # --- Data Preparation ---
    exp_rows = []
    for i, item in enumerate(results.get("comparison_results", [])):
        exp_rows.append({
            "review_id": paper_id,
            "review_title": getattr(metadata, "title", None) or "",
            "review_doi": getattr(metadata, "doi", None),
            "ref_index": i,
            "ref_candidate_text": item.get("candidate_text"),
            "ref_has_match": bool(item.get("has_match", False)),
            "ref_match_score": item.get("match_score"),
            "ref_title": none_if_null(item.get("reference_title")),
            "ref_authors": item.get("reference_authors"),
            "ref_journal": none_if_null(item.get("reference_journal")),
            "ref_year": item.get("reference_year"),
            "ref_doi": none_if_null(item.get("reference_doi")),
            "ref_crossref_doi": none_if_null(item.get("crossref_doi")),
            "ref_crossref_error": none_if_null(item.get("crossref_error")),
        })

    supplement_rows = [
        {"s_label": s.get("label"), "s_href": str(s.get("href")), "s_content_note": s.get("content_note")}
        for s in results.get("supplements") or []
    ]
    notes_rows = [{"note_text": note} for note in results.get("notes") or []]

    # --- Database Transaction ---
    with engine.begin() as conn:
        # 1. Upsert Review
        review_values = {
            "review_id": paper_id,
            "review_title": getattr(metadata, "title", None) or "",
            "review_doi": getattr(metadata, "doi", None),
            "publication_year": getattr(metadata, "publication_year", None),
            "journal": getattr(metadata, "journal", None),
            "file_path": getattr(metadata, "file_path", None),
        }
        upsert_stmt = sqlite_insert(tables["reviews"]).values(created_at=dt.datetime.now(dt.UTC), **review_values)
        update_dict = {k: v for k, v in review_values.items() if v is not None}
        conn.execute(upsert_stmt.on_conflict_do_update(index_elements=["review_id"], set_=update_dict))

        # 2. Insert Run
        run_values = {
            **provenance,
            "review_id": paper_id,
            "run_at": dt.datetime.now(dt.UTC),
            "review_type": summary.get("review_type"),
            "declared_included_count": summary.get("declared_included_count"),
            "declared_included_evidence": (payload.get("declared_included_count") or {}).get("evidence"),
            "inclusion_criteria_summary": payload.get("inclusion_criteria_summary"),
            "supplements_found": summary.get("supplements_found"),
            "total_candidates": summary.get("total_candidates"),
            "candidates_with_matches": summary.get("candidates_with_matches"),
            "crossref_dois_found": summary.get("crossref_dois_found"),
            "extraction_payload_json": json.dumps(payload),
        }
        run_id = conn.execute(sqlite_insert(tables["runs"]).values(**run_values).returning(tables["runs"].c.run_id)).scalar_one()

        # 3. Insert Child Records
        if exp_rows:
            for r in exp_rows: r["run_id"] = run_id
            conn.execute(sqlite_insert(tables["papers"]), exp_rows)
        
        if supplement_rows:
            for r in supplement_rows: r["run_id"] = run_id
            conn.execute(sqlite_insert(tables["supplements"]), supplement_rows)

        if notes_rows:
            for r in notes_rows: r["run_id"] = run_id
            conn.execute(sqlite_insert(tables["notes"]), notes_rows)

    logger.info(
        f"Saved run_id={run_id} for review_id={paper_id} with {len(exp_rows)} refs, "
        f"{len(supplement_rows)} supplements, {len(notes_rows)} notes"
    )
