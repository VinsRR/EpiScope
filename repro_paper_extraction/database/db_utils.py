# -*- coding: utf-8 -*-
"""
Database utility functions for the review expansion pipeline.
"""
from __future__ import annotations
from typing import List, Dict, Iterable, Optional
from sqlalchemy import create_engine, MetaData, select, desc

def get_review_references(
    review_id: str,
    db_path: str = "bibliography.db",
    *,
    columns: Optional[Iterable[str]] = None,
    latest_run_only: bool = False,
) -> List[Dict]:
    """
    Return references for a specific review_id from the database.

    Args:
        review_id: The ID of the review to fetch references for.
        db_path: Path to the SQLite database file.
        columns: A list of column names to return. If None, returns a default set.
        latest_run_only: If True, only return references from the latest run for the review.

    Returns:
        A list of dictionaries, where each dictionary represents a reference.
    """
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    meta = MetaData()
    meta.reflect(bind=engine)

    runs_tbl = meta.tables["review_expansion_runs"]
    papers_tbl = meta.tables["review_expansion_papers"]

    # Available columns you can request
    colmap = {
        "run_id":            papers_tbl.c.run_id,
        "ref_index":         papers_tbl.c.ref_index,
        "review_id":         papers_tbl.c.review_id,
        "review_title":      papers_tbl.c.review_title,
        "review_doi":        papers_tbl.c.review_doi,
        "ref_title":         papers_tbl.c.ref_title,
        "ref_crossref_doi":  papers_tbl.c.ref_crossref_doi,
        "ref_doi":           papers_tbl.c.ref_doi,
        "ref_has_match":     papers_tbl.c.ref_has_match,
        "ref_match_score":   papers_tbl.c.ref_match_score,
        "ref_authors":       papers_tbl.c.ref_authors,
        "ref_journal":       papers_tbl.c.ref_journal,
        "ref_year":          papers_tbl.c.ref_year,
        "ref_candidate_text":papers_tbl.c.ref_candidate_text,
    }

    if columns is None:
        columns = ["review_id", "review_doi", "ref_title", "ref_crossref_doi","ref_has_match"]

    # Validate requested columns
    bad = [c for c in columns if c not in colmap]
    if bad:
        raise ValueError(f"Unknown column(s): {bad}. Allowed: {sorted(colmap)}")

    selected_cols = [colmap[c] for c in columns]

    # Base filter
    where_clause = (papers_tbl.c.review_id == review_id,)

    # If we want only the latest run, compute it via subquery
    if latest_run_only:
        latest_run_subq = (
            select(runs_tbl.c.run_id)
            .where(runs_tbl.c.review_id == review_id)
            .order_by(desc(runs_tbl.c.run_at), desc(runs_tbl.c.run_id))
            .limit(1)
            .scalar_subquery()
        )
        where_clause = where_clause + (papers_tbl.c.run_id == latest_run_subq,)

    stmt = (
        select(*selected_cols)
        .where(*where_clause)
        .order_by(papers_tbl.c.run_id.asc(), papers_tbl.c.ref_index.asc())
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
        # Convert RowMapping -> dict with only requested keys
        return [ {k: row[k] for k in columns} for row in rows ]
