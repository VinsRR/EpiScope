# -*- coding: utf-8 -*-
"""
Data preparation and metadata backfilling.

This script provides functions to fill in missing DOIs for references
in the database. It queries Crossref and OpenAlex APIs to find DOIs based on
title, authors, and year, using fuzzy matching to score candidates.
"""
import os
import time
import random
import logging
import requests
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy import create_engine, update, select, desc, MetaData

# --- Configuration ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Be polite to APIs: set a real contact email.
USER_AGENT = os.environ.get("USER_AGENT", "EpiScope-Backfill/1.0 (mailto:your.email@example.com)")
SLEEP_BASE = 0.15
SLEEP_JITTER = 0.10
MAX_RETRIES = 3

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

# --- Text Normalization and Scoring ---

def _norm_text(s: str) -> str:
    """Normalizes text for comparison by lowercasing, removing punctuation, and ASCII folding."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _title_sim(a: str, b: str) -> float:
    """Calculates title similarity score."""
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()

def _author_tokens(authors: List[str]) -> set:
    """Extracts a set of author last names for comparison."""
    toks = set()
    for a in authors:
        parts = _norm_text(a).split()
        if parts:
            toks.add(parts[-1])
    return toks

def _candidate_score(cand_title: str, cand_authors: List[str], q_title: str, q_authors: List[str]) -> float:
    """Scores a candidate reference against a query reference."""
    t_score = _title_sim(cand_title, q_title)
    q_last = _author_tokens(q_authors)
    c_last = _author_tokens(cand_authors)
    a_overlap = len(q_last & c_last) / max(1, len(q_last))
    # Title is weighted more heavily than author overlap.
    return 0.85 * t_score + 0.15 * a_overlap

# --- API Interaction ---

def _sleep():
    """Pauses execution to respect API rate limits."""
    time.sleep(SLEEP_BASE + random.random() * SLEEP_JITTER)

def _crossref_search(title: str, authors: List[str], year: Optional[int] = None) -> List[Dict]:
    """Searches the Crossref API for a given reference."""
    params = {"query.bibliographic": title, "rows": 10, "select": "DOI,title,author,issued"}
    if year:
        params["filter"] = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
    
    for _ in range(MAX_RETRIES):
        try:
            r = _session.get("https://api.crossref.org/works", params=params, timeout=20)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            _sleep()
            results = []
            for it in items:
                it_auths = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])]
                results.append({
                    "source": "crossref", "doi": it.get("DOI"),
                    "title": " ".join(it.get("title") or []), "authors": it_auths,
                })
            return results
        except requests.RequestException:
            time.sleep(0.5)
    return []

def _openalex_search(title: str) -> List[Dict]:
    """Searches the OpenAlex API as a fallback."""
    params = {"search": title, "per_page": 10}
    for _ in range(MAX_RETRIES):
        try:
            r = _session.get("https://api.openalex.org/works", params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            _sleep()
            results = []
            for w in data.get("results", []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                it_auths = [a.get("author", {}).get("display_name") for a in w.get("authorships", [])]
                results.append({
                    "source": "openalex", "doi": doi,
                    "title": w.get("title"), "authors": it_auths,
                })
            return results
        except requests.RequestException:
            time.sleep(0.5)
    return []

def find_doi(title: str, authors: List[str], year: Optional[int] = None, min_confidence: float = 0.8) -> Tuple[Optional[str], float, dict]:
    """
    Finds the best DOI for a reference by querying Crossref and OpenAlex.
    """
    def get_best_candidate(candidates, query_title, query_authors):
        best_item, best_score = None, -1.0
        for item in candidates:
            score = _candidate_score(item["title"], item["authors"], query_title, query_authors)
            if score > best_score:
                best_item, best_score = item, score
        return best_item, best_score

    cr_candidates = _crossref_search(title, authors, year=year)
    best_cr, score_cr = get_best_candidate(cr_candidates, title, authors)

    if score_cr >= min_confidence:
        return best_cr["doi"], score_cr, best_cr

    oa_candidates = _openalex_search(title)
    best_oa, score_oa = get_best_candidate(oa_candidates, title, authors)

    if score_cr > score_oa:
        return best_cr.get("doi"), score_cr, best_cr
    elif best_oa:
        return best_oa.get("doi"), score_oa, best_oa
        
    return None, max(score_cr, score_oa), {}

# --- Main Backfill Logic ---

def backfill_review(
    review_id: str,
    db_path: str = "bibliography.db",
    latest_run_only: bool = True,
    min_confidence: float = 0.8,
    dry_run: bool = False
):
    """
    Finds and fills missing DOIs for a specific review.
    """
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    meta = MetaData()
    meta.reflect(bind=engine)
    papers_tbl = meta.tables["review_expansion_papers"]

    # Fetch rows that need a DOI
    with engine.connect() as conn:
        # Simplified query to get candidates
        # In a full implementation, this would use a helper like in the original script
        where_clauses = [papers_tbl.c.review_id == review_id, papers_tbl.c.ref_doi.is_(None)]
        # Add latest_run_only logic if needed
        stmt = select(papers_tbl).where(*where_clauses)
        rows = conn.execute(stmt).mappings().all()

    logger.info(f"Found {len(rows)} references missing a DOI for review '{review_id}'.")
    
    updated_count = 0
    with engine.begin() as conn:
        for row in rows:
            title = (row.get("ref_title") or "").strip()
            authors = (row.get("ref_authors") or "").split(';')
            year = row.get("ref_year")

            if not title or not authors:
                continue

            doi, conf, meta = find_doi(title, authors, year=year, min_confidence=min_confidence)

            if doi and conf >= min_confidence:
                logger.info(f"Found DOI {doi} for '{title}' with confidence {conf:.2f}")
                updated_count += 1
                if not dry_run:
                    values = {
                        "ref_doi": doi,
                        "ref_has_match": True,
                        "ref_match_score": conf,
                        "ref_crossref_doi": doi if meta.get("source") == "crossref" else row.get("ref_crossref_doi"),
                    }
                    update_stmt = (
                        update(papers_tbl)
                        .where(papers_tbl.c.exp_id == row['exp_id'])
                        .values(**values)
                    )
                    conn.execute(update_stmt)
    
    logger.info(f"Backfill complete for '{review_id}'. Updated {updated_count} records.")

if __name__ == "__main__":
    # Example: backfill a specific review
    backfill_review("2009_Lessler_incubation_period", dry_run=False)