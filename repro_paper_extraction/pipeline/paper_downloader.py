# -*- coding: utf-8 -*-
"""
Batch downloader for expanded references.

This module provides functions to download scientific papers given a list of
review IDs or a list of DOIs from text files. It uses a multi-step approach:
1. Tries to find an Open Access version via Unpaywall and CORE.
2. Falls back to publisher websites using content negotiation and HTML scraping.
3. As a last resort, uses PyPaperBot which may use sources like Sci-Hub.

It is designed to be resumable, keeping track of downloaded DOIs to avoid
re-downloading.

Configuration via environment variables:
- UNPAYWALL_EMAIL: Your email address for the Unpaywall API (required for OA).
- CORE_API_KEY: Your API key for the CORE API (optional).
"""
from __future__ import annotations

import os
import re
import csv
import json
import time
import random
import logging
from typing import Iterable, List, Dict, Any, Optional, Tuple

import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus

from sqlalchemy import create_engine, MetaData, select, desc
# PyPaperBot is an optional dependency for the final fallback
try:
    from PyPaperBot.Downloader import downloadPapers
    from PyPaperBot.Crossref import getPapersInfoFromDOIs
    PYPAPERBOT_AVAILABLE = True
except ImportError:
    PYPAPERBOT_AVAILABLE = False

# --- Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15',
]

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": random.choice(USER_AGENTS)})

UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL")
CORE_API_KEY = os.environ.get("CORE_API_KEY")
MAX_FETCH_RETRIES = 3
BACKOFF_BASE = 0.6

# --- Utility Functions ---

def sanitize_filename(s: str) -> str:
    """Removes characters that are invalid in filenames."""
    return re.sub(r"[^\w\-.]+", "_", s)

def _parse_authors(raw: Any) -> List[str]:
    """Parses author strings into a list of names."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    s = str(raw)
    if ";" in s:
        return [p.strip() for p in s.split(";")]
    return [p.strip() for p in s.split(",")]

def first_author_fullname(authors: List[str]) -> str:
    """Generates a clean 'Firstname_Surname' string for the first author."""
    if not authors:
        return "UnknownAuthor"
    raw = authors[0].strip().replace(".", "")
    if "," in raw:
        parts = [p.strip() for p in raw.split(",", 1)]
        surname, given = parts[0], parts[1] if len(parts) > 1 else ""
    else:
        tokens = raw.split()
        surname = tokens[-1] if tokens else ""
        given = " ".join(tokens[:-1]) if len(tokens) > 1 else ""
    return sanitize_filename(f"{given} {surname}".strip().replace(" ", "_"))

def filename_from_year_author(year: Optional[Any], authors: List[str]) -> str:
    """Creates a filename like 'YEAR_FIRSTAUTHOR.pdf'."""
    y = str(year) if (year and str(year).isdigit()) else "UnknownYear"
    au = first_author_fullname(authors)
    return f"{y}_{au}.pdf"

def is_valid_pdf_bytes(content: bytes) -> bool:
    """Checks if a byte stream is a valid, non-encrypted PDF."""
    try:
        with fitz.open("pdf", stream=content) as doc:
            return not doc.is_encrypted and len(doc) > 0
    except Exception:
        return False

def _save_pdf_bytes(folder: str, filename: str, data: bytes) -> str:
    """Saves PDF bytes to a file, avoiding overwrites."""
    path = os.path.join(folder, sanitize_filename(filename))
    base, ext = os.path.splitext(path)
    candidate = path
    i = 2
    while os.path.exists(candidate):
        candidate = f"{base}({i}){ext}"
        i += 1
    with open(candidate, "wb") as f:
        f.write(data)
    return candidate

# --- Database and CrossRef Access ---

def get_review_references(
    review_id: str,
    db_path: str = "bibliography.db",
    *,
    latest_run_only: bool = True,
) -> List[Dict[str, Any]]:
    """Fetches references for a given review ID from the database."""
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    meta = MetaData()
    meta.reflect(bind=engine)
    papers_tbl = meta.tables["review_expansion_papers"]
    runs_tbl = meta.tables["review_expansion_runs"]

    columns = [
        papers_tbl.c.ref_index,
        papers_tbl.c.ref_authors,
        papers_tbl.c.ref_year,
        papers_tbl.c.ref_crossref_doi,
        papers_tbl.c.ref_doi,
    ]
    where_clauses = [papers_tbl.c.review_id == review_id]

    if latest_run_only:
        latest_run_subq = (
            select(runs_tbl.c.run_id)
            .where(runs_tbl.c.review_id == review_id)
            .order_by(desc(runs_tbl.c.run_at), desc(runs_tbl.c.run_id))
            .limit(1)
            .scalar_subquery()
        )
        where_clauses.append(papers_tbl.c.run_id == latest_run_subq)

    stmt = select(*columns).where(*where_clauses).order_by(papers_tbl.c.ref_index.asc())
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(stmt).mappings().all()]

def get_crossref_year_and_authors(doi: str) -> Tuple[Optional[int], List[str]]:
    """Queries CrossRef for a DOI to get year and authors for filename generation."""
    try:
        url = f"https://api.crossref.org/works/{quote_plus(doi)}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        msg = r.json().get("message", {})
        
        year = (msg.get("issued", {}).get("date-parts", [[]])[0] or [None])[0]
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in msg.get("author", [])]
        
        return year, authors
    except Exception as e:
        logger.debug(f"CrossRef lookup for filename failed for {doi}: {e}")
        return None, []

def filename_from_crossref(doi: str) -> str:
    """Builds a filename using metadata from CrossRef."""
    year, authors = get_crossref_year_and_authors(doi)
    return filename_from_year_author(year, authors)

# --- Download Strategies ---

def _download_bytes(url: str) -> bytes:
    """Downloads a URL with retries and backoff."""
    last_exc = None
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        try:
            r = SESSION.get(url, stream=True, timeout=40, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
    raise last_exc or RuntimeError("Download failed")

def download_via_oa(doi: str, folder: str, filename: str) -> Tuple[str, Optional[str]]:
    """Tries to download a paper from Open Access sources."""
    if UNPAYWALL_EMAIL:
        try:
            url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            js = r.json()
            pdf_url = (js.get("best_oa_location") or {}).get("url_for_pdf")
            if pdf_url:
                data = _download_bytes(pdf_url)
                if is_valid_pdf_bytes(data):
                    return "oa_unpaywall", _save_pdf_bytes(folder, filename, data)
        except Exception as e:
            logger.debug(f"Unpaywall failed for {doi}: {e}")
    
    return "oa_none", None

def download_via_publisher(doi: str) -> Tuple[str, Optional[bytes]]:
    """Tries to download a paper directly from the publisher."""
    try:
        url = f"https://doi.org/{doi}"
        headers = {"Accept": "application/pdf", "User-Agent": random.choice(USER_AGENTS)}
        r = requests.get(url, headers=headers, timeout=40, allow_redirects=True)
        if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", "").lower():
            if is_valid_pdf_bytes(r.content):
                return "negotiation", r.content
    except Exception as e:
        logger.debug(f"Content negotiation failed for {doi}: {e}")
    return "fallback_none", None

def download_via_pypaperbot(doi: str, folder: str, filename: str, mirror: Optional[str]) -> Tuple[str, Optional[str]]:
    """Uses PyPaperBot as a last resort for downloading."""
    if not PYPAPERBOT_AVAILABLE:
        return "pypaperbot_unavailable", None
    
    encoded_doi = quote_plus(doi)
    temp_path = os.path.join(folder, f"{encoded_doi}.pdf")
    try:
        info = getPapersInfoFromDOIs(doi, restrict=1)
        info.use_doi_as_filename = True
        downloadPapers([info], folder, SciHub_URL=mirror)
        if os.path.exists(temp_path):
            with open(temp_path, 'rb') as f:
                data = f.read()
            if is_valid_pdf_bytes(data):
                final_path = os.path.join(folder, sanitize_filename(filename))
                os.rename(temp_path, final_path)
                return "pypaperbot", final_path
    except Exception as e:
        logger.error(f"PyPaperBot failed for {doi}: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return "pypaperbot_failed", None

# --- Main Processing Logic ---

def load_downloaded_set(base_dir: str) -> set:
    """Loads the set of already downloaded DOIs."""
    path = os.path.join(base_dir, "downloaded_dois.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(x.lower() for x in json.load(f))
        except (json.JSONDecodeError, TypeError):
            pass
    return set()

def persist_downloaded_set(base_dir: str, downloaded_set: set):
    """Saves the set of downloaded DOIs."""
    path = os.path.join(base_dir, "downloaded_dois.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(downloaded_set)), f, indent=2)

def _download_single_doi(doi: str, folder: str, filename: str, scihub_mirror: Optional[str]) -> Tuple[str, Optional[str]]:
    """Runs the download strategies for a single DOI."""
    status, saved_path = "failed", None

    # Strategy 1: Open Access
    status, saved_path = download_via_oa(doi, folder, filename)
    
    # Strategy 2: Publisher
    if not saved_path:
        status, pdf_bytes = download_via_publisher(doi)
        if pdf_bytes:
            saved_path = _save_pdf_bytes(folder, filename, pdf_bytes)

    # Strategy 3: PyPaperBot
    if not saved_path:
        status, saved_path = download_via_pypaperbot(doi, folder, filename, scihub_mirror)
    
    return status, saved_path

def process_reviews(
    review_ids: Iterable[str],
    base_output_dir: str,
    db_path: str = "bibliography.db",
    latest_run_only: bool = True,
    scihub_mirror: Optional[str] = None,
):
    """Downloads papers for a list of review IDs by fetching references from the database."""
    os.makedirs(base_output_dir, exist_ok=True)
    downloaded = load_downloaded_set(base_output_dir)
    logger.info(f"Loaded {len(downloaded)} previously downloaded DOIs.")

    for review_id in review_ids:
        logger.info(f"Processing review: {review_id}")
        folder = os.path.join(base_output_dir, review_id)
        os.makedirs(folder, exist_ok=True)
        
        rows = get_review_references(review_id, db_path, latest_run_only=latest_run_only)
        manifest = []

        for row in rows:
            doi = (row.get("ref_crossref_doi") or row.get("ref_doi") or "").strip()
            if not doi or doi.lower() in downloaded:
                status = "skipped_existing" if doi else "missing_doi"
                manifest.append({"doi": doi, "status": status, **row})
                continue

            filename = filename_from_year_author(row.get("ref_year"), _parse_authors(row.get("ref_authors")))
            status, saved_path = _download_single_doi(doi, folder, filename, scihub_mirror)

            if saved_path:
                logger.info(f"Successfully downloaded {doi} via {status}")
                downloaded.add(doi.lower())
                persist_downloaded_set(base_output_dir, downloaded)
            else:
                logger.warning(f"Failed to download {doi}")

            manifest.append({"doi": doi, "status": status, "saved_path": saved_path, **row})
            time.sleep(random.uniform(0.5, 1.5))

        manifest_path = os.path.join(folder, "download_manifest.csv")
        if manifest:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
                writer.writeheader()
                writer.writerows(manifest)
        logger.info(f"Wrote manifest for {review_id} to {manifest_path}")

def download_papers_from_doi_txts(
    txt_paths: Iterable[str],
    base_output_dir: str,
    scihub_mirror: Optional[str] = None,
):
    """Downloads papers from one or more text files containing DOIs."""
    os.makedirs(base_output_dir, exist_ok=True)
    downloaded = load_downloaded_set(base_output_dir)
    logger.info(f"Loaded {len(downloaded)} previously downloaded DOIs.")

    for txt_path in txt_paths:
        if not os.path.exists(txt_path):
            logger.warning(f"TXT file not found: {txt_path}")
            continue

        folder_name = os.path.splitext(os.path.basename(txt_path))[0]
        folder = os.path.join(base_output_dir, folder_name)
        os.makedirs(folder, exist_ok=True)
        
        with open(txt_path, "r", encoding="utf-8") as f:
            dois = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        manifest = []
        for doi in dois:
            if doi.lower() in downloaded:
                manifest.append({"doi": doi, "status": "skipped_existing"})
                continue

            filename = filename_from_crossref(doi)
            status, saved_path = _download_single_doi(doi, folder, filename, scihub_mirror)

            if saved_path:
                logger.info(f"Successfully downloaded {doi} via {status}")
                downloaded.add(doi.lower())
                persist_downloaded_set(base_output_dir, downloaded)
            else:
                logger.warning(f"Failed to download {doi}")
            
            manifest.append({"doi": doi, "status": status, "saved_path": saved_path, "filename": filename})
            time.sleep(random.uniform(0.5, 1.5))

        manifest_path = os.path.join(folder, "download_manifest.csv")
        if manifest:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
                writer.writeheader()
                writer.writerows(manifest)
        logger.info(f"Wrote manifest for {txt_path} to {manifest_path}")