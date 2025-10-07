from __future__ import annotations

import re
import math
import warnings
import json
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Sequence, Union, OrderedDict
import string
import os

import numpy as np
import pandas as pd
import ollama
from tqdm import tqdm

# Optional dependencies
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    HAS_RAPIDFUZZ = False

try:
    from gmft.auto import CroppedTable, TableDetector, AutoTableFormatter
    from gmft.pdf_bindings import PyPDFium2Document
    HAS_GMFT = True
except ImportError:
    HAS_GMFT = False

try:
    from gmft_pymupdf import PyMuPDFDocument
    HAS_GMFT_PYMUPDF = True
except ImportError:
    HAS_GMFT_PYMUPDF = False

try:
    import img2table
    HAS_IMG2TABLE = True
except ImportError:
    HAS_IMG2TABLE = False


try:
    import nltk
    # Attempt to access stopwords, and if it fails, download them
    try:
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))
    except OSError:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")
        from nltk.corpus import stopwords
        _STOPWORDS = set(stopwords.words("english"))

except Exception:
    # fallback minimal stopword list if nltk not available
    print("NLTK not found, using a fallback stopword list.")
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "by", "for",
        "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "to", "from", "up", "down", "out", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"
    }


logger = logging.getLogger(__name__)

# Regex helpers
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", flags=re.I)
AUTHOR_YEAR_RE = re.compile(r"([A-Z][A-Za-z\-]+(?: et al\.)?)\s*,?\s*(\d{4})")


def default_normalize(text: str) -> str:
    """Default text normalization function."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # lowercasing
    text = text.lower()
    # strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # remove stopwords (token-level)
    tokens = [tok for tok in text.split() if tok not in _STOPWORDS]
    return " ".join(tokens)


import requests
import difflib
from typing import Callable


def fetch_doi_from_crossref(
    title: str,
    authors: Optional[Union[str, Sequence[str]]] = None,
    journal: Optional[str] = None,
    year: Optional[Union[str, int]] = None,
    rows: int = 5,
    timeout: float = 10.0,
    min_crossref_title_score: float = 0.85,
    user_agent_email: str = "you@example.com",
) -> Dict[str, Any]:
    """
    Query Crossref REST API to find DOI for a paper given bibliographic hints.
    Returns dict: {
        "doi": str | None,
        "score": float (0..1) measuring title similarity,
        "item": minimal crossref item dict (title, author, publisher, issued, DOI),
        "error": str | None
    }
    """
    out = {"doi": None, "score": 0.0, "item": None, "error": None}
    if not title or not isinstance(title, str):
        out["error"] = "no_title"
        return out

    # Build query string
    q_parts = []
    q_parts.append(title)
    if authors:
        if isinstance(authors, (list, tuple)):
            q_parts.append(" ".join(authors[:2]))
        else:
            q_parts.append(str(authors))
    if journal:
        q_parts.append(str(journal))
    q = " ".join([p for p in q_parts if p]).strip()
    params = {
        "query.bibliographic": q,
        "rows": rows,
    }
    # Prefer to filter by year if present (Crossref supports filter on from-pub-date/until-pub-date)
    if year:
        try:
            y = int(year)
            params["filter"] = f"from-pub-date:{y}-01-01,until-pub-date:{y}-12-31"
        except Exception:
            pass

    headers = {
        "User-Agent": f"ReferenceMatcher/1.0 (mailto:{user_agent_email})"
    }

    try:
        resp = requests.get("https://api.crossref.org/works", params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("message", {}).get("items", []) or []
        if not items:
            return out

        # score candidates by title similarity (prefer exact or close title matches)
        best = None
        best_score = -1.0
        for it in items:
            it_titles = it.get("title") or []
            it_title = it_titles[0] if it_titles else ""
            # choose similarity measure
            if HAS_RAPIDFUZZ and fuzz is not None:
                score = fuzz.token_set_ratio(title, it_title) / 100.0
            else:
                score = difflib.SequenceMatcher(None, title.lower(), it_title.lower()).ratio()
            if score > best_score:
                best_score = score
                best = it

        if best and best_score >= min_crossref_title_score:
            out["doi"] = best.get("DOI")
            out["score"] = float(best_score)
            # minimal item info
            out["item"] = {
                "title": best.get("title", []),
                "author": best.get("author", []),
                "container-title": best.get("container-title", []),
                "issued": best.get("issued"),
                "DOI": best.get("DOI"),
                "type": best.get("type"),
            }
            return out
        else:
            # best exists but below threshold: return with score
            if best:
                out["score"] = float(best_score)
                out["item"] = {
                    "title": best.get("title", []),
                    "author": best.get("author", []),
                    "container-title": best.get("container-title", []),
                    "issued": best.get("issued"),
                    "DOI": best.get("DOI"),
                    "type": best.get("type"),
                }
            return out

    except requests.HTTPError as he:
        out["error"] = f"http_error: {he}"
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


class LateInteractionMatcher:
    """
    Implements a ColBERT-style MaxSim late-interaction matcher.

    - Build TF-IDF prefilter on reference textual reprs.
    - Compute token embeddings for references using a pretrained transformer.
    - For each query (candidate), prefilter top-K refs, compute query token
      embeddings and MaxSim scores against shortlisted refs.
    """

    def __init__(
        self,
        references_norm: List[Any],
        normalize_fn = None,
        transform_model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        top_k_prefilter: int = 200,
        idf_smoothing: float = 1.0,
        max_ref_tokens: int = 256,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch and transformers are required for LateInteractionMatcher")
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for LateInteractionMatcher")

        self.refs = list(references_norm)
        self.normalize = normalize_fn if normalize_fn is not None else default_normalize
        self.top_k = top_k_prefilter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = transform_model_name
        self.idf_smoothing = idf_smoothing
        self.max_ref_tokens = max_ref_tokens

        # Build textual representations for TF-IDF prefilter
        self.ref_texts = []
        for r in self.refs:
            authors = getattr(r, "authors", []) or []
            first_author = self.normalize(authors[0] if authors else "")
            parts = [
                # self.normalize(r.raw_text) if hasattr(r, "raw_text") else "", # could just pass it all...
                     self.normalize(r.title) if hasattr(r, "title") else "",
                     first_author,
                     self.normalize(r.year) if hasattr(r, "year") else "",
                     self.normalize(r.journal) if hasattr(r, "journal") else "",
                    #  self.normalize(r.doi) if hasattr(r, "doi") else "" # doi is not among the attributes of GROBID Reference objects
                     ]
            txt = " . ".join([p for p in parts if p])
            if txt.strip():
                self.ref_texts.append(self.normalize(txt))
            else:
                self.ref_texts.append("")

        # TF-IDF vectorizer (prefilter)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        try:
            self.ref_tfidf = self.vectorizer.fit_transform(self.ref_texts)
        except ValueError as e:
            logger.warning(f"TF-IDF vectorization failed: {e}. Using fallback.")
            # Fallback: create dummy vectorizer
            self.ref_tfidf = None
            return

        # Load tokenizer + model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load transformer model {self.model_name}: {e}")
            raise

        # Build token-level embeddings for refs and compute DF for IDF
        self.ref_token_ids: List[List[int]] = []
        self.ref_token_embs: List[np.ndarray] = []  # list of (L, d) arrays
        df_counts = {}

        with torch.no_grad():
            for text in self.ref_texts:
                if not text.strip():
                    # Handle empty text
                    self.ref_token_ids.append([])
                    self.ref_token_embs.append(np.zeros((1, 768)))  # Default embedding size
                    continue
                    
                try:
                    enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_ref_tokens)
                    input_ids = enc["input_ids"].squeeze(0).to(self.device)  # (L,)
                    attention_mask = enc["attention_mask"].squeeze(0).to(self.device)
                    outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                    last_hidden = outputs.last_hidden_state.squeeze(0)  # (L, d)
                    token_embs = last_hidden.cpu()
                    # normalize token vectors
                    token_embs = token_embs / (token_embs.norm(dim=1, keepdim=True) + 1e-12)
                    ids = input_ids.cpu().tolist()
                    self.ref_token_ids.append(ids)
                    self.ref_token_embs.append(token_embs.numpy())
                    unique_ids = set(ids)
                    for tid in unique_ids:
                        df_counts[tid] = df_counts.get(tid, 0) + 1
                except Exception as e:
                    logger.warning(f"Failed to process reference text: {text[:50]}... Error: {e}")
                    self.ref_token_ids.append([])
                    self.ref_token_embs.append(np.zeros((1, 768)))

        # compute idf
        N = len(self.ref_texts)
        self.idf = {}
        for tid, df in df_counts.items():
            self.idf[tid] = math.log((N + self.idf_smoothing) / (df + self.idf_smoothing)) + 1.0

    def split_subcands(self, text: str) -> List[str]:
        """Heuristic splitting for multi-reference cells."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        parts = [p.strip() for p in re.split(r"[;|/]\s*", text) if p.strip()]
        if len(parts) > 1:
            return parts
        years = list(YEAR_RE.finditer(text))
        if len(years) > 1:
            chunks = []
            for i, m in enumerate(years):
                start = max(0, m.start() - 40)
                end = years[i + 1].start() if i + 1 < len(years) else len(text)
                chunk = text[start:end].strip(" ,;.")
                if chunk:
                    chunks.append(chunk)
            return chunks
        return [text.strip()] if text.strip() else []

    def compute_query_token_embs(self, text: str, max_query_tokens: int = 128) -> Tuple[List[int], np.ndarray]:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        t = self.normalize(text)
        if not t.strip():
            # Return empty embeddings for empty text
            return [], np.zeros((1, 768))
            
        try:
            enc = self.tokenizer(t, return_tensors="pt", truncation=True, max_length=max_query_tokens)
            input_ids = enc["input_ids"].squeeze(0).to(self.device)
            attention_mask = enc["attention_mask"].squeeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                last_hidden = outputs.last_hidden_state.squeeze(0).cpu()
            token_embs = last_hidden / (last_hidden.norm(dim=1, keepdim=True) + 1e-12)
            return input_ids.cpu().tolist(), token_embs.numpy()
        except Exception as e:
            logger.warning(f"Failed to compute query embeddings for: {text[:50]}... Error: {e}")
            return [], np.zeros((1, 768))

    def maxsim_score(self, q_ids: List[int], q_embs: np.ndarray, ref_idx: int) -> Tuple[float, List[Tuple[int, int, float]]]:
        if not q_ids or ref_idx >= len(self.ref_token_embs):
            return 0.0, []
            
        r_embs = self.ref_token_embs[ref_idx]  # (n, d)
        if r_embs.size == 0:
            return 0.0, []
            
        # similarity matrix
        S = np.dot(q_embs, r_embs.T)  # (m, n)
        max_per_token = S.max(axis=1)  # (m,)
        argmax_pos = S.argmax(axis=1).tolist()
        q_idf_weights = []
        for qtid in q_ids:
            q_idf_weights.append(self.idf.get(qtid, 0.5))
        q_idf_weights = np.array(q_idf_weights)
        weighted = (max_per_token * q_idf_weights).sum()
        norm = q_idf_weights.sum() + 1e-12
        score = float(weighted / norm)
        evidence = []
        for i, (pos, sim) in enumerate(zip(argmax_pos, max_per_token.tolist())):
            evidence.append((i, int(pos), float(sim)))
        return score, evidence


    def match_candidate(self, candidate: str, top_n: int = 1, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Match a candidate and return up to `top_n` matches sorted by score (highest first).
        By default top_n=1 so each candidate is matched to only the single best reference.
        Matches with score < min_score are discarded.
        """
        if not isinstance(candidate, str):
            candidate = str(candidate) if candidate is not None else ""
            
        if not candidate or len(candidate.strip()) == 0:
            return [{"matched_index": None, "score": 0.0, "reason": "too_short"}]

        # Skip if TF-IDF failed to initialize
        if self.ref_tfidf is None:
            return [{"matched_index": None, "score": 0.0, "reason": "vectorizer_failed"}]

        # DOI fast path
        m = DOI_RE.search(candidate)
        if m:
            doi = m.group(0).lower()
            hits = []
            for r in self.refs:
                if hasattr(r, "doi") and r.doi and doi in str(r.doi).lower():
                    hits.append({
                        "matched_index": getattr(r, "index", None),
                        "score": 1.0,
                        "reason": "doi_exact",
                        "matched_title": getattr(r, "title", None),
                        "matched_doi": getattr(r, "doi", None),
                        "matched_first_author": getattr(r, "first_author", None),
                    })
            if hits:
                # top_n may be >1; return up to top_n doi hits (usually 1)
                return sorted(hits, key=lambda x: -x["score"])[:top_n]

        subcands = self.split_subcands(candidate)
        if not subcands:
            return [{"matched_index": None, "score": 0.0, "reason": "no_subcandidates"}]

        all_out = {}

        # precompute tfidf vectors for subcands
        sub_tfidfs = []
        for sub in subcands:
            try:
                sub_tfidf = self.vectorizer.transform([self.normalize(sub)])
                sub_tfidfs.append(sub_tfidf)
            except Exception as e:
                logger.warning(f"TF-IDF transform failed for subcandidate: {sub[:50]}... Error: {e}")
                sub_tfidfs.append(None)

        for sub, sub_tfidf in zip(subcands, sub_tfidfs):
            if sub_tfidf is None:
                continue
                
            try:
                sims = cosine_similarity(sub_tfidf, self.ref_tfidf).ravel()
                top_idxs = np.argsort(-sims)[: self.top_k]
                q_ids, q_embs = self.compute_query_token_embs(sub)

                for ridx in top_idxs:
                    if ridx >= len(self.refs):
                        continue
                        
                    sc, evidence = self.maxsim_score(q_ids, q_embs, int(ridx))
                    # boosts
                    ref = self.refs[ridx]
                    if hasattr(ref, "year") and ref.year and str(ref.year) in sub:
                        sc = min(1.0, sc + 0.08)
                    fa = getattr(ref, "first_author_norm", None) or getattr(ref, "first_author", "")
                    if fa and isinstance(fa, str) and fa.split()[-1] in self.normalize(sub):
                        sc = min(1.0, sc + 0.08)
                    # filter by min_score
                    if sc < min_score:
                        continue

                    idx = getattr(ref, "index", ridx)
                    rec = {
                        "matched_index": idx,
                        "score": float(sc),
                        "reason": "maxsim",
                        "matched_title": getattr(ref, "title", None),
                        "matched_doi": getattr(ref, "doi", None),
                        "matched_first_author": getattr(ref, "first_author", None),
                        "matched_subcandidate": sub,
                        "evidence": evidence,
                    }
                    prev = all_out.get(idx)
                    if not prev or rec["score"] > prev["score"]:
                        all_out[idx] = rec
            except Exception as e:
                logger.warning(f"MaxSim matching failed for subcandidate: {sub[:50]}... Error: {e}")
                continue

        if not all_out:
            return [{"matched_index": None, "score": 0.0, "reason": "no_match"}]

        out = sorted(all_out.values(), key=lambda x: -x["score"])
        # return only top_n matches
        return out[:max(1, int(top_n))]


import re
import unicodedata
from collections import defaultdict
from typing import List, Any, Optional, Dict, Tuple

# try fast fuzzy engine, fallback to difflib
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False

# small helper: extract surname heuristic (same logic as discussed earlier)
def _extract_surname(author_string: str) -> str:
    if not author_string:
        return ""
    a = author_string.strip()
    if "," in a:
        left = a.split(",")[0].strip()
        if len(left) > 1 and not re.fullmatch(r"[A-Z]\.?", left, flags=re.I):
            tokens = left.split()
            return tokens[-1]
    tokens = a.split()
    return tokens[-1] if tokens else ""

def _normalize_small(s: str, normalize_fn):
    # wrapper to use the existing normalization function when available
    if s is None:
        return ""
    try:
        return normalize_fn(s)
    except Exception:
        # fallback simple ascii normalization
        s = unicodedata.normalize("NFKD", str(s))
        s = s.encode("ascii", "ignore").decode("ascii", "ignore")
        s = re.sub(r"[^\w\s]", " ", s).lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

def _fuzzy_score(a: str, b: str) -> int:
    if _HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b)
    else:
        return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)

class SimpleSurnameMatcher:
    """
    Lightweight matcher that couples short surname-based candidates to a single
    reference from the provided references_norm list.
    Interface compatible with LateInteractionMatcher.match_candidate (returns list),
    but always returns at most one final matched item (since each short ref maps to one reference).
    """

    def __init__(
        self,
        references_norm: List[Any],
        normalize_fn=None,
        fuzzy_threshold: int = 70,
        fuzzy_title_threshold: int = 65,
    ):
        self.refs = list(references_norm)
        self.normalize = normalize_fn if normalize_fn is not None else default_normalize
        self.fuzzy_threshold = int(fuzzy_threshold)
        self.fuzzy_title_threshold = int(fuzzy_title_threshold)

        # build surname index: norm_surname -> list of entries
        # each entry: {"index": idx, "ref": ref, "year": year, "title": title, "raw_text": raw_text, "first_author_norm": fa_norm}
        self.surname_index = defaultdict(list)
        for i, r in enumerate(self.refs):
            authors = getattr(r, "authors", []) or []
            if authors:
                first_author = authors[0]
            else:
                first_author = ""
            surname = _extract_surname(first_author)
            surname_norm = _normalize_small(surname, self.normalize)
            entry = {
                "index": getattr(r, "index", i),
                "ref": r,
                "year": getattr(r, "year", None),
                "title": getattr(r, "title", "") or "",
                "raw_text": getattr(r, "raw_text", "") or "",
                "first_author_norm": surname_norm,
            }
            if surname_norm:
                self.surname_index[surname_norm].append(entry)

        # also keep a flattened list of all candidate surnames for fuzzy search
        self.unique_surnames = list(self.surname_index.keys())

        # prebuild simple title strings normalized for fuzzy
        self._norm_titles = {}
        for i, r in enumerate(self.refs):
            txt = (getattr(r, "title", "") or "") + " . " + (getattr(r, "raw_text", "") or "")
            self._norm_titles[getattr(r, "index", i)] = _normalize_small(txt, self.normalize)

    def match_candidate(self, candidate: str, top_n: int = 1, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Return list with single best match dict or single no-match dict.
        Dict keys: matched_index, score (0..1), reason, matched_title, matched_doi, matched_first_author
        """
        # normalize candidate input
        if isinstance(candidate, dict) and 'candidate' in candidate:
            candidate_str = candidate['candidate']
        else:
            candidate_str = candidate

        if not candidate_str or not str(candidate_str).strip():
            return [{"matched_index": None, "score": 0.0, "reason": "too_short"}]

        cand_norm = _normalize_small(candidate_str, self.normalize)

        # try to find an exact surname token in candidate text
        cand_tokens = [t for t in re.split(r"\s+", cand_norm) if t]
        exact_candidates = []
        for tok in cand_tokens:
            if tok in self.surname_index:
                exact_candidates.extend(self.surname_index[tok])

        # If we found exact surname candidates, disambiguate
        if exact_candidates:
            # if only one candidate -> return it
            if len(exact_candidates) == 1:
                e = exact_candidates[0]
                return [{
                    "matched_index": e["index"],
                    "score": 1.0,
                    "reason": "exact_surname",
                    "matched_title": getattr(e["ref"], "title", None),
                    "matched_doi": getattr(e["ref"], "doi", None),
                    "matched_first_author": getattr(e["ref"], "first_author", None),
                }]

            # still ambiguous: fuzzy-match the candidate string against titles/raw_text of the candidate refs
            best_score = -1
            best_entry = None
            for e in exact_candidates:
                idx = e["index"]
                title_norm = self._norm_titles.get(idx, "")
                sc = _fuzzy_score(cand_norm, title_norm)
                if sc > best_score:
                    best_score = sc
                    best_entry = e
            # accept if above threshold
            if best_entry and best_score >= self.fuzzy_title_threshold:
                e = best_entry
                return [{
                    "matched_index": e["index"],
                    "score": float(best_score / 100.0),
                    "reason": f"fuzzy_title_disambiguation({best_score})",
                    "matched_title": getattr(e["ref"], "title", None),
                    "matched_doi": getattr(e["ref"], "doi", None),
                    "matched_first_author": getattr(e["ref"], "first_author", None),
                }]
            # otherwise pick the best entry anyway but mark low confidence
            e = best_entry or exact_candidates[0]
            return [{
                "matched_index": e["index"],
                "score": float(max(0.0, (best_score if best_score >= 0 else 0) / 100.0)),
                "reason": "ambiguous_surname_best_guess",
                "matched_title": getattr(e["ref"], "title", None),
                "matched_doi": getattr(e["ref"], "doi", None),
                "matched_first_author": getattr(e["ref"], "first_author", None),
            }]

        # No exact surname token match — try fuzzy across all known surnames
        best_score = -1
        best_surname = None
        for surname in self.unique_surnames:
            sc = _fuzzy_score(cand_norm, surname)
            if sc > best_score:
                best_score = sc
                best_surname = surname

        if best_surname and best_score >= self.fuzzy_threshold:
            # pick the top candidate for that surname (if multiple, try year, then title fuzzy)
            cand_list = self.surname_index[best_surname]
            if len(cand_list) == 1:
                e = cand_list[0]
                return [{
                    "matched_index": e["index"],
                    "score": float(best_score / 100.0),
                    "reason": f"fuzzy_surname({best_score})",
                    "matched_title": getattr(e["ref"], "title", None),
                    "matched_doi": getattr(e["ref"], "doi", None),
                    "matched_first_author": getattr(e["ref"], "first_author", None),
                }]

            # fallback: fuzzy-match against titles
            best_title_score = -1
            best_entry = None
            for e in cand_list:
                idx = e["index"]
                title_norm = self._norm_titles.get(idx, "")
                sc2 = _fuzzy_score(cand_norm, title_norm)
                if sc2 > best_title_score:
                    best_title_score = sc2
                    best_entry = e
            if best_entry:
                return [{
                    "matched_index": best_entry["index"],
                    "score": float(max(best_score, best_title_score) / 100.0),
                    "reason": f"fuzzy_surname_title({best_score},{best_title_score})",
                    "matched_title": getattr(best_entry["ref"], "title", None),
                    "matched_doi": getattr(best_entry["ref"], "doi", None),
                    "matched_first_author": getattr(best_entry["ref"], "first_author", None),
                }]

        # give up
        return [{"matched_index": None, "score": 0.0, "reason": "no_match"}]


class ReferenceMatcher:
    """
    Wrapper that exposes a match_candidate(candidate, references_norm) API compatible
    with the original ReferenceMatcher. You can set matcher_backend="simple" in kwargs
    when creating ReferenceMatcher to use the lightweight SimpleSurnameMatcher.
    """

    def __init__(self, normalize_fn=None, **late_interaction_kwargs):
        self.normalize = normalize_fn if normalize_fn is not None else default_normalize
        self._lim_kwargs = late_interaction_kwargs
        self._lim: Optional[Any] = None
        self._cached_refs_id = None
        self.min_candidate_length = late_interaction_kwargs.get("min_candidate_length", 3)
        # which backend: "late" (default) or "simple"
        self.matcher_backend = late_interaction_kwargs.get("matcher_backend", "simple")

    def prepare(self, references_norm: List[Any]):
        """Build the chosen matcher for a given set of references."""
        refs_id = id(references_norm)
        if self._lim is not None and refs_id == self._cached_refs_id:
            return

        # choose backend
        if self.matcher_backend == "simple":
            try:
                # assert False, "Debug: stop here"
                self._lim = SimpleSurnameMatcher(references_norm, normalize_fn=self.normalize,
                                                 fuzzy_threshold=self._lim_kwargs.get("fuzzy_threshold", 70),
                                                 fuzzy_title_threshold=self._lim_kwargs.get("fuzzy_title_threshold", 65))
                self._cached_refs_id = refs_id
                return
            except Exception as e:
                logger.error(f"Failed to initialize SimpleSurnameMatcher: {e}")
                self._lim = None
                self._cached_refs_id = None
                return

        # fallback to LateInteractionMatcher (original heavy matcher)
        try:
            self._lim = LateInteractionMatcher(references_norm, normalize_fn=self.normalize, **self._lim_kwargs)
            self._cached_refs_id = refs_id
        except Exception as e:
            logger.error(f"Failed to initialize LateInteractionMatcher: {e}")
            self._lim = None
            self._cached_refs_id = None


    def match_candidate(self, candidate: Union[str, Dict[str, Any]], references_norm: Optional[List[Any]] = None,
                        top_n: int = 1, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Match a candidate against references. For compatibility we still return a list,
        but by default it contains the single best match (top_n=1). Pass top_n>1 to get more.
        """
        # Handle case where candidate is a dict (fix for the original error)
        if isinstance(candidate, dict):
            if 'candidate' in candidate:
                candidate_str = candidate['candidate']
            else:
                logger.warning(f"Candidate dict missing 'candidate' key: {candidate}")
                return [{"matched_index": None, "score": 0.0, "reason": "invalid_candidate_format"}]
        else:
            candidate_str = candidate

        # Ensure candidate is a string
        if not isinstance(candidate_str, str):
            candidate_str = str(candidate_str) if candidate_str is not None else ""


        if references_norm is not None:
            refs_id = id(references_norm)
            if refs_id != self._cached_refs_id:
                # rebuild
                self.prepare(references_norm)
        
        if self._lim is None:
            logger.error("ReferenceMatcher.prepare(references_norm) must be called before matching or failed to initialize")
            return [{"matched_index": None, "score": 0.0, "reason": "matcher_not_initialized"}]
        
        if not candidate_str or len(candidate_str.strip()) < self.min_candidate_length:
            return [{"matched_index": None, "score": 0.0, "reason": "too_short"}]

        # delegate to LateInteractionMatcher; forward top_n and min_score
        try:
            matches = self._lim.match_candidate(candidate_str, top_n=top_n, min_score=min_score)
        except Exception as e:
            logger.error(f"Matching failed for candidate: {candidate_str[:50]}... Error: {e}")
            matches = [{"matched_index": None, "score": 0.0, "reason": "matching_error"}]

        return matches

    def match_candidates(self, candidates: List[Union[str, Dict[str, Any]]], references_norm: Optional[List[Any]] = None,
                         top_n: int = 1, min_score: float = 0.0) -> List[List[Dict[str, Any]]]:
        """Batch version of match_candidate. Each returned list contains up to top_n matches."""
        results = []
        for c in candidates:
            try:
                res = self.match_candidate(c, references_norm=references_norm, top_n=top_n, min_score=min_score)
            except Exception as e:
                logger.error(f"Failed to match candidate: {c}. Error: {e}")
                res = [{"matched_index": None, "score": 0.0, "reason": "matching_error"}]
            results.append(res)
        return results


class TableExtractor:
    """Extract tables and build candidate strings. No matching logic here."""

    def __init__(self, pdf_path: Union[str, Path], output_root: Optional[Union[str, Path]] = None, debug: bool = False):
        self.pdf_path = str(pdf_path)
        self.paper_id = Path(pdf_path).stem
        self.base_dir = Path(pdf_path).parent
        self.output_root = Path(output_root) if output_root else self.base_dir
        self.tables_dir = self.output_root / f"{self.paper_id}_tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.debug = debug

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a pandas DataFrame."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype(str).str.strip()
        return df

    def _looks_like_header(self, text: str) -> bool:
        """Check if text looks like a table header."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
            
        text_lower = text.lower()
        header_keywords = ['author','study','year','sample','method','outcome','result','table','figure','reference','citation','source']
        words = text_lower.split()
        if len(words) <= 6:
            header_word_count = sum(1 for word in words if any(kw in word for kw in header_keywords))
            return header_word_count >= max(1, int(len(words) * 0.5))
        return False

    def build_candidates_from_tables(self, tables: Sequence[Dict[str, Any]], min_length: int = 8) -> List[Dict[str, Any]]:
        """Build candidate strings from table data."""
        candidates: List[Dict[str, Any]] = []
    
        candidates_set = set()         
        for t in tables:
              # to avoid duplicates within the same table
            # print(len(t.get('ref_lines')))
            if t.get('ref_lines') is None:
                continue
            # else:
            #     for line in t.get('ref_lines', []):
            #         print(line)
            for line in t.get('ref_lines', []):
                candidates_set.add(line)  # include full table text as candidate
        print(candidates_set)
        for c in list(candidates_set):
            if not c or not isinstance(c, str):
                continue
            c_str = c.strip()
            candidates.append({
                'candidate': c_str,
                'page': None,
                'table_index': None,
                'row_index': None,
                'source': None
            })

        return candidates

    def extract_gmft_tables(self) -> List[Dict]:
        """Extract tables using GMFT."""
        results = []
        if not HAS_GMFT:
            logger.info("gmft not installed -> skipping")
            return results

        doc = None
        try:
            # Use PyMuPDF binding if available, otherwise fallback to PyPDFium2
            if HAS_GMFT_PYMUPDF:
                doc = PyMuPDFDocument(self.pdf_path)
            else:
                doc = PyPDFium2Document(self.pdf_path)
            
            detector = TableDetector()
            formatter = AutoTableFormatter()

            for page_number in range(len(doc)):
                page = doc[page_number]
                
                # Extract tables from page
                cropped_tables = detector.extract(page)
                
                for t_idx, cropped_table in enumerate(cropped_tables):
                    try:
                        # Format the cropped table
                        formatted_table = formatter.extract(cropped_table)
                        df = formatted_table.df()
                        
                        if df is None or df.empty or df.shape[0] < 2 or df.shape[1] < 2:
                            continue

                        # Clean the dataframe
                        df = self._clean_df(df)
                        
                        if df is None or df.empty:
                            continue
                        
                        csv_path = self.tables_dir / f"{self.paper_id}_gmft_p{page_number + 1}_t{t_idx}.csv"
                        df.to_csv(str(csv_path), index=False)

                        results.append({
                            "page": page_number + 1,  # 1-indexed
                            "table_index": t_idx,
                            "data": df.fillna("").values.tolist(),
                            "csv_path": str(csv_path),
                            "source": "gmft"
                        })
                    except Exception as te:
                        logger.warning(f"gmft: failed to process table p={page_number + 1} t={t_idx}: {te}")
                        continue

        except Exception as e:
            logger.warning(f"gmft extraction failed: {e}")
        finally:
            # Ensure we close the document to release resources
            if doc is not None:
                try:
                    doc.close()
                except Exception:
                    pass

        logger.info(f"gmft extracted {len(results)} tables")
        return results



    # other option: https://ollama.com/library/qwen2.5vl seems also pretty good
    def extract_tables_with_vision_llm(self) -> List[Dict]:
        """
        Alternative pipeline:
        - Uses EfficientDet to locate table regions
        - For each region, sends the image to Llama 3.2 Vision via Ollama
        asking for Markdown table extraction
        - Parses Markdown, saves to CSV, and returns metadata
        """
        import cv2
        import fitz
        import layoutparser as lp
        import ollama
        import pandas as pd
        import os
        import tempfile

        results = []
        model_name = "llama3.2-vision"

        # Initialize model client
        client = ollama.Client()

        # Initialize layout detector
        detector = lp.AutoLayoutModel("lp://efficientdet/PubLayNet")

        pdf = fitz.open(self.pdf_path)
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            pix = page.get_pixmap(dpi=600)
            img = cv2.imdecode(
                np.frombuffer(pix.tobytes("png"), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            layout = detector.detect(img_rgb)
            table_regions = [b for b in layout if b.type.lower() == "table"]

            for region_idx, tbl in enumerate(table_regions):
                x1, y1, x2, y2 = map(int, tbl.coordinates)
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Save region to temp file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    crop_path = tf.name
                cv2.imwrite(crop_path, crop)

                logging.info(f"Processing page {page_idx}, table {region_idx} with Llama Vision")
                # Prompt Llama–Vision
                response = client.chat(
                    model=model_name,
                    messages=[{
                        'role': 'user',
                        'content': 'Extract the COMPLETE table from this image and return it in Markdown format. The output should CONTAIN ONLY the Markdown table, without any additional text or explanation. If the table is empty or cannot be extracted, return an empty table with headers only. Ensure that the table is well-formed and adheres to Markdown syntax.',
                        'images': [crop_path]
                    }]
                )
                os.unlink(crop_path)

                md = response['message']['content']
                logger.info("Extracted Markdown:", md)
                df = self._parse_markdown_table(md)
                if df.empty:
                    continue

                csv_name = f"{self.paper_id}_p{page_idx}_t{region_idx}_vision_llm.csv"
                os.makedirs(self.tables_dir, exist_ok=True)
                csv_path = self.tables_dir / csv_name
                df.to_csv(str(csv_path), index=False)

                results.append({
                    "page": page_idx,
                    "table_index": region_idx,
                    "bbox": (x1, y1, x2, y2),
                    "data": df.fillna("").values.tolist(),
                    "csv_path": str(csv_path),
                    "source": model_name 
                })

        pdf.close()
        return results



    def _parse_markdown_table(self, md: str) -> pd.DataFrame:
        """
        Parses a GitHub-flavored Markdown table into a pandas DataFrame.
        """
        import pandas as pd
        lines = [l.strip() for l in md.strip().splitlines() if l.strip()]
        if len(lines) < 2: 
            return pd.DataFrame()
        headers = [h.strip() for h in lines[0].strip('|').split('|')]
        rows = []
        for line in lines[2:]:
            parts = [c.strip() for c in line.strip('|').split('|')]
            if len(parts) == len(headers):
                rows.append(parts)
        return pd.DataFrame(rows, columns=headers)


    # https://medium.com/better-programming/extracting-tables-from-images-in-python-made-easy-ier-3be959555f6f
    def extract_img2table_tables(self, debug: bool = True, debug_dir: Optional[str] = None) -> List[Dict]:
        """
        Extract tables from PDF using layoutparser for detection and img2table for extraction.
        """
        import cv2
        import fitz  # PyMuPDF
        import layoutparser as lp
        from img2table.ocr import TesseractOCR
        import os

        results = []

        # Initialize models
        model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet")
        # psm instrucrtions: https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
        ocr = TesseractOCR(n_threads=1, lang="eng", psm=11)  # Sparse text. Find as much text as possible in no particular order.
        
        if debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        pdf = fitz.open(self.pdf_path)

        for page_num in tqdm(range(len(pdf)), desc="Processing PDF pages"):
            # Convert PDF page to image
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=600) # higher dpi for better OCR
            img_data = pix.tobytes("png")
            # more on using cv2: https://medium.com/@rajashekarganiger2002/detect-and-extract-table-data-using-opencv-3039df2b80b0
            img = cv2.imdecode(np.frombuffer(img_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue

            # Detect table regions
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            layout = model.detect(img_rgb)
            table_regions = [b for b in layout if b.type.lower() == "table"]

            for region_idx, table_region in enumerate(table_regions):
                # Crop table region
                x1, y1, x2, y2 = map(int, table_region.coordinates)
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue

                # Extract tables from crop
                tables = self._extract_tables_from_crop(crop, ocr)
                
                # Save each extracted table
                for table_idx, df in enumerate(tables):
                    logger.info(f"Extracted table on page {page_num}, region {region_idx}, table {table_idx} with shape {df.shape}")
                    if df.empty:
                        continue
                        
                    # Save to CSV
                    csv_name = f"{self.paper_id}_p{page_num}_r{region_idx}_t{table_idx}.csv"
                    csv_path = self.tables_dir / csv_name
                    
                    os.makedirs(self.tables_dir, exist_ok=True)
                    df.to_csv(str(csv_path), index=False)
                    
                    # Extract references
                    ref_lines = self._extract_references_with_ollama(df, csv_path)
                    
                    results.append({
                        "page": page_num,
                        "region": region_idx,
                        "table": table_idx,
                        "data": df.fillna("").values.tolist(),
                        "csv_path": str(csv_path),
                        "ref_lines": ref_lines,
                        "source": "layoutparser+img2table"
                    })

        pdf.close()
        logger.info(f"Extracted {len(results)} tables total")
        return results


    def _extract_tables_from_crop(self, crop_img, ocr):
        """
        Use img2table to extract all tables from a cropped image.
        Returns list of cleaned DataFrames.
        """
        from img2table.document import Image
        import tempfile
        import cv2
        import pandas as pd

        # Save crop to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpf:
            tmp_path = tmpf.name
        
        cv2.imwrite(tmp_path, crop_img)
        
        try:
            # Extract tables with img2table
            doc = Image(tmp_path, detect_rotation=False)
            extracted = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=True,
                implicit_columns=True,
                min_confidence=30
            )
            
            # Clean and return DataFrames
            tables = []
            for table in extracted:
                df = table.df if hasattr(table, "df") else table
                if df is not None and not df.empty:
                    # cleaned_df = self._clean_dataframe(df)
                    # if not cleaned_df.empty:
                    if df is not None and not df.empty:
                        tables.append(df)

            return tables
            
        except Exception as e:
            logger.debug(f"Table extraction failed: {e}")
            return []
        finally:
            os.unlink(tmp_path)


    def _clean_dataframe(self, df):
        """
        Basic DataFrame cleaning.
        """
        import pandas as pd
        
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        if df.empty:
            return df
        
        # Use first row as header if it's mostly text
        if len(df) > 1:
            first_row = df.iloc[0].astype(str)
            text_count = first_row.str.contains('[a-zA-Z]', na=False).sum()
            if text_count / len(first_row) >= 0.7:
                df.columns = first_row
                df = df.drop(df.index[0]).reset_index(drop=True)
        
        # Fill NaN values
        return df.fillna("")


    def _extract_references_with_ollama(self, df: pd.DataFrame, csv_path):
        """
        Extract paper references from table using Ollama.
        """
        import ollama
        
        client = ollama.Client()
        response = client.chat(
            model="qwen2.5vl:3b", #"tinyllama:1.1b",
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are an expert at extracting references from OCR tables. 
                    Do not write code. 
                    Do not explain ANYTHING. 
                    Always respond only with the requested list in plain text, one item per line. 
                    If there are no valid references, respond with <no references>.
                    """
                            },
                            {
                                "role": "user",
                                "content": f"""
                                Here is an OCR-extracted table from a review article (in CSV format):

                                {df.to_csv(index=False)}

                                Extract author surnames as short paper references. 
                                Output only the references, one per line, preserving the original order as much as possible.
                                Only return the surname of the first author. 
                                Return all the references you can find, but each surname should appear only once.

                                Now output the list of surnames as plain text, one per line:                                        
                                """
                },
            ],
            options={"temperature": 0}
        )
        
        content = response['message']['content']
        if "no references" not in content.lower():
            return [line.strip() for line in content.splitlines() if line.strip()]
        return []



    def run_extractors(self) -> List[Dict[str, Any]]:
        """Run all enabled extractors and return a unified list of table dicts."""
        tables: List[Dict[str, Any]] = []
        # In practice, you might choose to run only a subset of extractors depending on PDF

        tables.extend(self.extract_gmft_tables())
        tables.extend(self.extract_img2table_tables())
        # tables.extend(self.extract_tables_with_vision_llm())

        return tables

    def run(self, first_arg: Optional[Any] = None, extracted_tables: Optional[List[Dict[str, Any]]] = None, matcher: Optional[ReferenceMatcher] = None) -> Dict[str, Any]:
        """Run extraction and optionally matching."""
        # If caller passed actual extracted tables as first_arg (legacy positional), detect it
        if extracted_tables is None and first_arg is not None:
            # heuristics: if first element is dict with 'data' key -> it's tables
            if isinstance(first_arg, (list, tuple)) and len(first_arg) > 0 and isinstance(first_arg[0], dict) and 'data' in first_arg[0]:
                extracted_tables = list(first_arg)
                references = None
            else:
                # otherwise assume first_arg is references (legacy behaviour)
                references = list(first_arg) if isinstance(first_arg, (list, tuple)) else [first_arg]
        else:
            references = None

        # Acquire tables
        if extracted_tables is None:
            tables = self.run_extractors()
        else:
            tables = extracted_tables

        # Build candidates
        candidates = self.build_candidates_from_tables(tables)
        logger.info(f"Built {len(candidates)} candidates from {len(tables)} tables")
        logger.info(f"Candidates: {candidates}")

        result = {
            'paper_id': self.paper_id,
            'tables': tables,
            'candidates': candidates,
            'matches': [],
        }

        # If references were provided (legacy flow), and no matcher provided, create default
        if references is not None:
            if matcher is None:
                matcher = ReferenceMatcher()
            try:
                # Prepare matcher with references
                matcher.prepare(references)
                # Extract candidate strings from candidate dicts
                candidate_strings = []
                for c in candidates:
                    if isinstance(c, dict) and 'candidate' in c:
                        candidate_strings.append(c['candidate'])
                    else:
                        candidate_strings.append(str(c) if c is not None else "")
                
                matches = matcher.match_candidates(candidate_strings, references)
                result['matches'] = matches
            except Exception as e:
                logger.exception('Matching failed')
                result['matches'] = []

        # Save summary for traceability
        try:
            summary_path = Path(self.tables_dir) / f"{self.paper_id}_extraction_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f'Failed to write extraction summary: {e}')

        return result

    def _safe_get_reference_field(self, ref_obj: Any, field: str, default=None):
        """Safely extract a field from a reference object."""
        if ref_obj is None:
            return default
        
        # Try as object attribute first
        if hasattr(ref_obj, field):
            value = getattr(ref_obj, field, default)
            return str(value) if value is not None else default
        
        # Try as dict key
        if isinstance(ref_obj, dict):
            value = ref_obj.get(field, default)
            return str(value) if value is not None else default
        
        return default

    def _format_crossref_item(self, crossref_item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Format crossref item into a clean, readable structure."""
        if not crossref_item:
            return {}
        
        # Extract title (usually a list)
        titles = crossref_item.get("title", [])
        title = titles[0] if titles else ""
        
        # Extract authors
        authors = crossref_item.get("author", [])
        author_names = []
        for author in authors:
            if isinstance(author, dict):
                given = author.get("given", "")
                family = author.get("family", "")
                if given and family:
                    author_names.append(f"{given} {family}")
                elif family:
                    author_names.append(family)
                elif given:
                    author_names.append(given)
        
        # Extract journal/container
        containers = crossref_item.get("container-title", [])
        container = containers[0] if containers else ""
        
        # Extract year from issued date
        issued = crossref_item.get("issued", {})
        year = ""
        if isinstance(issued, dict):
            date_parts = issued.get("date-parts", [])
            if date_parts and len(date_parts) > 0 and len(date_parts[0]) > 0:
                year = str(date_parts[0][0])
        
        return {
            "title": title,
            "authors": author_names,
            "journal": container,
            "year": year,
            "doi": crossref_item.get("DOI", ""),
            "type": crossref_item.get("type", "")
        }

    def run_with_matcher(self, references: Sequence[Union[Dict, Any]], matcher: Optional[ReferenceMatcher] = None,
                        extracted_tables: Optional[List[Dict[str, Any]]] = None,
                        crossref_rows: int = 5,
                        crossref_title_threshold: float = 0.85,
                        crossref_email: str = "you@example.com") -> Dict[str, Any]:
        """
        Convenience helper: run extraction (or use provided tables) then match against references.
        Returns a flattened, easy-to-parse structure for comparison.
        """
        if matcher is None:
            matcher = ReferenceMatcher()

        if extracted_tables is None:
            tables = self.run_extractors()
        else:
            tables = extracted_tables

        candidates = self.build_candidates_from_tables(tables)

        try:
            # Prepare matcher with references
            matcher.prepare(list(references))
            # Match candidates against references (default top_n=1)
            raw_matches = matcher.match_candidates(candidates, list(references), top_n=1)
        except Exception as e:
            logger.exception('Matching failed in run_with_matcher')
            raw_matches = []

        # Build index map for references for quick lookup
        ref_map = {}
        for i, r in enumerate(references):
            # Try to get a proper index from the reference object
            ref_index = self._safe_get_reference_field(r, "index")
            if ref_index is None:
                ref_index = i  # Fallback to position in list
            ref_map[ref_index] = r

        # Create flattened comparison structure
        comparison_results = []
        
        for i, (cand_obj, candidate_matches) in enumerate(zip(candidates, raw_matches)):
            # Extract candidate string and metadata
            candidate_text = cand_obj.get('candidate') if isinstance(cand_obj, dict) else str(cand_obj)
            
            # Initialize result entry
            result_entry = {
                # Candidate information
                "candidate_id": i,
                "candidate_text": candidate_text,
                "candidate_page": cand_obj.get('page') if isinstance(cand_obj, dict) else None,
                "candidate_table_index": cand_obj.get('table_index') if isinstance(cand_obj, dict) else None,
                "candidate_row_index": cand_obj.get('row_index') if isinstance(cand_obj, dict) else None,
                "candidate_source": cand_obj.get('source') if isinstance(cand_obj, dict) else None,
                
                # Match information (will be filled if match exists)
                "has_match": False,
                "match_score": 0.0,
                "match_reason": "no_match",
                
                # Reference information (will be filled if match exists)
                "reference_index": None,
                "reference_title": None,
                "reference_authors": None,
                "reference_journal": None,
                "reference_year": None,
                "reference_doi": None,
                "reference_raw": None,
                
                # Crossref information
                "crossref_doi": None,
                "crossref_score": 0.0,
                "crossref_title": None,
                "crossref_authors": None,
                "crossref_journal": None,
                "crossref_year": None,
                "crossref_error": None
            }
            
            # Process match if it exists
            if candidate_matches and len(candidate_matches) > 0:
                top_match = candidate_matches[0]
                matched_index = top_match.get("matched_index")
                
                if matched_index is not None:
                    result_entry["has_match"] = True
                    result_entry["match_score"] = top_match.get("score", 0.0)
                    result_entry["match_reason"] = top_match.get("reason", "unknown")
                    result_entry["reference_index"] = matched_index
                    
                    # Find the reference object
                    ref_obj = ref_map.get(matched_index)
                    if ref_obj is None:
                        # Fallback: try to find by scanning all references
                        for r in references:
                            if self._safe_get_reference_field(r, "index") == matched_index:
                                ref_obj = r
                                break
                    
                    # Extract reference fields
                    if ref_obj is not None:
                        result_entry["reference_title"] = self._safe_get_reference_field(ref_obj, "title")
                        result_entry["reference_authors"] = self._safe_get_reference_field(ref_obj, "authors")
                        result_entry["reference_journal"] = self._safe_get_reference_field(ref_obj, "journal")
                        result_entry["reference_year"] = self._safe_get_reference_field(ref_obj, "year")
                        result_entry["reference_doi"] = self._safe_get_reference_field(ref_obj, "doi")
                        result_entry["reference_raw"] = self._safe_get_reference_field(ref_obj, "raw")
                    
                    # Perform Crossref lookup
                    existing_doi = result_entry["reference_doi"]
                    if existing_doi:
                        # Reference already has DOI
                        result_entry["crossref_doi"] = existing_doi
                        result_entry["crossref_score"] = 1.0
                    else:
                        # Try to get DOI from Crossref
                        try:
                            title_for_search = result_entry["reference_title"] or candidate_text
                            crossref_result = fetch_doi_from_crossref(
                                title=title_for_search,
                                authors=result_entry["reference_authors"],
                                journal=result_entry["reference_journal"],
                                year=result_entry["reference_year"],
                                rows=crossref_rows,
                                min_crossref_title_score=crossref_title_threshold,
                                user_agent_email=crossref_email,
                            )
                            
                            result_entry["crossref_doi"] = crossref_result.get("doi")
                            result_entry["crossref_score"] = crossref_result.get("score", 0.0)
                            result_entry["crossref_error"] = crossref_result.get("error")
                            
                            # Extract formatted crossref item information
                            crossref_item = crossref_result.get("item")
                            if crossref_item:
                                formatted_item = self._format_crossref_item(crossref_item)
                                result_entry["crossref_title"] = formatted_item.get("title")
                                result_entry["crossref_authors"] = ", ".join(formatted_item.get("authors", []))
                                result_entry["crossref_journal"] = formatted_item.get("journal")
                                result_entry["crossref_year"] = formatted_item.get("year")
                                
                        except Exception as e:
                            logger.warning(f"Crossref lookup failed for candidate {i}: {e}")
                            result_entry["crossref_error"] = str(e)
            
            comparison_results.append(result_entry)

        # Create output structure
        output = {
            "paper_id": self.paper_id,
            "summary": {
                "total_candidates": len(candidates),
                "candidates_with_matches": sum(1 for r in comparison_results if r["has_match"]),
                "crossref_dois_found": sum(1 for r in comparison_results if r["crossref_doi"]),
                "average_match_score": sum(r["match_score"] for r in comparison_results) / len(comparison_results) if comparison_results else 0.0
            },
            "tables": tables,
            "candidates": candidates,
            "raw_matches": raw_matches,
            "comparison_results": comparison_results
        }

        # Save results
        try:
            summary_path = Path(self.tables_dir) / f"{self.paper_id}_match_comparison.json"
            with open(summary_path, 'w', encoding='utf-8') as fh:
                json.dump(output, fh, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Match comparison results saved to: {summary_path}")
        except Exception as e:
            logger.warning(f'Failed to write match comparison results: {e}')

        # create another file with just the comparison results
        try:
            comparison_path = Path(self.tables_dir) / f"{self.paper_id}_comparison_results.csv"
            import pandas as pd
            df = pd.DataFrame(comparison_results)
            df.to_csv(comparison_path, index=False)
            logger.info(f"Comparison results CSV saved to: {comparison_path}")
        except Exception as e:
            logger.warning(f'Failed to write comparison results CSV: {e}')  

        # create another file with just the dois of the matched references
        try:
            dois_path = Path(self.tables_dir) / f"{self.paper_id}_matched_dois.txt"
            matched_dois = [r["crossref_doi"] for r in comparison_results if r["crossref_doi"]]
            with open(dois_path, 'w', encoding='utf-8') as fh:
                for doi in matched_dois:
                    fh.write(doi + "\n")
            output["matched_dois"] = matched_dois
            logger.info(f"Matched DOIs saved to: {dois_path}")
        except Exception as e:
            logger.warning(f'Failed to write matched DOIs: {e}')    

        return output

