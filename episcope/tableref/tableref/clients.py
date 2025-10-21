import requests
import difflib
from typing import Dict, Any, Optional, Union, Sequence

# try fast fuzzy engine, fallback to difflib
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    HAS_RAPIDFUZZ = False

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
