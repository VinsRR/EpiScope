import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .utils import DOI_RE, YEAR_RE, default_normalize, simple_normalize

from .config import LateInteractionConfig, SimpleSurnameMatcherConfig

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
    import difflib
    HAS_RAPIDFUZZ = False


logger = logging.getLogger(__name__)


class LateInteractionMatcher:
    """
    Implements a ColBERT-style MaxSim late-interaction matcher.
    """

    def __init__(
        self,
        references_norm: List[Any],
        config: "LateInteractionConfig",
        normalize_fn = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch and transformers are required for LateInteractionMatcher")

        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for LateInteractionMatcher")

        self.refs = list(references_norm)
        self.config = config
        self.normalize = normalize_fn if normalize_fn is not None else default_normalize
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.ref_texts = []
        for r in self.refs:
            authors = getattr(r, "authors", []) or []
            first_author_raw = authors[0] if authors else ""
            if not isinstance(first_author_raw, str):
                logger.warning(f"First author is not a string, but {type(first_author_raw)}. Value: {first_author_raw}. Skipping.")
                first_author_raw = ""
            first_author = self.normalize(first_author_raw)

            parts = [
                self.normalize(r.title) if hasattr(r, "title") else "",
                first_author,
                self.normalize(r.year) if hasattr(r, "year") else "",
                self.normalize(r.journal) if hasattr(r, "journal") else "",
            ]
            txt = " . ".join([p for p in parts if p])
            if txt.strip():
                self.ref_texts.append(self.normalize(txt))
            else:
                self.ref_texts.append("")

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        try:
            self.ref_tfidf = self.vectorizer.fit_transform(self.ref_texts)
        except ValueError as e:
            logger.warning(f"TF-IDF vectorization failed: {e}. Using fallback.")
            self.ref_tfidf = None
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.transform_model_name)
            self.model = AutoModel.from_pretrained(self.config.transform_model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load transformer model {self.config.transform_model_name}: {e}")
            raise

        self.ref_token_ids: List[List[int]] = []
        self.ref_token_embs: List[np.ndarray] = []
        df_counts = {}

        with torch.no_grad():
            for text in self.ref_texts:
                if not text.strip():
                    self.ref_token_ids.append([])
                    self.ref_token_embs.append(np.zeros((1, 768)))
                    continue
                try:
                    enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.config.max_ref_tokens)
                    input_ids = enc["input_ids"].squeeze(0).to(self.device)
                    attention_mask = enc["attention_mask"].squeeze(0).to(self.device)
                    outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
                    last_hidden = outputs.last_hidden_state.squeeze(0)
                    token_embs = last_hidden.cpu()
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

        N = len(self.ref_texts)
        self.idf = {}
        for tid, df in df_counts.items():
            self.idf[tid] = math.log((N + self.config.idf_smoothing) / (df + self.config.idf_smoothing)) + 1.0

    def split_subcands(self, text: str) -> List[str]:
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

        r_embs = self.ref_token_embs[ref_idx]
        if r_embs.size == 0:
            return 0.0, []

        S = np.dot(q_embs, r_embs.T)
        max_per_token = S.max(axis=1)
        argmax_pos = S.argmax(axis=1).tolist()
        q_idf_weights = np.array([self.idf.get(qtid, 0.5) for qtid in q_ids])
        weighted = (max_per_token * q_idf_weights).sum()
        norm = q_idf_weights.sum() + 1e-12
        score = float(weighted / norm)
        evidence = [(i, int(pos), float(sim)) for i, (pos, sim) in enumerate(zip(argmax_pos, max_per_token.tolist()))]
        return score, evidence

    def match_candidate(self, candidate: str, top_n: int = 1, min_score: float = 0.0) -> List[Dict[str, Any]]:
        if not isinstance(candidate, str):
            candidate = str(candidate) if candidate is not None else ""

        if not candidate or len(candidate.strip()) == 0:
            return [{"matched_index": None, "score": 0.0, "reason": "too_short"}]

        if self.ref_tfidf is None:
            return [{"matched_index": None, "score": 0.0, "reason": "vectorizer_failed"}]

        m = DOI_RE.search(candidate)
        if m:
            doi = m.group(0).lower()
            hits = [
                {
                    "matched_index": getattr(r, "index", None),
                    "score": 1.0,
                    "reason": "doi_exact",
                    "matched_title": getattr(r, "title", None),
                    "matched_doi": getattr(r, "doi", None),
                    "matched_first_author": getattr(r, "first_author", None),
                }
                for r in self.refs
                if hasattr(r, "doi") and r.doi and doi in str(r.doi).lower()
            ]
            if hits:
                return sorted(hits, key=lambda x: -x["score"])[:top_n]

        subcands = self.split_subcands(candidate)
        if not subcands:
            return [{"matched_index": None, "score": 0.0, "reason": "no_subcandidates"}]

        all_out = {}
        sub_tfidfs = [self.vectorizer.transform([self.normalize(sub)]) for sub in subcands]

        for sub, sub_tfidf in zip(subcands, sub_tfidfs):
            try:
                sims = cosine_similarity(sub_tfidf, self.ref_tfidf).ravel()
                top_idxs = np.argsort(-sims)[: self.config.top_k_prefilter]
                q_ids, q_embs = self.compute_query_token_embs(sub)

                for ridx in top_idxs:
                    if ridx >= len(self.refs):
                        continue

                    sc, evidence = self.maxsim_score(q_ids, q_embs, int(ridx))
                    ref = self.refs[ridx]
                    if hasattr(ref, "year") and ref.year and str(ref.year) in sub:
                        sc = min(1.0, sc + 0.08)
                    fa = getattr(ref, "first_author_norm", None) or getattr(ref, "first_author", "")
                    if fa and isinstance(fa, str) and fa.split()[-1] in self.normalize(sub):
                        sc = min(1.0, sc + 0.08)

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

        return sorted(all_out.values(), key=lambda x: -x["score"])[:max(1, int(top_n))]


def _extract_surname(author_string: str) -> str:
    if not author_string:
        return ""
    a = author_string.strip()
    if "," in a:
        left = a.split(",")[0].strip()
        if len(left) > 1 and not re.fullmatch(r"[A-Z]\.?", left, flags=re.I):
            return left.split()[-1]
    return a.split()[-1] if a.split() else ""

def _fuzzy_score(a: str, b: str) -> int:
    if HAS_RAPIDFUZZ:
        return fuzz.ratio(a, b)
    else:
        return int(difflib.SequenceMatcher(None, a, b).ratio() * 100)

class SimpleSurnameMatcher:
    """
    Lightweight matcher that couples short surname-based candidates to a single reference.
    """

    def __init__(
        self,
        references_norm: List[Any],
        config: "SimpleSurnameMatcherConfig",
        normalize_fn=None,
    ):
        self.refs = list(references_norm)
        self.config = config
        self.normalize = normalize_fn if normalize_fn is not None else default_normalize

        self.surname_index = defaultdict(list)
        for i, r in enumerate(self.refs):
            authors = getattr(r, "authors", []) or []
            first_author = authors[0] if authors else ""
            if not isinstance(first_author, str):
                logger.warning(f"First author is not a string, but {type(first_author)}. Value: {first_author}. Skipping.")
                first_author = ""
            surname = _extract_surname(first_author)
            surname_norm = simple_normalize(surname, self.normalize)
            entry = {
                "index": getattr(r, "index", i),
                "ref": r,
                "first_author_norm": surname_norm,
            }
            if surname_norm:
                self.surname_index[surname_norm].append(entry)

        self.unique_surnames = list(self.surname_index.keys())
        self._norm_titles = {
            getattr(r, "index", i): simple_normalize((getattr(r, "title", "") or "") + " . " + (getattr(r, "raw_text", "") or ""), self.normalize)
            for i, r in enumerate(self.refs)
        }

    def match_candidate(self, candidate: str, **kwargs) -> List[Dict[str, Any]]:
        if not candidate or not str(candidate).strip():
            return [{"matched_index": None, "score": 0.0, "reason": "too_short"}]

        cand_norm = simple_normalize(candidate, self.normalize)
        cand_tokens = [t for t in re.split(r"\s+", cand_norm) if t]
        exact_candidates = [item for tok in cand_tokens if tok in self.surname_index for item in self.surname_index[tok]]

        if exact_candidates:
            if len(exact_candidates) == 1:
                e = exact_candidates[0]
                return [{"matched_index": e["index"], "score": 1.0, "reason": "exact_surname", **self._ref_info(e)}]

            best_score, best_entry = -1, None
            for e in exact_candidates:
                sc = _fuzzy_score(cand_norm, self._norm_titles.get(e["index"], ""))
                if sc > best_score:
                    best_score, best_entry = sc, e

            if best_entry and best_score >= self.config.fuzzy_title_threshold:
                return [{"matched_index": best_entry["index"], "score": float(best_score / 100.0), "reason": f"fuzzy_title_disambiguation({best_score})", **self._ref_info(best_entry)}]

            e = best_entry or exact_candidates[0]
            return [{"matched_index": e["index"], "score": float(max(0.0, best_score) / 100.0), "reason": "ambiguous_surname_best_guess", **self._ref_info(e)}]

        best_score, best_surname = -1, None
        for surname in self.unique_surnames:
            sc = _fuzzy_score(cand_norm, surname)
            if sc > best_score:
                best_score, best_surname = sc, surname

        if best_surname and best_score >= self.config.fuzzy_threshold:
            # pick the top candidate for that surname (if multiple, try year, then title fuzzy)
            cand_list = self.surname_index[best_surname]
            if len(cand_list) == 1:
                e = cand_list[0]
                return [{"matched_index": e["index"], "score": float(best_score / 100.0), "reason": f"fuzzy_surname({best_score})", **self._ref_info(e)}]


        return [{"matched_index": None, "score": 0.0, "reason": "no_match"}]

    def _ref_info(self, entry: Dict) -> Dict:
        ref = entry.get("ref", {})
        return {
            "matched_title": getattr(ref, "title", None),
            "matched_doi": getattr(ref, "doi", None),
            "matched_first_author": getattr(ref, "first_author", None),
        }


class ReferenceMatcher:
    """
    Wrapper that exposes a unified API for different matching backends.
    """

    def __init__(self, config: "MatcherConfig", normalize_fn=None):
        self.normalize = normalize_fn if normalize_fn is not None else default_normalize
        self.config = config
        self._matcher: Optional[Any] = None
        self._cached_refs_id = None

    def prepare(self, references_norm: List[Any]):
        refs_id = id(references_norm)
        if self._matcher is not None and refs_id == self._cached_refs_id:
            return

        if isinstance(self.config, SimpleSurnameMatcherConfig):
            matcher_class = SimpleSurnameMatcher
        elif isinstance(self.config, LateInteractionConfig):
            matcher_class = LateInteractionMatcher
        else:
            raise TypeError(f"Unsupported config type: {type(self.config)}")

        try:
            self._matcher = matcher_class(references_norm, config=self.config, normalize_fn=self.normalize)
            self._cached_refs_id = refs_id
        except Exception as e:
            logger.error(f"Failed to initialize {matcher_class.__name__}: {e}")
            self._matcher = None
            self._cached_refs_id = None

    def match_candidate(self, candidate: Union[str, Dict[str, Any]], references_norm: Optional[List[Any]] = None,
                        top_n: int = 1, min_score: float = 0.0) -> List[Dict[str, Any]]:

        candidate_str = candidate.get('candidate') if isinstance(candidate, dict) else str(candidate)

        if references_norm is not None and id(references_norm) != self._cached_refs_id:
            self.prepare(references_norm)

        if self._matcher is None:
            logger.error("Matcher not initialized. Call prepare(references_norm) first.")
            return [{"matched_index": None, "score": 0.0, "reason": "matcher_not_initialized"}]

        if not candidate_str or len(candidate_str.strip()) < self.config.min_candidate_length:
            return [{"matched_index": None, "score": 0.0, "reason": "too_short"}]

        try:
            # Note: top_n and min_score are passed at runtime, overriding config for flexibility
            return self._matcher.match_candidate(candidate_str, top_n=top_n, min_score=min_score)
        except Exception as e:
            logger.error(f"Matching failed for candidate: {candidate_str[:50]}... Error: {e}")
            return [{"matched_index": None, "score": 0.0, "reason": "matching_error"}]

    def match_candidates(self, candidates: List[Union[str, Dict[str, Any]]], references_norm: Optional[List[Any]] = None,
                         top_n: int = 1, min_score: float = 0.0) -> List[List[Dict[str, Any]]]:
        return [self.match_candidate(c, references_norm=references_norm, top_n=top_n, min_score=min_score) for c in candidates]
