from __future__ import annotations

import re
from typing import Iterable

from rapidfuzz import fuzz


def normalize_text(value: str) -> str:
    value = value or ""
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def exact_match_score(response: str | None, reference: str) -> float:
    return float(normalize_text(response or "") == normalize_text(reference))


def _greedy_match_count(
    retrieved_contexts: Iterable[str],
    reference_contexts: Iterable[str],
    *,
    threshold: float,
) -> int:
    remaining = [normalize_text(item) for item in reference_contexts if normalize_text(item)]
    hits = 0
    for retrieved in retrieved_contexts:
        norm_retrieved = normalize_text(retrieved)
        if not norm_retrieved or not remaining:
            continue
        best_idx = None
        best_score = -1.0
        for idx, candidate in enumerate(remaining):
            score = float(fuzz.token_set_ratio(norm_retrieved, candidate))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= threshold:
            hits += 1
            remaining.pop(best_idx)
    return hits


def reference_context_prf(
    retrieved_contexts: Iterable[str],
    reference_contexts: Iterable[str],
    *,
    threshold: float = 90.0,
) -> tuple[float | None, float | None, float | None]:
    retrieved = [item for item in retrieved_contexts if normalize_text(item)]
    reference = [item for item in reference_contexts if normalize_text(item)]

    if not reference:
        return None, None, None

    hits = _greedy_match_count(retrieved, reference, threshold=threshold)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(reference) if reference else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
