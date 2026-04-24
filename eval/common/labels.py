from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any, FrozenSet

import pandas as pd

_CURLY_TO_STRAIGHT = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)


def canonicalize_text(value: str) -> str:
    """Normalize Unicode and whitespace for robust label comparisons."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.translate(_CURLY_TO_STRAIGHT)
    text = text.strip().upper()
    text = re.sub(r"\s+", " ", text)
    return text


def label_set_key(labels: FrozenSet[str]) -> str:
    """Stable string key for a set of labels."""
    return "|".join(sorted(labels))


def parse_label_set(value: Any) -> FrozenSet[str]:
    """Parse one label payload into a canonical frozenset of labels."""
    if value is None:
        return frozenset()
    if pd.isna(value):
        return frozenset()

    if isinstance(value, (list, tuple, set, frozenset)):
        items = [canonicalize_text(str(item)) for item in value]
        return frozenset(item for item in items if item)

    text = canonicalize_text(str(value))
    if not text:
        return frozenset()

    if (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
        or (text.startswith("{") and text.endswith("}"))
    ):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set, frozenset)):
            items = [canonicalize_text(str(item)) for item in parsed]
            return frozenset(item for item in items if item)
        if isinstance(parsed, str):
            parsed_text = canonicalize_text(parsed)
            return frozenset([parsed_text]) if parsed_text else frozenset()

    stripped = re.sub(r"^\[|\]$", "", text).strip()
    if "," in stripped:
        parts = [canonicalize_text(part) for part in stripped.split(",")]
        return frozenset(part for part in parts if part)

    return frozenset([stripped]) if stripped else frozenset()
