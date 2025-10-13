"""
retrieve/utils.py

Utility functions for retrieval modules.

Includes helper functions to detect pathogen keywords in file paths,
validate token budgets for language model prompts and a simplified
HYDE class that generates hypothetical documents using an LLM.  Most
functions in this module are adapted from the original ``rag_tool``
project and have been lightly refactored.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List, Optional

import ollama

# List of pathogen keywords recognised in file paths
LIST_PATHOGENS: List[str] = [
    "bunyavirales",
    "covid19",
    "evd_mvd",
    "influenza",
    "mpox",
    "nipah",
    "sars_mers",
    "zika",
]


def find_pathogen_keyword(text: str) -> Optional[str]:
    """Return the first pathogen keyword found in ``text``, or ``None``.

    The lookup is case‑insensitive and simply checks for substring
    matches.  If no keywords are present, ``None`` is returned.
    """
    lower_text = text.lower()
    for pathogen in LIST_PATHOGENS:
        if pathogen in lower_text:
            return pathogen
    return None


# Token budget estimation utilities
AVG_CHARS_PER_TOKEN = 4.0  # Approximate average characters per token
TOKEN_BUFFER = 1.10       # 10% safety margin on context window


def get_context_length(model_name: str) -> int:
    """Query the Ollama API for the model's declared context length."""
    info = dict(ollama.show(model_name)).get("modelinfo", {})
    for k, v in info.items():
        if "context_length" in k:
            return int(v)
    raise ValueError(f"No context_length found in Ollama info for '{model_name}'")


def estimate_token_count(messages: Iterable[dict[str, str]]) -> int:
    """Estimate token usage heuristically based on character counts."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return int(total_chars / AVG_CHARS_PER_TOKEN)


def validate_token_budget(model_name: str, messages: Iterable[dict[str, str]]) -> None:
    """Raise if estimated token usage would exceed the model's context window."""
    max_tokens = get_context_length(model_name)
    threshold = int(max_tokens * TOKEN_BUFFER)
    estimated = estimate_token_count(messages)
    if estimated > threshold:
        raise ValueError(
            f"Estimated token usage ({estimated}) exceeds safe threshold ({threshold}) for model '{model_name}'."
        )


# ---------------------------------------------------------------------------
# HYDE wrapper
#
# Historically, a simple HYDE implementation lived in this module to
# synthesise hypothetical documents for the Explorer pipeline.  The full
# featured HYDE is now implemented in ``episcope.core.hyde``.  To
# maintain backwards compatibility, we provide a thin wrapper that
# delegates to the unified implementation.  New code should import
# ``HYDE`` from ``episcope.core.hyde`` directly.
from .hyde import HYDE as _CoreHYDE


class HYDE(_CoreHYDE):
    """Backward‑compatible HYDE wrapper for retrieval utilities.

    This subclass does not add any new functionality; it merely adapts
    the unified HYDE implementation to accept the simple signature
    previously used in this module.  The ``generate`` method simply
    forwards to the parent class.
    """

    def __init__(self, model_name: str = "tinyllama:1.1b") -> None:
        super().__init__(model_name=model_name)

    # No override of generate() needed – inherited implementation


