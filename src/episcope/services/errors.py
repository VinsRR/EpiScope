"""Plain-language translation for error messages shown to end users.

Both the CLI (``episcope.py``) and the API (``api.py``) eventually surface
exceptions raised deep in ``rag``/``db``/``vectordb``/``workflows`` to a
person who may not know what Qdrant, Mongo, or an "embedding provider" are.
:func:`humanize_error` is the single place that translation happens, so the
two adapters give consistent messages instead of drifting apart.
"""

from __future__ import annotations

from pydantic import ValidationError

from episcope.workflows.registry import humanize_task_spec_error

# Ordered (substring, plain-language hint) pairs matched against str(exc).
# The hint is prepended to the original message, which is kept in full so
# the actionable detail (a URL, an env var name) is never lost.
_HINTS: list[tuple[str, str]] = [
    ("MONGO_URI", "No paper-metadata database is configured."),
    ("No Mongo URI configured", "No paper-metadata database is configured."),
    (
        "Could not infer an embedding provider",
        "EpiScope doesn't recognize this embedding model name.",
    ),
    (
        "Could not connect to Qdrant",
        "The vector search database isn't reachable.",
    ),
    (
        "is not in named-vector mode",
        "This vector collection was built with an older, incompatible format.",
    ),
    (
        "was not found at",
        "This corpus hasn't been indexed yet, or the collection name doesn't match.",
    ),
]


def humanize_error(exc: Exception) -> str:
    """Translate ``exc`` into a plain-language message for end users."""
    if isinstance(exc, ValidationError):
        return humanize_task_spec_error(exc)
    message = str(exc)
    for substring, hint in _HINTS:
        if substring in message:
            return f"{hint}\n{message}"
    return message
