"""Public package surface for EpiScope."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("epi-scope")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

from episcope.db import mongo_academic_db, in_memory_academic_db
from episcope import rag
from episcope.rag import generation, indexing, ingestion, retrieval
from episcope.services import EpiScopeRuntime, RuntimeConfig
from episcope import workflows
from episcope import utils


__all__ = [
    "EpiScopeRuntime",
    "RuntimeConfig",
    "mongo_academic_db",
    "in_memory_academic_db",
    "rag",
    "workflows",
    "utils",
    "generation",
    "indexing",
    "ingestion",
    "retrieval",
    "__version__",
]
