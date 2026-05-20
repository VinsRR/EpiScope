"""Public package surface for EpiScope."""

__version__ = "0.1.0"

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
