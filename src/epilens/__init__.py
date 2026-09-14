"""Public package surface for EpiLens."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("epilens")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

from epilens.db import mongo_academic_db, in_memory_academic_db
from epilens import rag
from epilens.rag import generation, indexing, ingestion, retrieval
from epilens.services import EpiLensRuntime, RuntimeConfig
from epilens import workflows
from epilens.workflows import PaperClassifier, PrecisionMiner
from epilens import utils


__all__ = [
    "EpiLensRuntime",
    "RuntimeConfig",
    "PaperClassifier",
    "PrecisionMiner",
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
