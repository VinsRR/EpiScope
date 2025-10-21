"""
Configuration modules for EpiScope.

This package exposes configuration objects for the various indexing and retrieval
strategies supported by EpiScope.  Each configuration file defines sensible
defaults for embedding models, index parameters, prompts and connection
settings.  See individual modules for details.
"""

# Explicitly import common configs so they are discoverable via
# `from episcope.configs import text_only, colpali, graph`.
from .parse_configs import *  # noqa: F401,F403

__version__ = "0.1.0"

from episcope.db import mongo_academic_db, in_memory_academic_db
from episcope import rag 
from episcope.rag import generation,indexing,ingestion, retrieval
from episcope import pipelines
from episcope import utils



__all__ = [
    "mongo_academic_db",
    "in_memory_academic_db",
    "rag",
    "pipelines",
    "utils",
    "generation",
    "indexing",
    "ingestion",
    "retrieval",
    "__version__",]
