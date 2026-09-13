from __future__ import annotations

import logging
from typing import Optional

from .academic_db import AcademicDB
from .in_memory_academic_db import InMemoryAcademicDB
from .mongo_academic_db import MongoAcademicDB, _PYMONGO_AVAILABLE

logger = logging.getLogger(__name__)


def get_academic_db(
    uri: Optional[str] = None,
    db_name: str = "AcademicCorpus",
    use_in_memory: bool = False,
    backup_file: Optional[str] = None,
) -> AcademicDB:
    """Factory function to get an academic database instance.

    This function provides a single point of entry for creating a database
    manager. It will attempt to connect to MongoDB by default, but will
    transparently fall back to an in-memory implementation if pymongo is
    not available or a connection cannot be established.

    Args:
        uri: MongoDB URI to connect to when pymongo is available.
        db_name: Name of the database to use in MongoDB.
        use_in_memory: Force the use of the in-memory fallback store.
        backup_file: Path to a JSON file on disk used to persist
            the fallback store. This is only used for the in-memory
            database.

    Returns:
        An instance of a class that implements the AcademicDB interface.
    """
    if use_in_memory or not _PYMONGO_AVAILABLE:
        if not use_in_memory and not _PYMONGO_AVAILABLE:
            logger.info("pymongo not available, falling back to in-memory store.")
        logger.info(
            "Using in-memory store for AcademicDBManager. Data will not persist across runs unless a backup file is specified."
        )
        return InMemoryAcademicDB(backup_file=backup_file)

    try:
        return MongoAcademicDB(uri=uri, db_name=db_name)
    except Exception as e:
        logger.warning(
            f"Failed to connect to MongoDB: {e}. Falling back to in-memory store."
        )
        return InMemoryAcademicDB(backup_file=backup_file)
