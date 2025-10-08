"""Academic database manager for extracted documents.

This module defines a thin abstraction over a persistent store
for the outputs of the PrecisionMiner extraction pipeline.  Each
extracted record corresponds to a particular paper and JSON
data type (``sections``, ``metadata`` or ``references``) under a
named extraction strategy.  Records are uniquely identified by
the tuple (paper_id, data_type, strategy_name).  By default
documents are stored in a MongoDB collection, but when the
``pymongo`` library or a running MongoDB instance is not
available the manager transparently falls back to an in‑memory
dictionary.  The fallback implementation is sufficient for unit
testing and development without requiring any external services.

The public API consists of simple CRUD operations:

* :meth:`insert` writes a new record.  If a record with the same
  compound key already exists the insertion is ignored.
* :meth:`retrieve` looks up a record by its compound key and
  returns the stored JSON object or ``None`` if not present.
* :meth:`delete_by_strategy` deletes all records associated with
  a given strategy name.  This is useful for cleaning up test
  runs or resetting the database between experiments.

Clients should construct a single instance of
``AcademicDBManager`` and re‑use it across operations.  When
initialising the manager a MongoDB URI may be supplied along
with an optional flag ``use_in_memory`` to force the fallback
store regardless of library availability.  The default
collection is ``ExtractedDocuments`` in the ``AcademicCorpus``
database.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

try:
    # Use lazy import to avoid hard dependency on pymongo when not
    # available in the environment.  The client will be created on
    # demand if present.  Consumers may set ``use_in_memory=True``
    # explicitly to skip any attempt at connecting to MongoDB.
    import pymongo  # type: ignore
    from pymongo import MongoClient
    _PYMONGO_AVAILABLE = True
except Exception:  # pragma: no cover - absence of pymongo is expected in many test environments
    pymongo = None  # type: ignore
    MongoClient = None  # type: ignore
    _PYMONGO_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from ..parse.blueprints.data_blueprints import PaperMetadata, Reference, StructuredSection

MONGO_URI = os.environ.get("mongo_uri", "mongodb://localhost:27017")
class AcademicDBManager:
    """Manager class for storing and retrieving extraction outputs.

    Each document in the database has the following fields:

    ``paper_id`` (str)
        Unique identifier for the source paper.  This is typically
        derived from the file name without extension or a DOI.

    ``data_type`` (str)
        Specifies which of the three JSON files the record holds.
        Must be one of ``"sections"``, ``"metadata"`` or
        ``"references"``.

    ``strategy_name`` (str)
        Name of the extraction strategy (e.g., ``"Strategy_V1_GROBID_Standard"``).
        This acts as a namespace to isolate test runs.

    ``content`` (dict)
        Parsed JSON data corresponding to the given data type.

    The manager enforces a unique compound key on
    (paper_id, data_type, strategy_name) to prevent accidental
    overwrites.  When running against MongoDB this is achieved via
    a unique index; in the fallback store uniqueness is enforced
    in Python.
    """

    def __init__(
        self,
        uri: str = None, # "mongodb://localhost:27017"
        db_name: str = "AcademicCorpus",
        use_in_memory: bool = False,
        backup_file: Optional[str] = None,
    ) -> None:
        """Initialise a new database manager.

        Args:
            uri: MongoDB URI to connect to when pymongo is available.
            db_name: Name of the database to use in MongoDB.
            collection_name: Name of the collection storing documents.
            use_in_memory: Force the use of the in‑memory fallback store.
            backup_file: Path to a JSON file on disk used to persist
                the fallback store.  When provided, data are loaded
                from this file at construction time and flushed back
                to disk on insert and delete operations.  If not
                specified the fallback store is kept purely in memory.
        """
        self.use_in_memory = use_in_memory or not _PYMONGO_AVAILABLE
        self.uri = uri if uri is not None else MONGO_URI
        self.db_name = db_name
        # Fallback store: mapping from (paper_id, data_type, strategy_name) to
        # content dict
        self._store: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._backup_file = backup_file
        if self._backup_file and os.path.exists(self._backup_file):
            try:
                with open(self._backup_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                # Keys are stored as joined strings; restore as tuples
                for key, value in data.items():
                    paper_id, data_type, strategy = key.split("||")
                    self._store[(paper_id, data_type, strategy)] = value
            except Exception as exc:
                logger.warning(f"Failed to load backup file {self._backup_file}: {exc}")
        if not self.use_in_memory:
            try:
                self._client = MongoClient(self.uri)
                self._db = self._client[self.db_name]
            except Exception as exc:
                logger.warning(
                    f"Failed to connect to MongoDB at {self.uri}: {exc}; falling back to in‑memory store"
                )
                self.use_in_memory = True
        if self.use_in_memory:
            logger.info(
                "Using in‑memory store for AcademicDBManager.  Data will not persist across runs unless a backup file is specified."
            )

    # ------------------------------------------------------------------
    # Public CRUD methods
    # ------------------------------------------------------------------
    def insert(
        self,
        paper_id: str,
        data_type: str,
        strategy_name: str,
        content: Dict[str, Any],
    ) -> None:
        """Insert a new extraction record into the store.

        If a document with the same compound key already exists the
        operation is ignored.  Callers should choose a unique
        ``strategy_name`` per extraction run to avoid collisions.

        Args:
            paper_id: Unique identifier for the source paper.
            data_type: One of ``"sections"``, ``"metadata"`` or ``"references"``.
            strategy_name: Name of the extraction strategy.
            content: Parsed JSON object representing the extracted
                content.  For sections and references this is
                typically a list of dictionaries; for metadata a
                single dictionary.
        """
        key = (paper_id, data_type, strategy_name)
        if self.use_in_memory:
            if key in self._store:
                logger.debug(f"Duplicate insertion ignored for {key}")
                return
            self._store[key] = content
            # Persist backup if configured
            if self._backup_file:
                self._flush_backup()
            return
        # MongoDB branch
        collection = self._get_collection(strategy_name)
        doc = {
            "paper_id": paper_id,
            "data_type": data_type,
            "content": content,
        }
        try:
            collection.insert_one(doc)
        except pymongo.errors.DuplicateKeyError:  # type: ignore[no-redef]
            logger.debug(f"Duplicate insertion ignored for {key}")



    def __deserialize_content(self, data_type: str, content: Any):
        """Helper to deserialize content based on data type."""

        if data_type == "sections":
            return [StructuredSection.from_dict(s) for s in content]
        elif data_type == "metadata":
            return PaperMetadata.from_dict(content)
        elif data_type == "references":
            return [Reference.from_dict(r) for r in content]
        else:
            return None

    def retrieve(
        self, paper_id: str, data_type: str, strategy_name: str
    ) -> Optional[Any]:
        """Retrieve a single extraction record.

        Args:
            paper_id: Identifier of the paper to look up.
            data_type: Data type of the record (``sections``, ``metadata`` or ``references``).
            strategy_name: Extraction strategy name under which the record was stored.

        Returns:
            The stored data deserialized into the appropriate object,
            or ``None`` if not found.
        """
        key = (paper_id, data_type, strategy_name)
        logger.debug(f"Attempting to retrieve document for key: {key}")

        if self.use_in_memory:
            result = self.store.get(key)
            if result:
                logger.debug(f"Found document in-memory for key: {key}")
            else:
                logger.debug(f"No document found in-memory for key: {key}")
            return result

        # MongoDB branch
        collection = self._get_collection(strategy_name)
        query = {
            "paper_id": paper_id,
            "data_type": data_type,
        }
        logger.debug(f"Executing MongoDB find_one with query: {query} in collection: {strategy_name}")
        doc = collection.find_one(
            query,
            projection={"_id": False, "content": True},
        )

        if not doc:
            logger.debug(f"No document found for {key}")
        return self.__deserialize_content(data_type, doc["content"]) if doc else None


    def delete_by_strategy(self, strategy_name: str) -> int:
        """Delete all records associated with a given strategy.

        Args:
            strategy_name: The strategy namespace to remove.

        Returns:
            The number of documents removed.
        """
        removed_count = 0
        if self.use_in_memory:
            keys_to_delete = [k for k in self._store if k[2] == strategy_name]
            for k in keys_to_delete:
                del self._store[k]
                removed_count += 1
            if self._backup_file:
                self._flush_backup()
            return removed_count
        # MongoDB branch
        collection = self._get_collection(strategy_name)
        count = collection.count_documents({})
        collection.drop()
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_collection(self, strategy_name: str):
        """Get MongoDB collection for a given strategy, creating index if new."""
        if self.use_in_memory:
            return None
        collection = self._db[strategy_name]
        # Create index if it's a new collection
        if strategy_name not in self._db.list_collection_names():
             collection.create_index(
                [
                    ("paper_id", pymongo.ASCENDING),
                    ("data_type", pymongo.ASCENDING),
                ],
                unique=True,
            )
        return collection

