from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

try:
    import pymongo
    from pymongo import MongoClient

    _PYMONGO_AVAILABLE = True
except ImportError:
    pymongo = None
    MongoClient = None
    _PYMONGO_AVAILABLE = False

from .academic_db import AcademicDB

logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("mongo_uri", "mongodb://localhost:27017")


class MongoAcademicDB(AcademicDB):
    """MongoDB implementation of AcademicDB."""

    def __init__(
        self,
        uri: str = None,
        db_name: str = "AcademicCorpus",
    ) -> None:
        """Initialise a new MongoDB-backed database manager.

        Args:
            uri: MongoDB URI to connect to.
            db_name: Name of the database to use in MongoDB.
        """
        if not _PYMONGO_AVAILABLE:
            raise ImportError(
                "pymongo is not installed. Please install it to use MongoAcademicDB."
            )

        self.uri = uri if uri is not None else MONGO_URI
        self.db_name = db_name
        try:
            self._client = MongoClient(self.uri)
            self._db = self._client[self.db_name]
        except Exception as exc:
            logger.error(f"Failed to connect to MongoDB at {self.uri}: {exc}")
            raise

    def _get_collection(self, strategy_name: str):
        """Get MongoDB collection for a given strategy, creating index if new."""
        collection = self._db[strategy_name]
        # Create index if it's a new collection
        if strategy_name not in self._db.list_collection_names():
            collection.create_index(
                [
                    ("doc_id", pymongo.ASCENDING),
                    ("data_type", pymongo.ASCENDING),
                ],
                unique=True,
            )
        return collection

    def insert(
        self,
        doc_id: str,
        data_type: str,
        strategy_name: str,
        content: Dict[str, Any],
    ) -> None:
        collection = self._get_collection(strategy_name)
        doc = {
            "doc_id": doc_id,
            "data_type": data_type,
            "content": content,
        }
        try:
            collection.insert_one(doc)
        except pymongo.errors.DuplicateKeyError:
            logger.info(
                f"Duplicate insertion ignored for {(doc_id, data_type, strategy_name)}"
            )

    def retrieve(
        self, doc_id: str, data_type: str, strategy_name: str
    ) -> Optional[Any]:
        key = (doc_id, data_type, strategy_name)
        logger.debug(f"Attempting to retrieve document for key: {key}")

        collection = self._get_collection(strategy_name)
        query = {
            "doc_id": doc_id,
            "data_type": data_type,
        }
        logger.debug(
            f"Executing MongoDB find_one with query: {query} in collection: {strategy_name}"
        )
        doc = collection.find_one(
            query,
            projection={"_id": False, "content": True},
        )

        if not doc:
            logger.debug(f"No document found for {key}")
            return None

        return self._deserialize_content(data_type, doc.get("content"))

    def delete_by_strategy(self, strategy_name: str) -> int:
        if strategy_name not in self._db.list_collection_names():
            return 0

        collection = self._db[strategy_name]
        count = collection.count_documents({})
        collection.drop()
        return count

    def list_docs(self, strategy_name: str) -> List[str]:
        """List all document IDs for a given strategy."""
        if strategy_name not in self._db.list_collection_names():
            return []
        
        collection = self._db[strategy_name]
        return collection.distinct("doc_id")
