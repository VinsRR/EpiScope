from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, List

from .academic_db import AcademicDB

logger = logging.getLogger(__name__)


class InMemoryAcademicDB(AcademicDB):
    """In-memory implementation of AcademicDB.

    Sufficient for unit testing and development without requiring
    any external services. Can be backed by a JSON file for persistence.
    """

    def __init__(self, backup_file: Optional[str] = "backup.json") -> None:
        """Initialise a new in-memory database.

        Args:
            backup_file: Path to a JSON file on disk used to persist
                the store. When provided, data are loaded from this
                file at construction time and flushed back to disk on
                insert and delete operations. If not specified the
                store is kept purely in memory.
        """
        self._store: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._backup_file = backup_file
        if self._backup_file and os.path.exists(self._backup_file):
            logger.info(f"Loading backup from {self._backup_file}")
            self._load_backup()

    def _load_backup(self) -> None:
        """Load data from the backup file."""
        try:
            with open(self._backup_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Keys are stored as joined strings; restore as tuples
            for key, value in data.items():
                paper_id, data_type, strategy = key.split("||")
                self._store[(paper_id, data_type, strategy)] = value
        except Exception as exc:
            logger.warning(f"Failed to load backup file {self._backup_file}: {exc}")

    def _flush_backup(self) -> None:
        """Write current data to the backup file."""
        if not self._backup_file:
            return
        # Keys are tuples, convert to strings for JSON
        data_to_persist = {"||".join(k): v for k, v in self._store.items()}
        try:
            with open(self._backup_file, "w", encoding="utf-8") as fh:
                json.dump(data_to_persist, fh, indent=2)
        except Exception as exc:
            logger.error(f"Failed to write backup to {self._backup_file}: {exc}")

    def insert(
        self,
        paper_id: str,
        data_type: str,
        strategy_name: str,
        content: Dict[str, Any],
    ) -> None:
        key = (paper_id, data_type, strategy_name)
        if key in self._store:
            logger.debug(f"Duplicate insertion ignored for {key}")
            return
        self._store[key] = content
        if self._backup_file:
            self._flush_backup()

    def retrieve(
        self, paper_id: str, data_type: str, strategy_name: str
    ) -> Optional[Any]:
        key = (paper_id, data_type, strategy_name)
        logger.debug(f"Attempting to retrieve document for key: {key}")
        content = self._store.get(key)
        if content:
            logger.debug(f"Found document in-memory for key: {key}")
            return self._deserialize_content(data_type, content)

        logger.debug(f"No document found in-memory for key: {key}")
        return None

    def delete_by_strategy(self, strategy_name: str) -> int:
        keys_to_delete = [k for k in self._store if k[2] == strategy_name]
        removed_count = len(keys_to_delete)
        for k in keys_to_delete:
            del self._store[k]

        if self._backup_file and removed_count > 0:
            self._flush_backup()
        return removed_count

    def list_papers(self, strategy_name: str) -> List[str]:
        """List all paper IDs for a given strategy."""
        return sorted(list(set(k[0] for k in self._store if k[2] == strategy_name)))
