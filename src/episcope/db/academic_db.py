from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from ..schemas import (
    PaperMetadata,
    Reference,
    StructuredSection,
)


class AcademicDB(ABC):
    """Abstract base class for academic database managers."""

    @abstractmethod
    def insert(
        self,
        doc_id: str,
        data_type: str,
        strategy_name: str,
        content: Any,
    ) -> None:
        """Insert a new extraction record into the store."""
        pass

    @abstractmethod
    def retrieve(
        self, doc_id: str, data_type: str, strategy_name: str
    ) -> Optional[Any]:
        """Retrieve a single extraction record."""
        pass

    def get_paper_metadata(self, doc_id: str, strategy_name: str):
        """Retrieve metadata for a given paper ID."""
        return self.retrieve(doc_id, "metadata", strategy_name)

    @abstractmethod
    def delete_by_strategy(self, strategy_name: str) -> int:
        """Delete all records associated with a given strategy."""
        pass

    @abstractmethod
    def list_docs(self, strategy_name: str) -> List[str]:
        """List all document IDs for a given strategy."""
        pass

    def delete_doc(self, doc_id: str, strategy_name: str) -> int:
        """Delete every stored data type for one paper.

        Backends that support document replacement should override this method.
        It is intentionally non-abstract so third-party read-only backends remain
        source compatible.
        """
        raise NotImplementedError("This academic database cannot delete one paper.")

    def _deserialize_content(self, data_type: str, content: Any) -> Optional[Any]:
        """Helper to deserialize content based on data type."""
        if content is None:
            return None

        if data_type == "sections":
            return [StructuredSection.from_dict(s) for s in content]
        elif data_type == "metadata":
            return PaperMetadata.from_dict(content)
        elif data_type == "references":
            return [Reference.from_dict(r) for r in content]
        else:
            return content
