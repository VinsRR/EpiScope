from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BatchRAGPipeline(ABC):
    """Abstract base class for batch RAG pipelines."""

    @abstractmethod
    def process_item(self, item: Any) -> Any:
        """Process a single item in the batch."""
        pass

    @abstractmethod
    def run(self, items: List[Any]) -> List[Any]:
        """Run the pipeline on a batch of items."""
        pass
