from abc import ABC, abstractmethod
from typing import List
from episcope.utils.types import SearchResult
from episcope.utils.processing_utils import TextProcessor

class Postprocessor(ABC):
    """Abstract base class for post-processors."""

    @abstractmethod
    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Process a list of search results."""
        pass

class ArtifactExtractor(Postprocessor):
    """A post-processor to extract artifacts from search results."""

    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor

    def process(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract artifacts for each search result."""
        for result in results:
            result.artifacts = self.text_processor.extract_artifacts(result.text)
        return results
