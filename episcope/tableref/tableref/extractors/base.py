from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseExtractor(ABC):
    """Abstract base class for a table extractor."""

    @abstractmethod
    def extract(self, pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Extracts tables from a given PDF file.

        Args:
            pdf_path: The path to the PDF file.
            output_dir: The directory to save any intermediate files (like CSVs).

        Returns:
            A list of dictionaries, where each dictionary represents a table.
        """
        raise NotImplementedError
