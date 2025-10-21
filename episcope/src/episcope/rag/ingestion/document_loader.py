"""Document loading utilities for EpiScope.

This module defines abstract and concrete classes for extracting
structured text from documents on disk.  It provides a thin
abstraction layer over various backends, such as Unstructured and
GROBID, to normalise extracted content into a common format
consisting of :class:`~episcope.core.blueprints.data_blueprints.StructuredSection`
instances accompanied by minimal metadata.  The goal is to make
documents available to indexers (e.g., :class:`~episcope.index.paper_indexer.PaperIndexer`)
without requiring callers to worry about low‑level parsing details.

Two concrete extractors are supplied:

* :class:`UnstructuredDocumentLoader` parses PDF and other files using
  the optional Unstructured library.  It groups titles and
  paragraphs into sections.  If Unstructured is not installed, it
  falls back to reading plain text files.
* :class:`GrobidDocumentLoader` leverages the grobid_client library to
  produce TEI XML and then extracts sections.  If the grobid
  dependency is missing or parsing fails, it falls back to the
  Unstructured loader.

The :class:`DocumentLoaderFactory` exposes a simple factory method to
instantiate a loader based on a string key (e.g., ``"unstructured"`` or
``"grobid"``).  Additional loaders can be registered by extending the
factory.
"""

from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from episcope.schemas import StructuredSection, PaperMetadata, Reference
from episcope.db.academic_db import AcademicDB

logger = logging.getLogger(__name__)


class AbstractDocumentLoader(abc.ABC):
    """Abstract base class for document loaders.

    Concrete subclasses must implement the :meth:`load` method to
    extract structured sections, metadata, and references from a single
    document.  Optionally subclasses may override :meth:`load_directory`
    to support batch loading.
    """
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def load(self, file_path: Union[str, Path]) -> Tuple[List[StructuredSection], PaperMetadata, List[Reference]]:
        """Extract structured sections, metadata, and references from a document.

        Args:
            file_path: Path to a single document (PDF, text, etc.).

        Returns:
            A tuple ``(sections, metadata, references)``. For loaders that do not
            support reference extraction, ``references`` will be an empty list.
        """

    def load_directory(self, dir_path: Union[str, Path]) -> Dict[str, Tuple[List[StructuredSection], PaperMetadata, List[Reference]]]:
        """Extract documents from all supported files under a directory.

        The default implementation walks the directory tree and
        applies :meth:`load` to each encountered file with a supported
        extension.  Subclasses may override this to provide more
        sophisticated behaviour (e.g. parallel extraction).

        Args:
            dir_path: Root of the directory to traverse.

        Returns:
            A mapping from file names (as strings) to the extracted
            sections, metadata, and references for each file.
        """
        results: Dict[str, Tuple[List[StructuredSection], PaperMetadata, List[Reference]]] = {}
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"Expected directory: {dir_path}")
        for child in path.rglob("*"):
            if child.is_file() and self._is_supported(child):
                sections, meta, refs = self.load(child)
                results[child.stem] = (sections, meta, refs)
        return results

    def _is_supported(self, file_path: Path) -> bool:
        """Return True if this loader can process the given file extension."""
        return file_path.suffix.lower() in {'.pdf', '.txt', '.md', '.text'}

    def extract_paper(
        self,
        file_path: str | Path,
        *,
        strategy_name: str,
        db: AcademicDB,
    ) -> None:
        """Extract structured data from a single document and persist it.

        This function uses the loader's ``load`` method to obtain
        sections, metadata and references from a document. The results
        are inserted into the provided :class:`AcademicDB` under
        the specified ``strategy_name``.

        Args:
            file_path: Location of the document to process.
            strategy_name: Namespace under which to store the results.
            db: Instance of :class:`AcademicDB` to persist data.

        Raises:
            FileNotFoundError: If the document file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        paper_id = path.stem

        sections, metadata, references = self.load(path)
        metadata.file_path = str(path)

        sections_dicts: List[Dict] = [s.to_dict() for s in sections]
        metadata_dict: Dict = metadata.to_dict()
        references_dicts: List[Dict] = [r.to_dict() for r in references]

        db.insert(paper_id, "sections", strategy_name, sections_dicts)
        db.insert(paper_id, "metadata", strategy_name, metadata_dict)
        db.insert(paper_id, "references", strategy_name, references_dicts)
        logger.info(
            f"Persisted extraction for {paper_id} under strategy {strategy_name}."
        )

    def extract_directory(
        self,
        dir_path: str | Path,
        *,
        strategy_name: str,
        db: AcademicDB,
    ) -> None:
        """Process all supported files in a directory.

        Recursively walks the directory tree rooted at ``dir_path`` and
        applies :meth:`extract_paper` to each supported file. Errors are
        logged and skipped rather than aborting the entire run.
        """
        root = Path(dir_path)
        if not root.is_dir():
            raise ValueError(f"Expected directory: {dir_path}")
        for child in root.rglob("*"):
            if child.is_file() and self._is_supported(child):
                try:
                    self.extract_paper(
                        child,
                        strategy_name=strategy_name,
                        db=db,
                    )
                except Exception as exc:
                    logger.error(f"Failed to process {child}: {exc}")


class UnstructuredDocumentLoader(AbstractDocumentLoader):
    """Document loader that uses the Unstructured library when available.

    This loader attempts to parse PDF files with the optional
    ``unstructured`` package (``unstructured.partition.pdf``).  It
    collects textual elements, grouping headings and paragraphs into
    sections.  If Unstructured or its dependencies are missing, it
    falls back to reading plain text files.  For non‑PDF formats
    (e.g., ``.txt`` or ``.md``) the loader simply reads the file
    contents and wraps it in a single :class:`StructuredSection`.
    """
    def __init__(self) -> None:
        super().__init__()

    def load(self, file_path: Union[str, Path]) -> Tuple[List[StructuredSection], PaperMetadata, List[Reference]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            sections, metadata = self._load_pdf(path)
        else:
            sections, metadata = self._load_text(path)
        
        return sections, metadata, []

    def _load_pdf(self, path: Path) -> Tuple[List[StructuredSection], PaperMetadata]:
        """Parse a PDF using Unstructured, falling back on plain text."""
        try:
            from unstructured.partition.pdf import partition_pdf
            from unstructured.documents.elements import Title, Header, Text as TextElement
        except ImportError:
            logger.warning("Unstructured library not found. PDFs will be treated as plain text.")
            return self._load_text(path)

        try:
            elements = partition_pdf(
                filename=str(path),
                infer_table_structure=False,
                strategy="hi_res",
                extract_image_block_types=[],
            )
        except Exception as exc:
            logger.warning(f"Failed to parse {path} with unstructured: {exc}")
            return self._load_text(path)

        sections: List[StructuredSection] = []
        current_title: str = ""
        current_content: List[str] = []
        for el in elements:
            if isinstance(el, (Title, Header)):
                if current_content:
                    sections.append(
                        StructuredSection(title=current_title, content="\n".join(current_content))
                    )
                    current_content = []
                current_title = el.text.strip()
            elif isinstance(el, TextElement):
                text = el.text.strip()
                if text:
                    current_content.append(text)
        
        if current_content:
            sections.append(
                StructuredSection(title=current_title, content="\n".join(current_content))
            )
        
        metadata = PaperMetadata(title=path.stem)
        return sections, metadata

    def _load_text(self, path: Path) -> Tuple[List[StructuredSection], PaperMetadata]:
        """Load a plain text or markdown file into a single section."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            with path.open("rb") as fh:
                content = fh.read().decode("utf-8", errors="ignore")
        
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        sections = [StructuredSection(title="", content=p) for p in paragraphs]
        metadata = PaperMetadata(title=path.stem)
        return sections, metadata


class GrobidDocumentLoader(AbstractDocumentLoader):
    """Document loader that uses GROBID to extract structured sections.

    This loader relies on the `grobid_client_python` package to
    convert a PDF into TEI XML.  It then parses the TEI using a
    minimal XPath approach to extract section titles and paragraphs.
    If GROBID is not installed or the parsing fails, the loader
    falls back to the :class:`UnstructuredDocumentLoader`.
    """

    def __init__(self, grobid_url: Optional[str] = None) -> None:
        self.grobid_url = grobid_url
        self._fallback = UnstructuredDocumentLoader()
        try:
            from episcope.rag.ingestion.local_grobid_client import GrobidClient
            self.client = GrobidClient(grobid_server=self.grobid_url or "http://localhost:8070")
        except ImportError as e:
            logger.warning(f"Could not import GrobidClient, GROBID loader will not be available: {e}")
            self.client = None

    def load(self, file_path: Union[str, Path]) -> Tuple[List[StructuredSection], PaperMetadata, List[Reference]]:
        """Load a PDF and return sections, metadata and references."""
        path = Path(file_path)
        if path.suffix.lower() != ".pdf" or not self.client:
            return self._fallback.load(path)

        try:
            metadata, sections, references = self.client.process_fulltext(str(path))
            return sections, metadata, references
        except Exception as exc:
            logger.warning(
                f"GROBID processing failed for {path}: {exc}; falling back to unstructured"
            )
            return self._fallback.load(path)


class DocumentLoaderFactory:
    """Factory for obtaining document loaders by name.

    This factory maintains a mapping of string keys to callable
    constructors.  To add new loaders, assign a function or class
    reference to :data:`LOADER_REGISTRY`.
    """

    LOADER_REGISTRY: Dict[str, callable] = {
        "unstructured": UnstructuredDocumentLoader,
        "grobid": GrobidDocumentLoader,
    }

    @classmethod
    def get_loader(cls, name: str, **kwargs: any) -> AbstractDocumentLoader:
        """Instantiate a document loader by name.

        Args:
            name: Name of the loader (e.g., ``"unstructured"``, ``"grobid"``).
            **kwargs: Additional keyword arguments passed to the loader constructor.

        Returns:
            An instance of :class:`AbstractDocumentLoader`.

        Raises:
            ValueError: If no loader is registered under the given name.
        """
        name = name.lower()
        if name not in cls.LOADER_REGISTRY:
            raise ValueError(f"Unknown document loader: {name}")
        return cls.LOADER_REGISTRY[name](**kwargs)
