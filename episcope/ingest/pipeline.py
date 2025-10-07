"""
ingest/pipeline.py

High level ingestion pipeline for EpiScope.

This pipeline supports ingesting documents from local PDF files or
remote identifiers such as DOIs.  For PDF ingestion the pipeline
indexes the document using the configured RAG method.  DOI ingestion
fetches the PDF via external services (e.g. PubMed or publishers) and
then indexes it.  Future enhancements may add OCR for scanned PDFs
and additional metadata extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from ..retrieve import RAGFactory
from ..settings import CONFIG

logger = logging.getLogger(__name__)


class IngestPipeline:
    """Pipeline responsible for ingesting documents into the system."""

    def __init__(self, rag_method: str = "text") -> None:
        self.rag = RAGFactory.get(rag_method)

    def ingest_pdf(self, file_path: str | Path) -> None:
        """Index a PDF file on disk."""
        path = str(file_path)
        logger.info(f"Ingesting PDF: {path}")
        self.rag.index(path)

    def ingest_directory(self, directory: str | Path) -> None:
        """Recursively index all PDFs in a directory."""
        path = str(directory)
        self.rag.index(path)

    def ingest_doi(self, doi: str) -> None:
        """Fetch a PDF via its DOI and index it.

        At present this method is a stub.  In a future release it will
        use PubMed or CrossRef to resolve the DOI and download the PDF
        for indexing.
        """
        logger.warning(f"DOI ingestion not yet implemented: {doi}")
        # TODO: implement DOI ingestion using pubmed_functions.py

