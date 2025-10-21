"""
TableRef Extractor Package
"""
from .extractor import TableExtractor
from .matching import ReferenceMatcher, SimpleSurnameMatcher, LateInteractionMatcher
from .clients import fetch_doi_from_crossref
from .utils import default_normalize

__all__ = [
    "TableExtractor",
    "ReferenceMatcher",
    "SimpleSurnameMatcher",
    "LateInteractionMatcher",
    "fetch_doi_from_crossref",
    "default_normalize",
]
