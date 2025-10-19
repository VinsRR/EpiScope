from typing import Any, Dict, List, Optional
from dataclasses import asdict, dataclass, field

from .serialization import Serializable, SerializableList

@dataclass
class StructuredSection(Serializable):
    """Represents a structured section from GROBID parsing."""
    title: str = ""
    content: str = ""
    section_type: str = "Other"
    references_cited: List[str] = field(default_factory=list)
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredSection':
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            section_type=data.get("section_type", "Other"),
            references_cited=data.get("references_cited", []),
            page_number=data.get("page_number")
        )


@dataclass
class Reference(Serializable):
    """Represents a bibliographic reference."""
    raw_text: str = ""
    is_data_source: bool = False
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Reference':
        return cls(
            raw_text=data.get("raw_text", ""),
            is_data_source=data.get("is_data_source", False),
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year"),
            journal=data.get("journal", ""),
            doi=data.get("doi", ""),
            url=data.get("url", "")
        )

@dataclass
class PaperMetadata(Serializable):
    """Represents metadata for an academic paper."""
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    keywords: List[str] = field(default_factory=list)
    first_author: str = ""
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperMetadata':
        return cls(
            title=data.get("title", ""),
            abstract=data.get("abstract", ""),
            authors=data.get("authors", []),
            publication_year=data.get("publication_year"),
            journal=data.get("journal", ""),
            doi=data.get("doi", ""),
            keywords=data.get("keywords", []),
            first_author=data.get("first_author", ""),
            file_path=data.get("file_path")
        )


class SectionList(SerializableList):
    """Container for structured sections."""
    
    @classmethod
    def _item_class(cls):
        return StructuredSection


class ReferenceList(SerializableList):
    """Container for references."""
    
    @classmethod
    def _item_class(cls):
        return Reference
