from typing import Any, Dict, List, Optional
from enum import Enum

from ..utils.serialization import Serializable, SerializableList



class StructuredSection(Serializable):
    """Represents a structured section from GROBID parsing."""
    
    def __init__(self, title: str = "", content: str = "", section_type: str = "Other", 
                 references_cited: Optional[List[str]] = None, page_number: Optional[int] = None):
        self.title = title
        self.content = content
        self.section_type = section_type
        self.references_cited = references_cited or []
        self.page_number = page_number

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "references_cited": self.references_cited,
            "page_number": self.page_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredSection':
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            section_type=data.get("section_type", "Other"),
            references_cited=data.get("references_cited", []),
            page_number=data.get("page_number")
        )


class Reference(Serializable):
    """Represents a bibliographic reference."""
    
    def __init__(self, raw_text: str = "", is_data_source: bool = False, 
                 title: str = "", authors: Optional[List[str]] = None, 
                 year: Optional[int] = None, journal: str = "", doi: str = "", url: str = ""):
        self.raw_text = raw_text
        self.is_data_source = is_data_source
        self.title = title
        self.authors = authors or []
        self.year = year
        self.journal = journal
        self.doi = doi or ""
        self.url = url or ""


    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "is_data_source": self.is_data_source,
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "journal": self.journal,
            "doi": self.doi,
            "url": self.url
        }
    
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


class PaperMetadata(Serializable):
    """Represents metadata for an academic paper."""
    
    def __init__(self, title: str = "", abstract: str = "", authors: Optional[List[str]] = None,
                 publication_year: Optional[int] = None, journal: str = "", 
                 doi: str = "", keywords: Optional[List[str]] = None,
                 first_author: str = ""):
        self.title = title
        self.abstract = abstract
        self.authors = authors or []
        self.publication_year = publication_year
        self.journal = journal
        self.doi = doi
        self.keywords = keywords or []
        self.first_author = first_author
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "publication_year": self.publication_year,
            "journal": self.journal,
            "doi": self.doi,
            "keywords": self.keywords,
            "first_author": self.first_author
        }
    
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
            first_author=data.get("first_author", "")
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
    





# ============================================================================
# DATA MODELS
# ============================================================================

from pydantic import BaseModel, Field 
from dataclasses import dataclass, field

# @dataclass
# class Reference:
#     raw_text: str
#     is_data_source: bool = False
    
@dataclass
class DataSource:
    source_name: str
    url: str = "N/A"
    explanation: str = ""
    section_found: str = ""
    
@dataclass
class ExtractionResult:
    description: str
    data_sources: List[DataSource] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)



@dataclass
class ClassificationResult:
    paper_type: 'PaperType'
    confidence: float
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    evidence: Dict = field(default_factory=dict)


class PaperType(Enum):
    LITERATURE_REVIEW = "literature_review"
    DATA_ANALYSIS = "data_analysis" 
    # METHODS_TOOLS = "methods_tools"
    # CASE_STUDY = "case_study"
    # COMMENTARY = "commentary"
    # OTHER = "other"
    # UNCLEAR = "unclear"


# Pydantic schema for structured LLM output
class ClassificationOutput(BaseModel):
    classification: str = Field(..., description="One of: A,B")
    reasoning: str = Field(..., description="Short 1-2 sentence explanation")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    class_probabilities: Optional[Dict[str, float]] = None





@dataclass
class Chunk:
    id: str
    section_type: str
    title: str
    text: str
    metadata: Dict = field(default_factory=dict)