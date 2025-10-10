from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import asdict, dataclass, field

from ...utils.serialization import Serializable, SerializableList



from dataclasses import dataclass, field

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


from dataclasses import asdict, dataclass, field

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