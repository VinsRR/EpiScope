from typing import Any, Dict, List, Optional
from ...utils.serialization import Serializable
from .data_blueprints import Reference


class DataSource(Serializable):
    """Represents a data source identified in a paper."""
    
    def __init__(self, source_name: str = "", url: str = "N/A", 
                 explanation: str = "", section_found: str = ""):
        self.source_name = source_name
        self.url = url
        self.explanation = explanation
        self.section_found = section_found
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_name": self.source_name,
            "url": self.url,
            "explanation": self.explanation,
            "section_found": self.section_found
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSource':
        return cls(
            source_name=data.get("source_name", ""),
            url=data.get("url", "N/A"),
            explanation=data.get("explanation", ""),
            section_found=data.get("section_found", "")
        )


class ExtractionResult(Serializable):
    """Container for extraction results from LLM processing."""
    
    def __init__(self, data_sources_description: str = "", 
                 analysis_type: str = "", data_sources: Optional[List[DataSource]] = None,
                 references: Optional[List[Reference]] = None):
        self.data_sources_description = data_sources_description
        self.analysis_type = analysis_type
        self.data_sources = data_sources or []
        self.references = references or []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_sources_description": self.data_sources_description,
            "analysis_type": self.analysis_type,
            "data_sources": [ds.to_dict() for ds in self.data_sources],
            "references": [ref.to_dict() for ref in self.references]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        data_sources = [DataSource.from_dict(ds) for ds in data.get("data_sources", [])]
        references = [Reference.from_dict(ref) for ref in data.get("references", [])]
        
        return cls(
            data_sources_description=data.get("data_sources_description", ""),
            analysis_type=data.get("analysis_type", ""),
            data_sources=data_sources,
            references=references
        )
    






from pydantic import BaseModel, ValidationError

# Small, strict output schema (adapt to your real ExtractionResult)
class DataSourceItem(BaseModel):
    source_name: str
    url: Optional[str] = None
    explanation: Optional[str] = None
    section_found: Optional[str] = None

class ExtractionResult(BaseModel):
    data_sources_description: Optional[str] = None
    data_sources: Optional[List[DataSourceItem]] = []
    references: Optional[List[Dict[str, Any]]] = []
    matched_dois: Optional[List[str]] = []


    def add_matched_dois(self, dois: List[str]):
        if self.matched_dois is None:
            self.matched_dois = []
        self.matched_dois.extend(dois)

# Minimal validator for the LLM output
class DataSourceSchema(BaseModel):
    data_sources_description: str
    data_sources: Optional[List[Dict]] = []
    references: Optional[List[Dict]] = []