from typing import List, Optional
from pydantic import BaseModel, Field

# class ExtractionItemSchema(BaseModel):
#     item_type: str
#     name: str
#     url: Optional[str] = None
#     explanation: Optional[str] = None
#     section_found: Optional[str] = None
#     raw_text: Optional[str] = None

# class ExtractionResultSchema(BaseModel):
#     description: str
#     items: List[ExtractionItemSchema] = []

class ExtractionItemSchema(BaseModel):
    # item_type: str = Field(..., description="Type of item, e.g., 'reference'")
    name: str = Field(..., description="Full source name as appears in paper")
    url: Optional[str] = Field(None, description="URL/DOI if available else null")
    explanation: Optional[str] = Field(None, description="Why this item is key")
    # section_found: Optional[str] = Field(None, description="Paper section where it is pivotal")
    raw_text: Optional[str] = Field(None, description="Exact reference snippet from the paper")

class ExtractionResultSchema(BaseModel):
    description: str = Field(..., description="Short summary of why these references were selected")
    items: List[ExtractionItemSchema] = Field(default_factory=list, description="List of key references")


class DataSourceItemSchema(BaseModel):
    source_name: str
    url: str
    explanation: str
    section_found: str

class ReferenceSchema(BaseModel):
    raw_text: str
    is_data_source: bool = False
    title: str = ""
    authors: List[str] = []
    year: Optional[int] = None
    journal: str = ""
    doi: str = ""
    url: str = ""

class DataSourceSchema(BaseModel):
    data_sources_description: str
    data_sources: List[DataSourceItemSchema]
    references: Optional[List[ReferenceSchema]] = None