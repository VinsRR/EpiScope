from typing import List, Optional
from pydantic import BaseModel

class ExtractionItemSchema(BaseModel):
    item_type: str
    name: str
    url: Optional[str] = None
    explanation: Optional[str] = None
    section_found: Optional[str] = None
    raw_text: Optional[str] = None

class ExtractionResultSchema(BaseModel):
    description: str
    items: List[ExtractionItemSchema] = []

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