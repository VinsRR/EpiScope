import re
from typing import Any, Dict, List, Tuple

class TextProcessor:
    """Handles text normalization and pattern matching."""
    
    URL_PATTERN = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
    DOI_PATTERN = re.compile(r"""\b10\.\d{4,9}/[^\s"'<>{}\[\]]+\b""", re.IGNORECASE)
    DOI_URL_PATTERN = re.compile(r'(?:https?://(?:doi\.org|dx\.doi\.org)/\s*)?(10\.\d{4,9}/[^\s"\'<>{}\[\]]+)', re.IGNORECASE)
    ACCESSION_PATTERN = re.compile(r"\b(?:GSE|PRJNA|SRP|ERP|SRR|SRA|ENA|PRJEB)\d{3,7}\b", re.IGNORECASE)
    SENTENCE_SPLIT = re.compile(r'(?<=[.?!])\s+')
    
    AVAILABILITY_TERMS = {
        "data available", "data are available", "data is available", "available at", 
        "available from", "available upon request", "upon request", "available on request",
        "data availability", "accession number", "deposited in", "deposited at",
        "uploaded to", "figshare", "zenodo", "github", "dryad", "ncbi", "sra", "ena"
    }
    
    REPOSITORIES = {"zenodo", "figshare", "dryad", "github", "ncbi", "gisaid", "sra", "ena", "dataverse"}
    
    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalize text for robust pattern matching."""
        if not text:
            return ""
        
        # Remove soft hyphens and normalize spaces
        text = text.replace("\u00AD", "").replace("\u00A0", " ")
        
        # Fix common broken patterns
        text = re.sub(r"https?\s*:\s*/\s*/", "https://", text, flags=re.IGNORECASE)
        text = re.sub(r"\bdoi\s*[.:]?\s*org\s*/\s*", "doi.org/", text, flags=re.IGNORECASE)
        text = re.sub(r"[\r\n]+", " ", text)
        text = re.sub(r"/\s+", "/", text)
        text = re.sub(r"\.\s+", ".", text)
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    @classmethod
    def extract_artifacts(cls, text: str) -> Dict[str, Any]:
        """Extract URLs, DOIs, and accession numbers from text."""
        normalized = cls.normalize_text(text)
        
        urls = cls.URL_PATTERN.findall(normalized)
        
        # Extract DOIs from both patterns
        dois = set()
        for match in cls.DOI_PATTERN.finditer(normalized):
            dois.add(match.group())
        for match in cls.DOI_URL_PATTERN.finditer(normalized):
            dois.add(match.group(1))
        
        accessions = cls.ACCESSION_PATTERN.findall(normalized)
        
        availability_score = sum(1 for term in cls.AVAILABILITY_TERMS 
                               if term in normalized.lower())
        
        return {
            "urls": urls,
            "dois": list(dois),
            "accessions": accessions,
            "availability_score": availability_score,
            "has_artifacts": bool(urls or dois or accessions),
            "normalized_text": normalized
        }

