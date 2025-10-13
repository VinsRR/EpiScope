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


class QueryGenerator:
    """Generates queries for different paper types."""
    
    PAPER_TYPE_QUERIES = {
        "data_analysis": {
            "primary": [
                'Supplementary data are available at"', 
                '"Detailed data of this study are provided in the Supplementary Information and will be available upon"',
                '"source data is available at"',
                "The data collection integrates multiple data sources"
            ],
            "secondary": [
                "Based on the data provided by",
                "We collected data on",
                
            ]
        },

        "literature_review": {
            "primary": [
                'The full list of included studies is available at', 
                'Appendix: List of included studies', 
                'The list of data entries for all included studies are provided in',
                'Supplementary material for this article can be found online at'
            ],
            "secondary": [
            ]
        },
        "methods_tools": {
            "primary": [
                '"code available"', '"repository"', '"github"',
                '"software availability"'
            ],
            "secondary": [
                "implementation benchmark dataset",
                "source code docker package"
            ]
        },
        "case_study": {
            "primary": [
                '"data availability"', "surveillance data",
                "institutional data"
            ],
            "secondary": [
                "case study dataset",
                "outbreak data collection"
            ]
        }
    }
    
    @classmethod
    def generate_queries(cls, paper_type: str, metadata=None, hyde_callable=None) -> List[Tuple[str, float, str]]:
        """Generate weighted queries for the given paper type."""
        queries = []
        
        # Get paper-type specific queries
        type_queries = cls.PAPER_TYPE_QUERIES.get(paper_type, cls.PAPER_TYPE_QUERIES["data_analysis"])
        
        # Add primary queries (high weight)
        for query in type_queries["primary"]:
            queries.append((query, 5.0, f"primary_{paper_type}"))
        
        # Add secondary queries (medium weight)
        for query in type_queries["secondary"]:
            queries.append((query, 3.0, f"secondary_{paper_type}"))
        
        # Generate HyDE queries if available
        if hyde_callable and metadata:
            try:
                hyde_queries = hyde_callable(f"Generate data availability statements for {paper_type} paper")
                for i, hq in enumerate(hyde_queries[:3]):
                    queries.append((hq, 3.5, f"hyde_{i}"))
            except Exception as e:
                logger.debug(f"HyDE generation failed: {e}")
        
        # Deduplicate and sort by weight
        unique_queries = {}
        for query, weight, reason in queries:
            key = query.lower().strip()
            if key not in unique_queries or weight > unique_queries[key][1]:
                unique_queries[key] = (query, weight, reason)
        
        result = list(unique_queries.values())
        result.sort(key=lambda x: x[1], reverse=True)
        return result
