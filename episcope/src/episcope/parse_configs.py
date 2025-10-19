from episcope.schemas import PaperType
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class PaperClassifierConfig:
    """Configuration for the paper classifier."""
    model_name: str = "qwen2.5vl:3b" #"deepseek-r1:7b"
    similarity_threshold: float = 0.75
    top_k: int = 10
    embedding_model: str = "jinaai/jina-embeddings-v3"
    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "literature_review": [
            "Several studies have examined the relationship between X and Y. Smith et al. (2020) found significant associations, while Jones et al. (2021) reported mixed results. A systematic review by Brown et al. (2019) identified 45 relevant studies.",
            "We conducted a systematic literature search across PubMed, Embase, and Web of Science databases. Studies were included if they met inclusion criteria. Two reviewers independently screened titles and abstracts.",
        ],
        "data_analysis": [
            "We analyzed data from the National Health Survey (n=15,432 participants). Data collection occurred between January 2020 and December 2022. Statistical analyses were performed using R version 4.2.",
            "The dataset contained 23,891 observations across 15 variables. Missing data patterns were examined using multiple imputation. Primary outcomes were measured using validated instruments.",
        ],
    })
    classification_mapping: Dict[str, PaperType] = field(default_factory=lambda: {
        "A": PaperType.LITERATURE_REVIEW,
        "B": PaperType.DATA_ANALYSIS
    })
    category_labels: Dict[str, str] = field(default_factory=lambda: {
        "A": "Literature Review",
        "B": "Data Analysis"
    })
    prompt_template: str = """You are an expert academic classifier. Your task is to determine the primary type of a research paper.
**Categories:**
{categories}

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to determine the paper's main contribution.
2. Select a single letter that best represents the paper's primary classification.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""





#########################################################################################################
#########################################################################################################
#########################################################################################################



@dataclass
class PrecisionMinerConfig:
    """Base configuration for the Precision Miner. Do not use directly.
    Instead, use one of the specialized configurations below."""
    model_name: str = "deepseek-r1:7b"
    top_k: int = 10
    retrieval_templates: List[str] = field(default_factory=list)
    section_filters: Optional[List[str]] = None
    prompt_template: str = ""

@dataclass
class FindDataSourcesConfig(PrecisionMinerConfig):
    """Configuration for finding data sources."""
    retrieval_templates: List[str] = field(default_factory=lambda: [
        "What are the primary data sources used in this study?",
        "Describe the data collection methods and sources.",
        "What is the population or sample for this study?",
    ])
    section_filters: Optional[List[str]] = field(default_factory=lambda: ["Methods", "Data", "Study Design", "Participants"])
    prompt_template: str = """You are an expert data extractor. Your task is to extract structured information about data sources from a research paper.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to identify the key data sources.
2. For each data source, provide its name, a URL if available, a brief explanation, and the section where it was found.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""

@dataclass
class FindSupplementaryLinksConfig(PrecisionMinerConfig):
    """Configuration for finding supplementary links."""
    top_k: int = 5
    retrieval_templates: List[str] = field(default_factory=lambda: [
        "Are there any supplementary materials or appendices?",
        "Where can I find the supplementary data?",
        "Is there a link to the appendix?",
    ])
    prompt_template: str = """You are an expert data extractor. Your task is to find links to supplementary materials from a research paper.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to identify links to supplementary materials, appendices, or supplementary data.
2. For each link, provide the name of the material and the URL.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""

@dataclass
class IdentifyKeyReferencesConfig(PrecisionMinerConfig):
    """Configuration for identifying key references."""
    top_k: int = 15
    retrieval_templates: List[str] = field(default_factory=lambda: [
        "What are the key references in the introduction?",
        "What are the most cited works in the literature review?",
        "What are the foundational papers for this study?",
    ])
    section_filters: Optional[List[str]] = field(default_factory=lambda: ["Introduction", "Literature Review", "Background"])
    prompt_template: str = """You are an expert literature analyst. Your task is to identify key references from a research paper.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to identify the most important references cited in the paper.
2. For each key reference, provide the raw text of the reference.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""