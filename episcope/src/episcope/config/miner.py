from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PrecisionMinerConfig:
    """Base configuration for the Precision Miner. Do not use directly.
    Instead, use one of the specialized configurations below."""
    top_k: int = 15
    retrieval_templates: List[str] = field(default_factory=list)
    section_filters: Optional[List[str]] = None
    system_prompt: str = ""
    user_prompt_template: str = ""

@dataclass
class FindDataSourcesConfig(PrecisionMinerConfig):
    """Configuration for finding data sources."""
    retrieval_templates: List[str] = field(default_factory=lambda: [
        "What datasets, databases, or data repositories were used in this study?",
        "Where did the data come from? What are the sources of the data analyzed?",
        "Which surveys, cohorts, or existing data collections does this study draw upon?",
        "What external data sources, registries, or archives are referenced?",
        "Describe the origin and provenance of the data used in the analysis.",
        "What are the URLs or DOIs for the data used in this study?",
        "Which data was obtained from external sources versus collected by the authors?",
        "What publicly available datasets are cited or referenced?",
    ])
    section_filters: Optional[List[str]] = field(default_factory=lambda: None) # ["Methods", "Data", "Study Design", "Participants"]
    system_prompt: str = """You are a senior epidemiologist. 
You are scanning a research paper to identify key data sources utilized in the study.
You are interested in understanding where the data originated, if it was collected by the authors or sourced from existing datasets and, in the case of existing datasets, what they are.
You are particularly interested in collecting references to datasets, surveys, or repositories mentioned in the paper."""
    user_prompt_template: str = """You have been provided with extracts from the paper that may contain relevant information.
**Relevant Extracts:**
{chunks_info}
**Instructions:**
1. Analyze the evidence to identify the key data sources.
2. For each data source, provide its name, a URL if available, a brief explanation
3. Return a single JSON object adhering to the schema. Do not add extra text.
**Schema:**
{schema}

**Example output:**
{{
"description": "Brief summary of data sources identified",
"items": [
{{
"name": "Dataset/database name",
"url": "Full URL or null",
"explanation": "How it was used in the study",
"raw_text": "Direct quote from paper"
}}
// ... more items as needed
]
}}
DO NOT return the example output above. Instead,

Output **only** a single JSON object matching the schema above. **Do not** print the schema, do not add commentary. Do not just return the example output. 
If no data sources are found, return an empty list for items with an appropriate description.
"""

@dataclass
class FindSupplementaryLinksConfig(PrecisionMinerConfig):
    """Configuration for finding supplementary materials and their links."""
    retrieval_templates: List[str] = field(default_factory=lambda: [
        "Are there any supplementary materials, appendices, or additional files?",
        "Where can I find supplementary data, tables, or figures?",
        "What supplementary information is available online?",
        "Are there links to supplementary methods, results, or datasets?",
        "Where are the supplementary materials hosted or referenced?",
        "What additional resources or files accompany this paper?",
        "Are there DOIs or URLs for supplementary content?",
        "Which appendices or supporting information files exist?",
    ])
    section_filters: Optional[List[str]] = field(default_factory=lambda: None)  # Search entire paper
    system_prompt: str = """You are a senior research librarian and data specialist.
You are scanning a research paper to identify and catalog all supplementary materials, appendices, and additional resources.
You are interested in finding URLs, DOIs, file references, and any mentions of supplementary content that supports the main paper.
You are particularly focused on extracting accessible links and understanding what each supplementary resource contains."""
    user_prompt_template: str = """You have been provided with extracts from the paper that may contain relevant information.
**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to identify all supplementary materials, appendices, and additional files.
2. For each supplementary resource, provide its name/identifier, URL/DOI if available, and a brief description of its contents.
3. Look for patterns like "Supplementary Figure", "Appendix", "Supporting Information", "Additional File", "S1 Table", etc.
4. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}

**Example output:**
{{
"description": "Brief summary of supplementary materials found",
"items": [
{{
"name": "Supplementary material identifier (e.g., 'Appendix A', 'S1 Data')",
"url": "Full URL, DOI, or null if not provided",
"explanation": "What the supplementary material contains",
"raw_text": "Direct quote mentioning this resource"
}}
// ... more items as needed
]
}}
DO NOT return the example output above. Instead,

Output **only** a single JSON object matching the schema above. **Do not** print the schema, do not add commentary. Do not just return the example output.
If no supplementary materials are found, return an empty list for items with an appropriate description.
"""


@dataclass
class IdentifyKeyReferencesConfig(PrecisionMinerConfig):
    """Configuration for identifying key references and foundational literature."""
    retrieval_templates: List[str] = field(default_factory=lambda: [
        "What are the key references and foundational works cited in this study?",
        "Which prior studies are most central to the theoretical framework?",
        "What seminal papers or influential works does this research build upon?",
        "Which citations appear most frequently or are emphasized in the text?",
        "What are the methodological references that inform the study design?",
        "Which previous research findings are critically discussed or extended?",
        "What landmark studies are referenced in the introduction or background?",
        "Which works establish the scientific basis for this investigation?",
    ])
    section_filters: Optional[List[str]] = field(default_factory=lambda: ["Introduction", "Literature Review", "Background", "Methods", "Discussion"])
    system_prompt: str = """You are a senior academic researcher and bibliometric analyst.
You are scanning a research paper to identify the most important and influential references that form the scientific foundation of the study.
You are interested in distinguishing between casual citations and truly foundational works that are central to the paper's research questions, methods, or theoretical framework.
You are particularly focused on identifying references that are repeatedly mentioned, discussed in depth, or represent seminal contributions to the field."""
    user_prompt_template: str = """You have been provided with extracts from the paper that may contain relevant information.
**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to identify the most important references cited in the paper.
2. Prioritize references that are: discussed extensively, cited multiple times, central to the study's rationale, or represent foundational methodologies.
3. For each key reference, provide the citation text as it appears, an explanation of its importance to the study, and any identifiers (DOI, PMID).
4. Focus on quality over quantity - identify truly key references rather than listing all citations.
5. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}

**Example output:**
{{
"description": "Brief summary of the key literature identified",
"items": [
{{
"name": "Short identifier (e.g., 'Smith et al. 2020' or first author + year)",
"url": "DOI, PubMed link, or null if not available",
"explanation": "Why this reference is foundational to the study",
"raw_text": "Full citation as it appears in the paper"
}}
// ... more items as needed
]
}}
DO NOT return the example output above. Instead,

Output **only** a single JSON object matching the schema above. **Do not** print the schema, do not add commentary. Do not just return the example output.
If no key references are identified, return an empty list for items with an appropriate description.
"""
