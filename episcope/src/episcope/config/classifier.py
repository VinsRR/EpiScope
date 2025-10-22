from episcope.schemas import PaperType, DataAccessibility, DataNation, DataType
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class PaperClassifierConfig:
    """Configuration for the paper classifier."""
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
    system_prompt: str = "You are an senior epidemiologist doing reviewing the recent literatures. You are sorting your papers into two categories: Literature Review and Data Analysis."

    user_prompt_template: str = """
    You are an senior academic epidemiologist. Your task is to determine the primary type of a research paper.

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

@dataclass
class DataAccessibilityClassifierConfig(PaperClassifierConfig):
    """Configuration for classifying data accessibility."""
    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "open_access": [
            "All data and code used in this study are publicly available at [URL/DOI].",
            "The dataset can be downloaded from the project's GitHub repository.",
            "Data is available in a public repository."
        ],
        "restricted_access": [
            "Data are available from the authors upon reasonable request and with permission of the steering committee.",
            "Access to the data requires a data use agreement.",
            "Due to the sensitive nature of the data, access is restricted."
        ],
        "not_available": [
            "The data that support the findings of this study are not publicly available.",
            "Data sharing is not applicable to this article as no new data were created or analyzed.",
            "The data for this study are proprietary and cannot be shared."
        ]
    })
    classification_mapping: Dict[str, DataAccessibility] = field(default_factory=lambda: {
        "A": DataAccessibility.OPEN_ACCESS,
        "B": DataAccessibility.RESTRICTED_ACCESS,
        "C": DataAccessibility.NOT_AVAILABLE
    })
    category_labels: Dict[str, str] = field(default_factory=lambda: {
        "A": "Open Access",
        "B": "Restricted Access",
        "C": "Not Available"
    })
    system_prompt: str = "You are a data librarian classifying research papers based on the accessibility of their supporting data."
    user_prompt_template: str = """
    You are a data librarian. Your task is to determine the accessibility of the supporting data in a research paper.

        **Categories:**
{categories}

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to determine the data accessibility statement.
2. Select a single letter that best represents the data's accessibility.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""

@dataclass
class DataNationClassifierConfig(PaperClassifierConfig):
    """Configuration for classifying the nation(s) of data origin."""
    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "usa": ["Data was sourced from the US National Health and Nutrition Examination Survey (NHANES)."],
        "uk": ["We used data from the UK Biobank, a large-scale biomedical database and research resource."],
        "china": ["This study is based on data from the China Health and Retirement Longitudinal Study (CHARLS)."],
        "europe_multiple": ["Data from several European countries were included in this analysis."],
        "global": ["We conducted a global survey covering participants from multiple continents."],
        "synthetic": ["We generated a synthetic dataset to simulate patient profiles for this study."],
        "not_specified": ["The geographic origin of the data was not specified in the paper."]
    })
    classification_mapping: Dict[str, DataNation] = field(default_factory=lambda: {
        "A": DataNation.USA, "B": DataNation.UK, "C": DataNation.CHINA,
        "D": DataNation.EUROPE_MULTIPLE, "E": DataNation.GLOBAL,
        "F": DataNation.SYNTHETIC, "G": DataNation.NOT_SPECIFIED
    })
    category_labels: Dict[str, str] = field(default_factory=lambda: {
        "A": "USA", "B": "UK", "C": "China", "D": "Europe (Multiple)",
        "E": "Global", "F": "Synthetic", "G": "Not Specified"
    })
    system_prompt: str = "You are a research analyst identifying the geographic origin of data in epidemiological studies."
    user_prompt_template: str = """
    You are a research analyst. Your task is to identify the geographic origin of the data used in a research paper.

        **Categories:**
{categories}

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to determine the nation(s) the data is from.
2. Select a single letter that best represents the data's origin.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""

@dataclass
class DataTypeClassifierConfig(PaperClassifierConfig):
    """Configuration for classifying the type of data used."""
    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "traditional": [
            "We analyzed electronic health records from a large hospital network.",
            "Survey data was collected using a standardized questionnaire."
        ],
        "non_traditional": [
            "We collected data from Twitter to analyze public sentiment.",
            "Participants wore Fitbit devices to track physical activity."
        ],
        "synthetic": ["A synthetic dataset was generated to test our model."],
        "not_specified": ["The type of data used was not explicitly described in the methods."]
    })
    classification_mapping: Dict[str, DataType] = field(default_factory=lambda: {
        "A": DataType.TRADITIONAL, "B": DataType.NON_TRADITIONAL,
        "C": DataType.SYNTHETIC, "D": DataType.NOT_SPECIFIED
    })
    category_labels: Dict[str, str] = field(default_factory=lambda: {
        "A": "Traditional (e.g., surveys, EHR)",
        "B": "Non-Traditional (e.g., social media, wearables)",
        "C": "Synthetic",
        "D": "Not Specified"
    })
    system_prompt: str = "You are a data scientist classifying papers by the type of data used."
    user_prompt_template: str = """
    You are a data scientist. Your task is to classify the type of data used in a research paper.

        **Categories:**
{categories}

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Analyze the evidence to determine the type of data used.
2. Select a single letter that best represents the data type.
3. Return a single JSON object adhering to the schema. Do not add extra text.

**Schema:**
{schema}
"""
