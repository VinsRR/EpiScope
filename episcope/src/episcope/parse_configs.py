GT_PAPER_TYPES = {
    "2020_He_infectious_period": "data",
    "2020_Lin_r0_basic_": "review",
    "2021_Ahammed_r0": "review",
    "2021_Zhu_infectious_period": "data",
    "2022_Du_k": "review",
    "2020_Xie_r0_basic_": "review",
    "2020_Izadi_r0_basic_": "review",
    "2020_Park_ifr": "review",
    "2020_Rai_serial_interval": "review",
    "2020_Yang_serial_interval": "data",
    "2021_Alene_incubation_period": "review",
    "2021_Ali_serial_interval": "review",
    "2021_Davies_r0": "data",
    "2021_Liu_and_Rocklöv_r0_basic_": "review",
    "2022_Águila-Mejía_infectious_period": "data",
    "2022_Garcia-Knight_infectious_period": "data",
    "2022_Guo_k": "review",
    "2022_Hart_latent_period": "data",
    "2022_Heiden_and_Buchholz_serial_interval": "data",
    "2022_Kremer_serial_interval": "data",
    "2022_Liu_r0_basic_": "review",
    "2022_Ryu__k": "data",
    "2022_Wu_incubation_period": "review", # PDF WAS WRONG (only supp was available)
    "2022_Zhao_k": "data",
    "2023_Galmiche_incubation_period": "data",
    "2023_Xu_incubation_period": "review",
    "2023_Yuan_cfr": "review",
    "2023_Zeng_serial_interval": "data",
    "2023_Zhang_and_Nishiura__ifr": "data",
    "2024_Ahmad__cfr": "review",
    "2024_Li__incubation_period": "data",
    "2024_Ward_ifr": "data"
}

SELECTED_TYPE = "review"  # for development/testing

grobid_url="http://192.168.1.250:8070"

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    """Configuration for search strategies."""
    use_semantic_search: bool = True
    use_keyword_search: bool = True
    use_context_extraction: bool = True
    top_k_semantic: int = 20
    top_k_keyword: int = 10
    top_k_final: int = 30
    context_window_chars: int = 500
    similarity_threshold: float = 0.0

@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    embedding_model: str = "jinaai/jina-embeddings-v3"
    llm_model: str = "deepseek-r1:7b" # "qwen2.5vl:3b" #
    hyde_model: str = "tinyllama:1.1b"
    classifier_model: str = "qwen2.5vl:3b"
    use_hyde: bool = False
    max_workers: int = 2
    search: SearchConfig = field(default_factory=SearchConfig)


from episcope.utils.data_blueprints import PaperType
from typing import Dict, List

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

