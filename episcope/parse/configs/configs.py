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
    model_name: str = "allenai-specter"
    llm_model: str = "qwen2.5vl:3b" #"deepseek-r1:7b"
    hyde_model: str = "tinyllama:1.1b"
    classifier_model: str = "qwen2.5vl:3b"
    use_hyde: bool = True
    max_workers: int = 2
    search: SearchConfig = field(default_factory=SearchConfig)