from .schemas import (
    PaperType, DataAccessibility, GeoRegion, DataType,
    ClassificationOutput, PaperTypeClassificationOutput,
    DataAccessibilityClassificationOutput, GeoClassificationOutput,
    DataTypeClassificationOutput
)
from typing import Dict, List, Any
from dataclasses import dataclass, field

@dataclass
class BaseClassifierConfig:
    """Base configuration for a paper classifier."""
    similarity_threshold: float = 0.75
    top_k: int = 10
    # embedding_model: str = "jinaai/jina-embeddings-v3"
    template_paragraphs: Dict[str, List[str]] = field(default_factory=dict)
    classification_mapping: Dict[str, Any] = field(default_factory=dict)
    category_labels: Dict[str, str] = field(default_factory=dict)
    system_prompt: str = ""
    user_prompt_template: str = ""
    output_schema: Any = ClassificationOutput
    default_classification: Any = None
    max_validation_retries: int = 1
    extra_output_fields: Dict[str, Any] = field(default_factory=dict)




@dataclass
class PaperTypeClassifierConfig(BaseClassifierConfig):
    """Configuration for the paper type classifier."""
    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "literature_review": [
            "Several studies have examined the relationship between X and Y. Smith et al. (2020) found significant associations, while Jones et al. (2021) reported mixed results. A systematic review by Brown et al. (2019) identified 45 relevant studies.",
            "We conducted a systematic literature search across PubMed, Embase, and Web of Science databases. Studies were included if they met inclusion criteria. Two reviewers independently screened titles and abstracts.",
            "The existing literature can be categorized into three main approaches. Meta-analyses consistently show effect sizes ranging from 0.2 to 0.6. Previous reviews have identified several gaps in the literature.",
            "This scoping review aims to map the available evidence on X. We searched five databases and included 127 studies. The review follows PRISMA guidelines for systematic reviews."
        ],
        "data_analysis": [
            "We analyzed data from the National Health Survey (n=15,432 participants). Data collection occurred between January 2020 and December 2022. Statistical analyses were performed using R version 4.2.",
            "The dataset contained 23,891 observations across 15 variables. Missing data patterns were examined using multiple imputation. Primary outcomes were measured using validated instruments.",
            "Participants were recruited through stratified random sampling. Data were collected through structured interviews. The final analytic sample included 8,734 individuals after exclusions.",
            "Descriptive statistics were calculated for all variables. Logistic regression models were fitted with adjustment for confounders. Effect sizes and 95% confidence intervals are reported."
        ],
    })
    classification_mapping: Dict[str, PaperType] = field(default_factory=lambda: {
        "A": PaperType.LITERATURE_REVIEW,
        "B": PaperType.DATA_ANALYSIS,
        "C": PaperType.UNCLEAR
    })
    category_labels: Dict[str, str] = field(default_factory=lambda: {
        "A": "Literature Review",
        "B": "Data Analysis",
        "C": "Unclear"
    })
    system_prompt: str = """You are an senior epidemiologist doing reviewing the recent literatures. You are sorting your papers into {n_categories} categories: {category_labels}."""
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

    **Schema:**
    {schema}    

    **Instructions:**
    1. Analyze the evidence to determine the paper's main contribution.
    2. Select a single letter that best represents the paper's primary classification.
    3. Return a single JSON object adhering to the schema. Do not add extra text.
    """
    output_schema: Any = PaperTypeClassificationOutput
    default_classification: Any = PaperType.UNCLEAR







from .schemas import DATA_ACCESS_CODE_TO_ENUM, DATA_ACCESS_CODE_LABELS

@dataclass
class DataAccessibilityClassifierConfig(BaseClassifierConfig):
    """Configuration for classifying data accessibility."""

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "open_access": [
            "All data and code used in this study are publicly available at [URL].",
            "The dataset can be downloaded from the public GitHub repository.",
            "Data are openly available in a public repository without restrictions.",
        ],
        "upon_request": [
            "Data are available from the corresponding author upon reasonable request.",
            "Access to the data requires a data use agreement and approval from the data custodian.",
            "Due to privacy regulations, data access is restricted to qualified researchers who apply.",
        ],
        "not_available": [
            "The data that support the findings of this study are not publicly available "
            "due to legal or ethical restrictions.",
            "The data are proprietary and cannot be shared.",
            "Data sharing is not applicable as no dataset was created or analyzed.",
        ],
        "not_stated": [
            "No explicit data availability statement was provided.",
            "The manuscript does not specify whether the data can be accessed.",
        ],
    })

    classification_mapping: Dict[str, DataAccessibility] = field(
        default_factory=lambda: DATA_ACCESS_CODE_TO_ENUM.copy()
    )

    category_labels: Dict[str, str] = field(
        default_factory=lambda: DATA_ACCESS_CODE_LABELS.copy()
    )

    system_prompt: str = (
        "You are a data librarian classifying research papers based on the accessibility of the "
        "dataset(s) actually used in the study. You must distinguish between: "
        "OPEN_ACCESS (public repository, no permission), UPON_REQUEST (available but restricted), "
        "NOT_AVAILABLE (cannot be shared or no dataset), and NOT_STATED (nothing is said). "
        "Do NOT guess accessibility when it is not described."
    )

    user_prompt_template: str = """
You are a data librarian. Your task is to classify **how the dataset(s) USED in the study are made available**, if at all.

**Categories (A–D):**
{categories}

Definitions:
- **A – OPEN_ACCESS**: The dataset used in the analysis is available in a public repository or URL,
  and can be accessed without individual permission (no request, no approval, no DUA).
- **B – UPON_REQUEST**: The dataset can be accessed, but **only** if the reader contacts the authors
  or applies to a data custodian (e.g. "available upon reasonable request", "requires data use agreement").
- **C – NOT_AVAILABLE**: The dataset cannot be shared (proprietary, legal/ethical limits) or there is
  explicitly no sharable dataset (e.g. purely synthetic example data are not shared, or no data were used).
- **D – NOT_STATED**: The text does not explain whether or how the dataset is available.

IMPORTANT:
- Focus only on the **dataset(s) actually analyzed** in the study.
- If there is **no statement at all** about data availability, choose **D (NOT_STATED)**, not C.
- Use **C (NOT_AVAILABLE)** only when the text clearly says the data cannot be shared or there is no dataset.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts (candidate evidence):**
{chunks_info}

**Instructions:**
1. Look for explicit statements about data availability, repositories, URLs, or conditions for access.
2. If you find multiple statements, base your decision on the main dataset used in the analysis.
3. Choose exactly one letter (A, B, C, or D) according to the definitions above.
4. Return a single JSON object that exactly matches the schema below. Do NOT include any extra text.

**Schema:**
{schema}
"""

    output_schema: Any = DataAccessibilityClassificationOutput

    # UNCLEAR is reserved as an internal fallback if parsing fails etc.
    default_classification: Any = DataAccessibility.UNCLEAR



# @dataclass
# class DataAccessibilityClassifierConfig(BaseClassifierConfig):
#     """Classify data accessibility of the dataset(s) used in the study."""

#     template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
#         "open_access": [
#             "All data and code used in this study are publicly available at [URL].",
#             "The dataset can be downloaded from the public GitHub repository.",
#         ],
#         "restricted_access": [
#             "Data are available from the authors upon reasonable request.",
#             "Access to the data requires a data use agreement.",
#         ],
#         "not_available": [
#             "The data that support the findings of this study are not publicly available.",
#             "The data are proprietary and cannot be shared.",
#         ],
#         "unclear": [
#             "Data availability was not explicitly discussed.",
#             "No data accessibility statement was provided.",
#         ],
#     })

#     classification_mapping : Dict[str, DataType] = field(default_factory=lambda: DATA_ACCESS_CODE_TO_ENUM.copy())
#     category_labels : Dict[str, str] = field(default_factory=lambda: DATA_ACCESS_CODE_LABELS.copy())

#     system_prompt: str = (
#         "You are a data librarian classifying papers based on the accessibility of the data "
#         "actually used in the study. Your task is to decide whether the data are open, "
#         "restricted, not available, or unclear."
#     )

#     user_prompt_template: str = """
# You are a data librarian. Determine the **accessibility** of the dataset(s) used in the study.

# **Categories (A–D):**
# {categories}

# Guidance:
# - **A: Open Access** — data are publicly available at a URL or repository with no approval needed.
# - **B: Restricted Access** — data are only available via request, approval, DUA, or secure access.
# - **C: Not Available** — data explicitly cannot be shared OR no dataset exists.
# - **D: Unclear** — no information is provided about data sharing.

# Focus ONLY on the accessibility of the dataset(s) USED in the analysis.

# **Paper Content**
# Title: {title}
# Abstract: {abstract}
# Keywords: {keywords}

# **Relevant Extracts:**
# {chunks_info}

# **Instructions**
# 1. Identify statements about sharing of the data used.
# 2. Choose one letter (A–D).
# 3. Return JSON exactly matching the schema below.

# **Schema**
# {schema}
# """

#     output_schema: Any = DataAccessibilityClassificationOutput
#     default_classification: Any = DataAccessibility.UNCLEAR






from .schemas import DATA_TYPE_CODE_TO_ENUM, DATA_TYPE_CODE_LABELS

@dataclass
class DataTypeClassifierConfig(BaseClassifierConfig):
    """
    Configuration for classifying the type(s) of data used in an
    epidemiological research paper (traditional, non-traditional subtypes,
    synthetic, no data, unclear).
    """

    # Template paragraphs for retrieval: examples of each data type
    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        # --- Traditional data ---
        "traditional": [
            # classic surveillance + survey examples
            "We analyzed routinely collected case notifications from the national surveillance system.",
            "We used individual-level electronic health records from a network of hospitals.",
            "Survey data on health behaviours were collected using a standardized questionnaire "
            "administered to a representative sample of the population.",
            "We linked administrative hospital discharge data with vital statistics to estimate mortality.",
        ],

        # --- Non-traditional health data ---
        # (Digital patient data, symptom apps, wastewater, wearables, etc.)
        "non_traditional_health": [
            "Self-reported COVID-19 symptoms were collected using a mobile application deployed "
            "at national scale.",
            "We used aggregated data from an online symptom checker platform to detect emerging hotspots.",
            "Wastewater samples from treatment plants were analyzed to quantify SARS-CoV-2 viral load.",
            "Continuous heart rate and activity data from wearable devices were used to identify "
            "abnormal physiological patterns consistent with infection.",
            "Digital patient data from multiple hospital EHR systems were harmonized to build a large "
            "COVID-19 cohort for risk factor analysis.",
        ],

        # --- Non-traditional mobility & geolocation data ---
        # (Telecom CDRs, GPS, SDK data, Bluetooth, etc.)
        "non_traditional_mobility": [
            "Anonymized mobile phone call detail records were used to estimate population movements "
            "between regions.",
            "We used GPS traces from a location-based smartphone application to reconstruct mobility networks.",
            "Aggregated mobility indicators provided by a telecom operator were used to evaluate "
            "compliance with lockdown measures.",
            "Bluetooth-based digital contact tracing data were used to identify close contacts and "
            "estimate the effective reproduction number.",
        ],

        # --- Non-traditional economic data ---
        # (Card transactions, supply chain, open contracting, etc.)
        "non_traditional_economic": [
            "Aggregated credit and debit card transaction data were used as a proxy for local economic activity.",
            "Supply-chain shipment data were analyzed to assess the impact of COVID-19 on the distribution "
            "of essential goods.",
            "Open contracting data on emergency procurement were used to analyze government spending patterns.",
        ],

        # --- Non-traditional sentiment data ---
        # (Social media, crowdsourced sentiments, etc.)
        "non_traditional_sentiment": [
            "We collected tweets mentioning COVID-19-related keywords and analyzed their content to assess "
            "public sentiment towards non-pharmaceutical interventions.",
            "We downloaded Facebook posts and comments from public pages and used them to quantify sentiment "
            "towards vaccination campaigns.",
        ],

        # --- Synthetic data ---
        "synthetic": [
            "We generated a synthetic population using an agent-based model to simulate disease transmission.",
            "A fully synthetic dataset was created to mimic the distribution of real patient records "
            "while preserving privacy.",
            "Synthetic mobility trajectories were simulated based on a metapopulation model to evaluate "
            "alternative intervention scenarios.",
        ],

        # --- No empirical data ---
        "no_empirical_data": [
            "This work is a narrative review of existing literature and does not include new data analysis.",
            "We present a conceptual framework for pandemic preparedness without analyzing empirical data.",
            "The paper discusses ethical considerations and policy implications of data use in pandemics, "
            "but no dataset was collected or analyzed.",
        ],

        # # --- Unclear / not specified ---
        # "unclear": [
        #     "The authors describe using data from public sources, but the exact type of data is not specified.",
        #     "The methods section refers to 'available datasets' without sufficient detail to determine "
        #     "whether the data are traditional or non-traditional.",
        # ],
    })

    # Mapping from letter codes to enum values
    classification_mapping: Dict[str, DataType] = field(
        default_factory=lambda: DATA_TYPE_CODE_TO_ENUM.copy()
    )

    # Human-readable labels
    category_labels: Dict[str, str] = field(
        default_factory=lambda: DATA_TYPE_CODE_LABELS.copy()
    )

    system_prompt: str = (
        "You are an epidemiologist and data scientist classifying research papers by the "
        "type(s) of data used as INPUT to the study’s analysis. "
        "You ONLY care about datasets that are actually collected, analyzed, or modeled in the study. "
        "You must IGNORE: (a) data sources mentioned only as background or motivation, "
        "(b) topics or themes discussed in the text, and (c) auxiliary processes such as "
        "laboratory confirmation of cases or standard clinical diagnostics that merely define outcomes. "
        "Prefer fewer labels over more: only assign a data type if there is clear evidence that "
        "this type of data was used in the main analysis or in a core secondary analysis."
    )


    user_prompt_template: str = """
    You are an epidemiologist and data scientist. Your task is to classify the type(s) of data
    USED in an academic paper, with a focus on epidemiological and public health research.

    **Key Principle**

    Only label **data that are actually used as input to the study’s analysis**.
    This means data that are collected, assembled, or modeled and then analyzed in the paper.

    Do NOT assign a label based on:
    - Data sources that are mentioned only as examples, motivation, or related work.
    - Topics or themes that are discussed (e.g. “social media misinformation”) without actually
    collecting or analyzing that data.
    - Auxiliary processes that define or verify the outcome, such as:
    - “laboratory-confirmed cases”
    - “PCR confirmation”
    - “serological confirmation”
    These count as HOW the outcome is measured, not as a separate data type.

    **Categories (multi-label, A–H):**
    {categories}

    - A: Traditional data – classical epidemiological data such as surveillance case notifications,
        electronic health records, clinical registries, administrative health data, and structured surveys
        that are used in the analysis.
    - B: Non-Traditional Health Data – digitally captured health-related data such as digital patient data,
        symptom apps, online symptom checkers, wearable and biometric data, or wastewater surveillance
        that are used in the analysis.
    - C: Non-Traditional Mobility Data – data capturing movements or physical proximity, such as telecom
        CDRs, GPS traces, SDK-derived mobility indicators, and Bluetooth-based contact tracing, used
        as inputs to the analysis.
    - D: Non-Traditional Economic Data – data reflecting economic activity, such as card transactions,
        supply-chain and shipping data, or open contracting data, used as inputs to the analysis.
    - E: Non-Traditional Sentiment Data – social media data or other crowdsourced data that capture
        attitudes, perceptions, or emotions, and are actually collected/processed in the study.
    - F: Synthetic Data – fully or largely simulated data (e.g. synthetic patients, synthetic mobility
        traces, agent-based simulations) that form a dataset used in the analysis (even if they are
        used only for model testing or scenario analysis).
    - G: No Empirical Data – conceptual, theoretical, policy, or methodological work where no empirical
        dataset is collected or analyzed.
    - H: Unclear / Not Specified – insufficient information to determine the data type(s) used.

    **Paper Content:**
    Title: {title}
    Abstract: {abstract}
    Keywords: {keywords}

    **Relevant Extracts (candidate evidence):**
    {chunks_info}

    **Instructions:**

    1. Carefully read the title, abstract, keywords, and the extracted snippets.

    2. For each category A–F, only assign the label if there is **explicit or very strong implicit evidence**
    that this type of data is actually used in the analysis. Look for verbs such as:
    - "we collected", "we assembled", "we used", "we analyzed", "we obtained", "we retrieved",
        "we constructed a dataset of", "data were extracted from".
    Mentions like "Twitter is an important data source" or "lab-confirmed cases" WITHOUT an indication
    that Twitter data (or lab data) were collected and analyzed as a dataset should NOT trigger a label.

    3. Distinguish between:
    - A paper discussing a data source in general (NO label).
    - A paper actually using that data source in an analysis (label).

    4. Prefer **fewer labels over more**:
    - Start from the assumption of NO label.
    - Add a label only when you see concrete evidence that the data type is used.
    - If in doubt between “included” vs “not included”, choose “not included”.
    - Use label G only when you are confident that no empirical data are analyzed.
    - Use label H if the description is too vague to decide what was actually used.

    5. Return a JSON object that strictly follows the schema below. Do NOT include any extra text.

    **Schema:**
    {schema}
    """

    output_schema: Any = DataTypeClassificationOutput
    # if we give a simple list dataclass stops up since it leads all instances to the same (mutable) object
    default_classification: List[DataType] = field(
        default_factory=lambda: [DataType.UNCLEAR]  
    )








from .schemas import GEO_CODE_TO_ENUM, GEO_CODE_LABELS


@dataclass
class GeoClassifierConfig(BaseClassifierConfig):
    """
    Configuration for classifying the geographic origin of the data used.

    Multi-label at continent level (A–F), plus G=synthetic, H=unclear,
    with fine-grained 'locations' (countries and cities/regions).
    """

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "geo_origin": [
            "We analyzed COVID-19 case data from hospitals in Italy.",
            "Data were obtained from national surveillance systems in Brazil.",
            "We used electronic health records from a large hospital network in the United States.",
            "Participants were recruited from multiple regions in China.",
            "Data came from primary care practices in the United Kingdom.",
            "The dataset includes observations from several European countries.",
            "We generated a fully synthetic dataset to simulate the spread of infection.",
        ],
    })

    classification_mapping: Dict[str, GeoRegion] = field(
        default_factory=lambda: GEO_CODE_TO_ENUM.copy()
    )

    category_labels: Dict[str, str] = field(
        default_factory=lambda: GEO_CODE_LABELS.copy()
    )

    system_prompt: str = (
        "You are a research analyst determining the geographic origin of the data used in "
        "epidemiological studies. Your goal is to identify where the **analyzed dataset(s)** "
        "come from, not where the authors work or which places are mentioned in general.\n\n"
        "You MUST distinguish between:\n"
        "- Locations that are the source of the data analyzed (e.g. 'we used case data from Italy'), and\n"
        "- Locations mentioned only in background, examples, comparisons, citations, affiliations, or discussion.\n\n"
        "Only locations that clearly refer to the origin of the data used in the analysis should be extracted. "
        "Prefer missing a location over including one that is only mentioned in passing and not actually used."
    )

    user_prompt_template: str = """
You are a research analyst. Your task is to identify the **geographic origin of the data USED** in the study.

You must do two things:

1. **Extract locations (data origin only)**:
   - Collect country names that clearly refer to where the **analyzed dataset(s)** come from and place them in `extras.countries`.
   - Collect cities/regions/states/provinces for the analyzed data and place them in `extras.cities`.
   - Ignore locations from author affiliations, institutional addresses, background context, examples, or citations.
   - Only include a location if data were **collected from**, **obtained from**, or **analyzed for** that place.

2. **Classify by region (multi-label)**:
   - Use one or more of the continent/region letters (A–F) if data are from those continents.
   - Use **G (Synthetic)** if the data are purely synthetic.
   - Use **H (Unclear)** if you cannot infer any geographic origin.
   - If data originate from multiple continents, include **all relevant letters**.

**Categories (A–H):**
{categories}

Examples (what to include vs ignore):
- "We analyzed COVID-19 case data from hospitals in Italy and Spain." →
    - `classification` = ["C"]
    - `extras` = {{"countries": ["Italy", "Spain"], "cities": []}}
- "Data were obtained from national surveillance in Brazil and Argentina." →
    - `classification` = ["E"]
    - `extras` = {{"countries": ["Brazil", "Argentina"], "cities": []}}
- "Data were from the US, Italy, and Brazil." →
    - `classification` = ["D", "C", "E"]
    - `extras` = {{"countries": ["United States", "Italy", "Brazil"], "cities": []}}
- "We simulated a synthetic population of 1 million individuals." →
    - `classification` = ["G"]
    - `extras` = {{"countries": [], "cities": []}}

**Important rules:**
- The **`classification` field is a list of letters**. Include **all continents** represented.
- Do **NOT** collapse multi-continent data into "unclear".
- `extras.countries` and `extras.cities` must contain **only locations that are sources of the analyzed data**.
  If none are described, return empty lists.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts (candidate evidence):**
{chunks_info}

**Instructions:**
1. Read the title, abstract, and extracts, focusing only on where the **data used in the analysis** come from.
2. Fill the `extras.countries` field with all country names that are clearly data origins.
3. Fill the `extras.cities` field with city/region/state names that are clearly data origins.
4. Build `classification` as a list of letters (A–H) using the rules above.
5. Optionally set `primary_label` if one continent clearly dominates; otherwise, leave it null.
6. Return a single JSON object that strictly follows the schema below. Do NOT include any extra text.

**Schema:**
{schema}
"""

    output_schema: Any = GeoClassificationOutput

    default_classification: Any = field(
        default_factory=lambda: [GeoRegion.UNCLEAR]
    )


# @dataclass
# class GeoClassifierConfig(BaseClassifierConfig):
#     """
#     Configuration for classifying the geographic origin of the data used.

#     Multi-label at continent level (A–F), plus G=synthetic, H=unclear,
#     with fine-grained 'locations' (countries and cities/regions).
#     """

#     # Single bin of templates that fish for geo-origin sentences
#     template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
#         "geo_origin": [
#             "We analyzed COVID-19 case data from hospitals in Italy.",
#             "Data were obtained from national surveillance systems in Brazil.",
#             "We used electronic health records from a large hospital network in the United States.",
#             "Participants were recruited from multiple regions in China.",
#             "Data came from primary care practices in the United Kingdom.",
#             "The dataset includes observations from several European countries.",
#             "We generated a fully synthetic dataset to simulate the spread of infection.",
#         ],
#     })

#     classification_mapping: Dict[str, GeoRegion] = field(
#         default_factory=lambda: GEO_CODE_TO_ENUM.copy()
#     )

#     category_labels: Dict[str, str] = field(
#         default_factory=lambda: GEO_CODE_LABELS.copy()
#     )

#     system_prompt: str = (
#         "You are a research analyst determining the geographic origin of the data used in epidemiological "
#         "studies. You must (1) extract concrete geographic names (countries, cities, regions), and "
#         "(2) classify the data origin at the continent level. The classification can contain multiple "
#         "continents when data come from more than one region. Synthetic data and unclear cases are "
#         "handled with explicit labels."
#     )

#     user_prompt_template: str = """
# You are a research analyst. Your task is to identify the **geographic origin of the data USED** in the study.

# You must do two things:

# 1. **Extract locations**:
#    - Collect all country names that clearly refer to where the analyzed dataset comes from.
#    - Collect all cities/regions/states/provinces that are clearly locations of the analyzed data.
#    - Ignore author affiliations and places mentioned only in background or examples.

# 2. **Classify by region (multi-label)**:
#    - Use one or more of the continent/region letters (A–F) if data are from those continents.
#    - Use **G (Synthetic)** if the data are purely synthetic (no real geographic origin).
#    - Use **H (Unclear)** only if you truly cannot infer any geographic origin.

# **Categories (A–H):**
# {categories}

# Examples:
# - Data from Italy and Spain → classification includes **C (Europe)**; locations.countries include ["Italy", "Spain"].
# - Data from the US and Canada → classification includes **D (North America)**; locations.countries include ["United States", "Canada"].
# - Data from China and India → classification includes **B (Asia)**; locations.countries include ["China", "India"].
# - Data from Brazil and Argentina → classification includes **E (South America)**.
# - Data from multiple continents (e.g. USA, Italy, Brazil) → classification includes **all relevant letters**:
#   D (North America), C (Europe), E (South America).
# - Purely simulated population with no real-country data → classification = ["G"] and locations lists are empty.

# **Important rules:**
# - The **classification field is a list of letters**. Include **all continents** represented in the data used.
# - Do **NOT** collapse multi-continent data into "unclear". Return all applicable continent letters instead.
# - Only use **H (Unclear)** if you cannot identify any geographic origin from the text.
# - Always set `locations["countries"]` and `locations["cities"]`. If none are mentioned, return empty lists.

# **Paper Content:**
# Title: {title}
# Abstract: {abstract}
# Keywords: {keywords}

# **Relevant Extracts (candidate evidence):**
# {chunks_info}

# **Instructions:**
# 1. Read title, abstract, and extracts, focusing on where the data used in the analysis come from.
# 2. Fill `extra_output_fields["countries"]` with all country names in the data origin description.
# 3. Fill `extra_output_fields["cities"]` with city/region/state names when present.
# 4. Build `classification` as a list of letters (A–H) using the rules above.
# 5. Optionally set `primary_label` if one region clearly dominates; otherwise, leave it null.
# 6. Return a single JSON object that strictly follows the schema below. Do NOT include extra text.

# **Schema:**
# {schema}
# """

#     extra_output_fields: Dict[str, Any] = field(
#         default_factory=lambda: {
#             "countries": "list of countries the data come from",
#             "cities": "list of cities/regions/states the data come from",
#         }
#     )

#     output_schema: Any = GeoClassificationOutput

#     # Default internal fallback (e.g. if parsing fails):
#     default_classification: Any = field(
#         default_factory=lambda: [GeoRegion.UNCLEAR]
#     )


