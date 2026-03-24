from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schemas import (
    # Enums
    PaperType, DataAccessibility, GeoRegion, DataType,
    # Output schemas
    BaseClassificationSchema, PaperTypeClassificationOutput,
    DataAccessibilityClassificationOutput, GeoClassificationOutput,
    DataTypeClassificationOutput,
)

"""
Classifier configuration objects.

These configs are intended to be:
- Prompt-canonical: each prompt definition should reflect the formal labeling protocol.
- Schema-canonical: each config must point to the Pydantic schema that validates the LLM output.
- Retrieval-friendly: template_paragraphs should contain short, realistic snippets that maximize recall.

IMPORTANT:
- The labeling protocol (GEO/DAVAIL/DTYPE) is enforced primarily via prompts and via schema validators
  in schemas.py. Keep the two in sync.
"""


@dataclass
class BaseClassifierConfig:
    """Base configuration for a paper classifier."""
    similarity_threshold: float = 0.75
    top_k: int = 10

    template_paragraphs: Dict[str, List[str]] = field(default_factory=dict)
    cr_template_sentences: Dict[str, List[str]] = field(default_factory=dict)
    classification_mapping: Dict[str, Any] = field(default_factory=dict)
    category_labels: Dict[str, str] = field(default_factory=dict)

    system_prompt: str = ""
    user_prompt_template: str = ""

    output_schema: Any = BaseClassificationSchema
    default_classification: Any = None

    # How many times to re-ask the LLM if JSON does not validate.
    max_validation_retries: int = 3

    # Extra output fields (legacy; prefer Pydantic schema fields).
    extra_output_fields: Dict[str, Any] = field(default_factory=dict)



# -----------------------------------------------------------------------------
#  PAPER TYPE (parameter-estimation focused taxonomy)
# -----------------------------------------------------------------------------

# from .schemas import _CODE_TO_ENUM, _CODE_LABELS, _CODE_DEFINITIONS


# @dataclass
# class PaperTypeClassifierConfig(BaseClassifierConfig):
#     """Configuration for the parameter-estimation focused paper-type classifier ()."""

#     template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
#         "primary_empirical": [
#             "We analyzed patient-level line-list data from the first 250 confirmed cases to estimate the incubation period.",
#             "Using contact tracing records from the outbreak investigation, we estimated the secondary attack rate and generation time.",
#             "We followed a cohort of exposed individuals and computed incidence and case fatality risk over 30 days.",
#         ],
#         "modeling_inference": [
#             "We fit a renewal model to incidence data to infer time-varying R_t under different reporting assumptions.",
#             "An SEIR model was calibrated to hospitalization and death time series to back-calculate R_0 via maximum likelihood.",
#             "We used Bayesian MCMC to infer latent infection dynamics and estimate the infection fatality ratio (IFR).",
#         ],
#         "meta_analysis": [
#             "We conducted a systematic review and random-effects meta-analysis to obtain a pooled estimate of the serial interval.",
#             "A PRISMA-guided meta-analysis aggregated incubation period estimates across 32 studies.",
#             "We pooled vaccine effectiveness estimates across observational studies using inverse-variance weighting.",
#         ],
#         "methodological": [
#             "We propose a new estimator for R_t from wastewater measurements and validate it on historical outbreaks.",
#             "We introduce an open-source software package to estimate transmission parameters from incidence data.",
#             "We compare two competing methods for estimating the incubation period and quantify bias under censoring.",
#         ],
#         "review_commentary": [
#             "We review published estimates of R_0 and discuss why early estimates may be biased upward.",
#             "This narrative review summarizes evidence on incubation periods and implications for quarantine policy.",
#             "We provide a perspective on interpreting case fatality rates during ongoing outbreaks.",
#         ],
#     })

#     classification_mapping: Dict[str, PaperType] = field(default_factory=lambda: dict(_CODE_TO_ENUM))
#     category_labels: Dict[str, str] = field(default_factory=lambda: dict(_CODE_LABELS))

#     system_prompt: str = (
#         "You are a precise epidemiological methods expert. "
#         "Classify the paper into exactly one  category based only on the provided text."
#     )

#     user_prompt_template: str = r"""
# You are a senior academic epidemiologist. Determine the primary * paper type*.

# **Categories (choose ONE):**
# A. Primary_Empirical - original patient/field/surveillance/outbreak data used to estimate a parameter.
# B. Modeling_Inference - parameters inferred from existing data via mathematical/statistical models.
# C. Meta_Analysis - systematic review with quantitative pooling of parameter estimates.
# D. Methodological - new/validated estimation methods and/or software/tools for parameter estimation.
# E. Review_Commentary - narrative reviews, perspectives, editorials without new data or pooled estimates.
# F. Unclear - not enough information in the excerpt to decide (use sparingly).

# **Definitions:**
# {definitions}

# **Paper Content:**
# Title: {title}
# Abstract: {abstract}
# Keywords: {keywords}

# **Relevant Extracts (candidate evidence):**
# {chunks_info}

# **Instructions:**
# 1. Focus on *how the parameter estimate is produced* (new raw data vs inference vs synthesis vs method/tool vs commentary).
# 2. Select exactly one letter (A-F).
# 3. Return a single JSON object matching the schema below. Do NOT include extra text.

# **Schema:**
# {schema}
# """

#     # Prompt-time helper: definitions are injected by the caller when formatting user_prompt_template.
#     extra_output_fields: Dict[str, Any] = field(default_factory=lambda: {
#         "definitions": "\n".join([f"{k}. {v}" for k, v in _CODE_DEFINITIONS.items() if k != "F"])
#     })

#     output_schema: Any = PaperTypeClassificationOutput
#     default_classification: Any = field(default_factory=lambda: [PaperType.UNCLEAR])
# -----------------------------------------------------------------------------
# Data Availability
# -----------------------------------------------------------------------------

from .schemas import DATA_ACCESS_CODE_TO_ENUM, DATA_ACCESS_CODE_LABELS, DATA_ACCESS_CODE_DEFINITIONS

@dataclass
class DataAccessibilityClassifierConfig(BaseClassifierConfig):
    """Configuration for classifying data availability (DAVAIL)."""

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        # OPEN
        "open": [
            "All data used in this study are available in a public repository (DOI/URL provided).",
            "The dataset is deposited on Zenodo under DOI: 10.xxxx/zenodo.xxxxx.",
            "De-identified data and analysis code are provided as supplementary CSV files.",
            # Paper-as-release (protocol: still OPEN)
            "We report a line list of confirmed cases and provide the full dataset in Table S1.",
        ],
        # REPORTED
        "reported": [
            "We provide stratified case counts by age group and week in Table 2, but do not release the underlying dataset.",
            "Table S3 reports detailed frequency tables that allow recomputation of key rates, but no repository link is provided.",
            "The manuscript includes partial time-series points sufficient to re-aggregate trends, without providing the full dataset.",
        ],
        # AVAILABLE_UPON_REQUEST
        "upon_request": [
            "Data are available from the corresponding author upon reasonable request.",
            "Access requires signing a data use agreement and approval by the data custodian.",
            "Researchers may apply to the institutional data access committee for permission.",
        ],        
        # REFERENCED
        "referenced": [
            "We used case counts reported by the Ministry of Health, but provide no stable link or retrieval instructions.",
            "Data were taken from a named public dashboard, but the paper provides only derived metrics and no downloadable dataset or stable identifier.",
            "We extracted summary statistics from Smith et al. (#b12) without providing re-aggregatable tables or a stable access path to the underlying data.",
        ],
        # CLOSED
        "closed": [
            "The data cannot be shared due to legal and ethical restrictions.",
            "Patient-level data are confidential and not available for public release.",
            "The dataset is proprietary and cannot be made available.",
        ],
        # # NOT_STATED
        # "not_stated": [ 
        # ],
    })

    cr_template_sentences: Dict[str, List[str]] = field(default_factory=lambda: { # prototypical evidence sentences 
        # OPEN
        "open": [
            "The data are publicly available in a repository, supplement, paper package, or the full dataset is published in the paper itself."
        ],
        # REPORTED
        "reported": [
            "We report detailed counts or tables that allow the main results to be recomputed, but we do not release the full dataset."
        ],
        # AVAILABLE_UPON_REQUEST
        "upon_request": [
            "The data are available only on request, by author contact, committee approval, or an institutional access process."
        ],        
        # REFERENCED
        "referenced": [
                "We used data reported by a previous study or made available by a public health authority."
        ],
        # CLOSED
        "closed": [
            "The data cannot be shared because of legal, ethical, confidentiality, or licensing restrictions."
        ],
        # # NOT_STATED
        # "not_stated": [ 
        # ],
    })

    classification_mapping: Dict[str, DataAccessibility] = field(
        default_factory=lambda: DATA_ACCESS_CODE_TO_ENUM.copy()
    )

    category_labels: Dict[str, str] = field(
        default_factory=lambda: DATA_ACCESS_CODE_LABELS.copy()
    )

    system_prompt: str = (
        "You are a research data librarian classifying epidemiological papers by *data availability*. "
        "Follow the formal definitions strictly and base labels ONLY on explicit statements in the paper. "
        "Do not verify links or use external knowledge. "
        # "If the paper is itself the primary data release (outbreak report / line list / transmission chain), "
        # "use OPEN."
    )

    user_prompt_template: str = r"""
You are a research data librarian. Classify the *data availability* of the dataset(s) USED in the paper.

Key principle:
- Assign labels strictly from statements in the paper. Do not verify links or rely on outside knowledge.
- Extracts may indicate a citation via "Author et al." or "(\#bn)" where n is a number. #This counts as a reference.

**Categories (multi-label, A–F):**
{categories}

**Definitions:**
{definitions}

Multi-dataset papers:
- If the paper clearly uses multiple datasets with different availability mechanisms, you may include multiple letters.
- Do NOT add a label for a dataset mentioned only as background, comparison, or citation (not used in the methodology).

Ambiguity handling:
- If the paper provides granular, extractable values that satisfy the re-aggregation criterion but no stable public access path, prefer **B (REPORTED)**.
- If a specifically identifiable third-party source is named but no stable retrieval path is provided and reporting is data-thin, prefer **C (REFERENCED)**.
- If the attribution is vague (e.g., "government data") or there is no availability statement at all, choose **F (NOT_STATED)**.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts (candidate evidence):**
{chunks_info}

**Instructions:**
1. Identify datasets actually used in the study (methods/results), and what the paper says about accessing each.
2. Choose one or more letters (A–F) based on the definitions above.
3. Set `primary_label` only if one label clearly dominates; otherwise leave it null.
4. Return a single JSON object that exactly matches the schema below. Do NOT include any extra text.

**Schema:**
{schema}
"""

    extra_output_fields: Dict[str, Any] = field(default_factory=lambda: {
        "definitions": "\n".join([f"- **{k} – {v}**" for k, v in DATA_ACCESS_CODE_DEFINITIONS.items()])
    })

    output_schema: Any = DataAccessibilityClassificationOutput
    default_classification: Any = field(default_factory=lambda: [DataAccessibility.UNCLEAR])





# -----------------------------------------------------------------------------
# PAPER TYPE (parameter-estimation focused taxonomy; paper-level, primary + secondary)
# -----------------------------------------------------------------------------

from .schemas import PAPER_TYPE_DEFINITIONS


@dataclass
class PaperTypeClassifierConfig(BaseClassifierConfig):
    """Configuration for the parameter-estimation focused PAPER_TYPE classifier."""

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "EMPIRICAL": [
            "We report a prospective cohort study of exposed contacts and estimate the incubation period from observed symptom onset times.",
            "Using contact tracing line-list data collected during the outbreak, we estimate the secondary attack rate and serial interval.",
            "We describe a new surveillance dataset and analyze observed case notifications to estimate age-specific case fatality risk.",
        ],
        "INFERENCE": [
            "We fit a renewal model to incidence data to infer time-varying R_t under alternative reporting assumptions.",
            "An SEIR model was calibrated to hospitalization and death time series to estimate R_0 via maximum likelihood and Bayesian inference.",
            "We use phylodynamic modeling of viral sequences to reconstruct infection dynamics and estimate the effective population size over time.",
        ],
        "FORECAST": [
            "We project epidemic trajectories under alternative intervention scenarios and compare expected hospital demand over the next 12 weeks.",
            "We generate short-term forecasts of incident cases and evaluate predictive performance across multiple regions.",
            "We perform counterfactual simulations to quantify the expected impact of school closures on future transmission.",
        ],
        "SYNTHESIS": [
            "We conducted a systematic review (PRISMA) and meta-analysis to obtain pooled estimates of the serial interval across studies.",
            "We systematically searched the literature and synthesized published estimates of the infection fatality ratio using random-effects models.",
            "We appraised study quality and summarized parameter estimates across prior outbreaks using predefined inclusion criteria.",
        ],
        "METHODOLOGICAL": [
            "We propose a new estimator for generation time that corrects right truncation and assess identifiability under different sampling schemes.",
            "We develop an inference framework with bias/uncertainty quantification and provide open-source software to estimate R_t from incidence.",
            "We derive theoretical results about epidemic thresholds on networks and validate the method with illustrative simulations.",
        ],
        "COMMENTARY": [
            "We discuss challenges in interpreting early R_0 estimates and provide recommendations for reporting uncertainty and assumptions.",
            "This perspective reviews common pitfalls in parameter estimation and offers guidance for public health decision-making.",
            "We provide a narrative overview of evidence on transmission dynamics without presenting new quantitative estimates.",
        ],
        "OTHER": [
            "This paper focuses on virology and laboratory assays without estimating epidemiological parameters or proposing parameter-estimation methods.",
            "The work is primarily clinical case management guidance and does not include parameter estimation, forecasting, or evidence synthesis.",
        ],
    })

    classification_mapping: Dict[str, PaperType] = field(
        default_factory=lambda: {pt.value: pt for pt in PaperType}
    )

    category_labels: Dict[str, str] = field(
        default_factory=lambda: {pt.value: pt.value for pt in PaperType if pt != PaperType.UNCLEAR}
    )

    system_prompt: str = (
        "You are a precise epidemiological methods expert. "
        "Assign exactly one PRIMARY PAPER_TYPE label and, when warranted, any SECONDARY PAPER_TYPE labels. "
        "Base labels only on the provided text and do not over-label."
    )

    user_prompt_template: str = r"""
You are a senior academic epidemiologist. Classify the paper by its contribution type to epidemiological parameter estimation.

Epidemiological papers often combine tasks (data collection, parameter inference, and forecasting). To reflect this:
- Assign exactly **one primary_label** capturing the paper's main scientific claim or deliverable.
- Optionally assign **secondary_labels** for other substantial contributions that enable or accompany the primary contribution.
- The same label set is used for both primary and secondary contributions.

**PAPER_TYPE labels (for both primary and secondary):**
{definitions}

Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts (candidate evidence):**
{chunks_info}

**Instructions:**
1. Base your decision ONLY on the provided text (abstract/keywords/extracts). Do not use outside knowledge.
2. Choose exactly one `primary_label` from the label set above.
3. Add zero or more `secondary_labels` only if the paper makes substantial additional contributions. Do NOT include the primary label in `secondary_labels`.
4. Use `OTHER` as the primary label if none of the substantive categories apply; in that case leave `secondary_labels` empty.
5. Use `UNCLEAR` only if the excerpt is genuinely insufficient to classify; if used, set it as `primary_label` and leave `secondary_labels` empty.
6. Return a single JSON object that exactly matches the schema below. Do NOT include any extra text.

**Schema:**
{schema}
"""

    # Prompt-time helper: definitions are injected by the caller when formatting user_prompt_template.
    extra_output_fields: Dict[str, Any] = field(default_factory=lambda: {
        "definitions": "\n".join(
            [f"- {k}: {v}" for k, v in PAPER_TYPE_DEFINITIONS.items() if k not in {"UNCLEAR"}]
        )
    })

    output_schema: Any = PaperTypeClassificationOutput
    default_classification: Any = field(default_factory=lambda: [PaperType.UNCLEAR])





# -----------------------------------------------------------------------------
# DTYPE (Data Type) – protocol-aligned
# -----------------------------------------------------------------------------

from .schemas import DATA_TYPE_CODE_TO_ENUM, DATA_TYPE_CODE_LABELS, DATA_TYPE_CODE_DEFINITIONS


@dataclass
class DataTypeClassifierConfig(BaseClassifierConfig):
    """
    Configuration for classifying the type(s) of data used (DTYPE), aligned with the protocol:
    - TRADITIONAL
    - NON_TRADITIONAL_HEALTH
    - NON_TRADITIONAL_MOBILITY
    - NON_TRADITIONAL_SENTIMENT
    - NON_TRADITIONAL_ECONOMIC
    - SYNTHETIC
    - NO_EMPIRICAL_DATA
    """

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "traditional": [
            "We analyzed routinely collected case notifications from the national surveillance system.",
            "Hospital admissions and deaths were extracted from administrative health records.",
            "We used PCR-confirmed test line lists reported by regional health authorities.",
        ],
        "non_traditional_health": [
            "Self-reported symptoms were collected using a mobile application deployed at national scale.",
            "Wastewater samples were analyzed to quantify viral load as a proxy for community spread.",
            "Wearable-device heart rate data were used to detect anomalous physiological patterns.",
            "Digital patient data from a large-scale EHR-based research network were harmonized for analysis.",
        ],
        "non_traditional_mobility": [
            "Anonymized mobile phone call detail records were used to estimate population movements.",
            "GPS traces from a smartphone application were used to reconstruct mobility networks.",
            "Bluetooth-based proximity signals from digital contact tracing were used to infer contact patterns.",
        ],
        "non_traditional_sentiment": [
            "We collected tweets and analyzed content to assess public risk perception.",
            "Survey panel data tracked attitudes and behaviors in response to interventions.",
            "Crowdsourced reports of symptoms and perceptions were used to monitor behavioral changes.",
        ],
        "non_traditional_economic": [
            "Aggregated credit and debit card transaction data were used as a proxy for local economic activity.",
            "Supply-chain shipment data were analyzed to assess the impact of COVID-19 on the distribution "
            "of essential goods.",
            "Open contracting data on emergency procurement were used to analyze government spending patterns.",
        ],
        "synthetic": [
            "We simulated epidemic trajectories to evaluate policy scenarios and support the main conclusions.",
            "An agent-based model generated simulated outcomes used as evidence for intervention effects.",
        ],
        "no_empirical_data": [
            "We present a conceptual framework without analyzing empirical data.",
            "This paper provides a theoretical derivation and does not fit to data.",
        ],
    })

    classification_mapping: Dict[str, DataType] = field(default_factory=lambda: DATA_TYPE_CODE_TO_ENUM.copy())
    category_labels: Dict[str, str] = field(default_factory=lambda: DATA_TYPE_CODE_LABELS.copy())

    system_prompt: str = (
        "You are an epidemiologist classifying papers by the type(s) of data used as evidence (DTYPE). "
        "Only label data that are actually used as inputs to the paper's analysis. "
        "Prefer under-labeling to over-labeling: if evidence is weak, do not assign the label."
    )

    user_prompt_template: str = r"""
You are an epidemiologist. Classify the *type(s) of data used* (DTYPE) in the paper.

Key principles:
- Label only data that are actually used as inputs to the study's analysis (collected/assembled/modeled then analyzed).
- Ignore data sources mentioned only as background, motivation, or related work.

**Categories (multi-label, A–H):**
{categories}

**Definitions (protocol-aligned):**
{definitions}

**What counts as Non-Traditional Data (NTD):**
Non-traditional data are datasets that are not purpose-collected through classical epidemiological instruments (e.g., surveys, clinical exams, laboratory assays), but instead originate from digital systems, platforms, sensors, or operational processes and are repurposed as proxies for health, behavior, mobility, or economic activity.  
They are typically passively generated, high-volume, or platform-mediated, and were not originally designed for epidemiological research.  
Thematic content alone (e.g., “economic”, “behavioral”, “social”) does NOT make data non-traditional; the data-generation mechanism does.


**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Start from the assumption that no label applies; add a label only with clear evidence.
2. Label only based on data actually used as inputs to the analysis.
3. Use multiple labels only when multiple distinct data types are clearly used.
4. If you choose **G**, return only **G**.
5. If the paper is too vague, choose **H**.
6. Return a single JSON object matching the schema below, ouput an instance of the schema WITH VALUES, not the schema. Do NOT add extra text.
**Schema:**
{schema}
# Carefully populate at least all compulsory fields and as many optional fields as possible. Do not leave any field blank or null if you have information to populate it. Return a single JSON object matching the schema above. Do NOT add extra text.
"""

    extra_output_fields: Dict[str, Any] = field(default_factory=lambda: {
        "definitions": "\n".join([f'- **{k} – {v}**' for k, v in DATA_TYPE_CODE_DEFINITIONS.items()])
    })

    output_schema: Any = DataTypeClassificationOutput
    default_classification: Any = field(default_factory=lambda: [DataType.UNCLEAR])


# -----------------------------------------------------------------------------
# GEO (Geography) – protocol-aligned
# -----------------------------------------------------------------------------

from .schemas import GEO_CODE_TO_ENUM, GEO_CODE_LABELS, GEO_CODE_DEFINITIONS


@dataclass
class GeoClassifierConfig(BaseClassifierConfig):
    """
    Configuration for classifying the geographic scope (GEO) of the epidemiological evidence used.

    Output is multi-label at continent level (A–F), plus:
    - G = IRRELEVANT (e.g., strictly controlled laboratory setting, or no meaningful geographic origin stated)
    - H = UNCLEAR

    Additional extracted fields:
    - extras.countries
    - extras.cities
    """

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "geo_evidence": [
            "Cases were detected in Germany among travelers infected in Turkey.",
            "We analyzed an outbreak in hospitals in Italy and cases reported in Spain.",
            "The study used surveillance data from Brazil, Argentina, and Chile.",
            "Samples were collected in a controlled laboratory setting using laboratory-bred animals.",
            "This study provides a global overview using data from multiple continents.",
            "We analyzed data from Moscow and Saint Petersburg.",
            "Participants were recruited from Cairo and Alexandria.",
        ],
    })

    classification_mapping: Dict[str, GeoRegion] = field(default_factory=lambda: GEO_CODE_TO_ENUM.copy())
    category_labels: Dict[str, str] = field(default_factory=lambda: GEO_CODE_LABELS.copy())

    system_prompt: str = (
        "You are a research analyst assigning continent-level geographic labels (GEO) to epidemiological papers. "
        "The GEO label set is intended as a high-recall filter: include all locations explicitly tied to the "
        "epidemiological evidence used in the paper, even when multiple geographic perspectives coexist "
        "(e.g., location of exposure vs location of detection). "
        "Do not use external knowledge except to map a stated location (e.g., a country) to a continent."
    )

    user_prompt_template: str = r"""
You are a research analyst. Determine the *continent-level geography* (GEO) associated with the epidemiological evidence used in the paper.

What to include:
- Include every location that is **explicitly tied** to the epidemiological evidence/data used in the paper.
- If infections are acquired in one location but detected/reported in another, include **both** locations if both are tied to the evidence.

What to ignore:
- Ignore locations mentioned only as background, comparison, related work, author affiliations, or incidental discussion.
- Do NOT infer locations from strain identifiers, sample naming conventions, or external knowledge unless the paper explicitly states the location.
- Ignore the location where ONLY sequencing, PCR confirmation, or computational analysis happened. E.g., if samples were collected in country 1 but sequenced in country 2, consider **only country 1**. 
- Ignore the location where materials were produced if not tied to epidemiological evidence (e.g., lab-grown samples or materials). 

When GEO is IRRELEVANT:
- If the analysis is entirely in a strictly controlled laboratory setting (no meaningful geographic/ecological variation),
  use **G (IRRELEVANT)**.
- If the paper uses purely synthetic/no-geo data and does not meaningfully tie evidence to real-world locations, use **G (IRRELEVANT)**.

Global scope:
- If the paper explicitly states a global scope (e.g., “global”, “worldwide”), include **all continents** (A–F).

**Transcontinental mapping (UN M49 convention for reproducibility):**
{definitions}

**Categories (multi-label, A–H):**
{categories}

Output requirements:
1) Extract location names tied to evidence:
   - Put country names in `extras.countries`
   - Put cities/regions/states/provinces in `extras.cities`

2) Continent labels:
   - `classification` is a list of letters. Include all applicable continents (A–F), or G, or H.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Identify locations explicitly tied to the evidence/data used in the paper.
2. Map those locations to continents and build `classification` as a list of letters.
3. If no meaningful geo is stated or geo is irrelevant, use G or H as appropriate.
4. Optionally set `primary_label` if one continent clearly dominates; otherwise leave it null.
5. Return a single JSON object matching the schema below. Do NOT include extra text.

**Schema:**
{schema}
"""

    extra_output_fields: Dict[str, Any] = field(default_factory=lambda: {
        "definitions": "\n".join([
            "- Russia → Europe (C)",
            "- Turkey → Asia (B)",
            "- Egypt → Africa (A)",
            "- Kazakhstan → Asia (B)",
            "- Azerbaijan → Asia (B)",
            "- Georgia → Asia (B)",
            "- Cyprus → Asia (B)",
            "- Armenia → Asia (B)",
        ])
    })

    output_schema: Any = GeoClassificationOutput
    default_classification: Any = field(default_factory=lambda: [GeoRegion.UNCLEAR])
