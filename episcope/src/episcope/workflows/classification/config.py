from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schemas import (
    # Enums
    PaperType, DataAccessibility, GeoRegion, DataType,
    # Output schemas
    ClassificationOutput, PaperTypeClassificationOutput,
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
    classification_mapping: Dict[str, Any] = field(default_factory=dict)
    category_labels: Dict[str, str] = field(default_factory=dict)

    system_prompt: str = ""
    user_prompt_template: str = ""

    output_schema: Any = ClassificationOutput
    default_classification: Any = None

    # How many times to re-ask the LLM if JSON does not validate.
    max_validation_retries: int = 1

    # Extra output fields (legacy; prefer Pydantic schema fields).
    extra_output_fields: Dict[str, Any] = field(default_factory=dict)



# -----------------------------------------------------------------------------
#  PAPER TYPE (parameter-estimation focused taxonomy)
# -----------------------------------------------------------------------------

from .schemas import _CODE_TO_ENUM, _CODE_LABELS, _CODE_DEFINITIONS


@dataclass
class PaperTypeClassifierConfig(BaseClassifierConfig):
    """Configuration for the parameter-estimation focused paper-type classifier ()."""

    template_paragraphs: Dict[str, List[str]] = field(default_factory=lambda: {
        "primary_empirical": [
            "We analyzed patient-level line-list data from the first 250 confirmed cases to estimate the incubation period.",
            "Using contact tracing records from the outbreak investigation, we estimated the secondary attack rate and generation time.",
            "We followed a cohort of exposed individuals and computed incidence and case fatality risk over 30 days.",
        ],
        "modeling_inference": [
            "We fit a renewal model to incidence data to infer time-varying R_t under different reporting assumptions.",
            "An SEIR model was calibrated to hospitalization and death time series to back-calculate R_0 via maximum likelihood.",
            "We used Bayesian MCMC to infer latent infection dynamics and estimate the infection fatality ratio (IFR).",
        ],
        "meta_analysis": [
            "We conducted a systematic review and random-effects meta-analysis to obtain a pooled estimate of the serial interval.",
            "A PRISMA-guided meta-analysis aggregated incubation period estimates across 32 studies.",
            "We pooled vaccine effectiveness estimates across observational studies using inverse-variance weighting.",
        ],
        "methodological": [
            "We propose a new estimator for R_t from wastewater measurements and validate it on historical outbreaks.",
            "We introduce an open-source software package to estimate transmission parameters from incidence data.",
            "We compare two competing methods for estimating the incubation period and quantify bias under censoring.",
        ],
        "review_commentary": [
            "We review published estimates of R_0 and discuss why early estimates may be biased upward.",
            "This narrative review summarizes evidence on incubation periods and implications for quarantine policy.",
            "We provide a perspective on interpreting case fatality rates during ongoing outbreaks.",
        ],
    })

    classification_mapping: Dict[str, PaperType] = field(default_factory=lambda: dict(_CODE_TO_ENUM))
    category_labels: Dict[str, str] = field(default_factory=lambda: dict(_CODE_LABELS))

    system_prompt: str = (
        "You are a precise epidemiological methods expert. "
        "Classify the paper into exactly one  category based only on the provided text."
    )

    user_prompt_template: str = r"""
You are a senior academic epidemiologist. Determine the primary * paper type* for an epidemiological
parameter-estimation corpus (e.g., R0/Rt, incubation period, CFR/IFR, vaccine efficacy/effectiveness).

**Categories (choose ONE):**
A. Primary_Empirical - original patient/field/surveillance/outbreak data used to estimate a parameter.
B. Modeling_Inference - parameters inferred from existing data via mathematical/statistical models.
C. Meta_Analysis - systematic review with quantitative pooling of parameter estimates.
D. Methodological - new/validated estimation methods and/or software/tools for parameter estimation.
E. Review_Commentary - narrative reviews, perspectives, editorials without new data or pooled estimates.
F. Unclear - not enough information in the excerpt to decide (use sparingly).

**Definitions:**
{definitions}

**Instructions:**
1. Focus on *how the parameter estimate is produced* (new raw data vs inference vs synthesis vs method/tool vs commentary).
2. Select exactly one letter (A-F).
3. Return a single JSON object matching the schema below. Do NOT include extra text.

**Schema:**
{schema}
"""

    # Prompt-time helper: definitions are injected by the caller when formatting user_prompt_template.
    extra_output_fields: Dict[str, Any] = field(default_factory=lambda: {
        "definitions": "\n".join([f"{k}. {v}" for k, v in _CODE_DEFINITIONS.items() if k != "F"])
    })

    output_schema: Any = PaperTypeClassificationOutput
    default_classification: Any = PaperType.UNCLEAR

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
            "This outbreak report provides daily case counts and timelines sufficient to reconstruct the time series from the manuscript.",
            "The appendix contains patient-level rows allowing reuse for additional analyses.",
        ],
        # AVAILABLE_UPON_REQUEST
        "upon_request": [
            "Data are available from the corresponding author upon reasonable request.",
            "Access requires signing a data use agreement and approval by the data custodian.",
            "Researchers may apply to the institutional data access committee for permission.",
        ],
        # REFERENCED
        "referenced": [
            "Data were obtained from publicly reported case counts, but no download link is provided.",
            "We used data from the national surveillance system as reported by the Ministry of Health.",
            "Data were obtained from the Johns Hopkins COVID-19 dashboard.",
            "We analyzed surveillance data reported by the national public health institute.",
            "We extracted hospital records, but the underlying dataset is not shared and no access mechanism is stated.",
        ],
        # CLOSED
        "closed": [
            "The data cannot be shared due to legal and ethical restrictions.",
            "Patient-level data are confidential and not available for public release.",
            "The dataset is proprietary and cannot be made available.",
        ],
        # NOT_STATED
        "not_stated": [
            "No data availability statement is provided in the manuscript.",
            "The paper does not describe whether or how the data can be accessed.",
        ],
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
        "If the paper is itself the primary data release (outbreak report / line list / transmission chain), "
        "use OPEN."
    )

    user_prompt_template: str = r"""
You are a research data librarian. Classify the *data availability* of the dataset(s) USED in the paper.

Key principle:
- Assign labels strictly from statements in the paper. Do not verify links or rely on outside knowledge.

**Categories (multi-label, A–E):**
{categories}

**Definitions:**
{definitions}

Multi-dataset papers:
- If the paper clearly uses multiple datasets with different availability mechanisms, you may include multiple letters.
- Do NOT add a label for a dataset mentioned only as background, comparison, or citation (not used in the methodology).

Ambiguity handling:
- If the paper merely says "data are in the public domain" without a clear retrieval mechanism, prefer **C (REFERENCED)**
  or **E (NOT_STATED)** depending on whether a specific third-party source is named.
- If there is no availability statement at all, choose **E (NOT_STATED)**.

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts (candidate evidence):**
{chunks_info}

**Instructions:**
1. Identify datasets actually used in the study (methods/results), and what the paper says about accessing each.
2. Choose one or more letters (A–E) based on the definitions above.
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

**Categories (multi-label, A–G):**
{categories}

**Definitions (protocol-aligned):**
{definitions}

**Paper Content:**
Title: {title}
Abstract: {abstract}
Keywords: {keywords}

**Relevant Extracts:**
{chunks_info}

**Instructions:**
1. Start from the assumption that no label applies; add a label only with clear evidence.
2. Use multiple labels only when multiple distinct data types are clearly used.
3. If you choose **G**, return only **G**.
4. If the paper is too vague, choose **H**.
5. Return a single JSON object matching the schema below. Do NOT add extra text.

**Schema:**
{schema}
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
