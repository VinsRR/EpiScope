"""Helpers that turn `per_run_metrics.csv` into compact, paper-ready views.

`run_classification_eval.py` already emits one row per result TSV with the full
metric battery, plus a wide summary with five aggregate stats per metric. That
output is comprehensive but unwieldy at a glance — typically what we actually
want for the paper (and for sanity-checking baseline runs) is a single row per
baseline with one column per task and "mean ± std (n)" in each cell.

This module derives:

  * `family`        — zero_shot | unsupervised | supervised | llm | external,
                      inferred from the ``outputs/baselines{,_old}/<family>/...``
                      prefix in the source file path. Falls back to "external"
                      when the prefix cannot be identified.
  * `display_label` — a friendly baseline name derived from the path-encoded
                      model string. Driven by ordered regex rules so that new
                      baselines can be added without touching the renderer.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import pandas as pd


# --- family extraction ------------------------------------------------------

_FAMILY_TOKENS = ("zero_shot", "unsupervised", "supervised", "llm")


def extract_family(source_file: str | Path) -> str:
    """Return the baseline family encoded in the result-file path.

    The expected layout is
    ``outputs/<root>/<family>/<task>/<baseline>/.../final_*.tsv`` where
    ``<root>`` is e.g. ``baselines`` or ``baselines_old``. We scan the path
    parts for the first match against the known family tokens.
    """
    parts = Path(source_file).parts
    for part in parts:
        if part in _FAMILY_TOKENS:
            return part
    return "external"


# --- pretty labelling ------------------------------------------------------

# Rules are applied in order; the first matching pattern wins. Each rule maps
# the raw model string to a human-readable label. Keep new rules narrower than
# the ones below them so additions stay non-destructive.
_LABEL_RULES: list[tuple[re.Pattern[str], str]] = [
    # LLM ablations -------------------------------------------------------
    (
        re.compile(r"^metadata[-_]llm[-_]gemini[-_]gemini-([\d.]+)-(flash|pro)$"),
        lambda m: f"Metadata-only LLM (Gemini {m.group(1)} {m.group(2).capitalize()})",
    ),
    (
        re.compile(r"^metadata[-_]llm[-_]gemini[-_]gemini-([\d.]+)-(flash|pro).*$"),
        lambda m: f"Metadata-only LLM (Gemini {m.group(1)} {m.group(2).capitalize()})",
    ),
    (
        re.compile(
            r"^random[-_]chunk[-_]llm[-_]gemini[-_]gemini-([\d.]+)-(flash|pro)[-_]k_(\d+)[-_]seed_(\d+)$"
        ),
        lambda m: f"Random-chunk LLM (Gemini {m.group(1)} {m.group(2).capitalize()}, K={m.group(3)})",
    ),
    # Supervised on engineered features ----------------------------------
    (
        re.compile(r"^supervised_tfidf_logreg-cv_leave_one_out$"),
        "TF-IDF + LogReg (LOOCV)",
    ),
    (
        re.compile(r"^supervised_tfidf_linear_svm-cv_leave_one_out$"),
        "TF-IDF + Linear SVM (LOOCV)",
    ),
    (
        re.compile(
            r"^supervised_frozen_logreg-enc_(.+)-src_(.+)-cv_leave_one_out$"
        ),
        lambda m: f"Frozen {_clean_encoder(m.group(1))} + LogReg (src={m.group(2)}, LOOCV)",
    ),
    (
        re.compile(r"^supervised_bertopic-cv_leave_one_out$"),
        "BERTopic + LogReg (LOOCV)",
    ),
    (
        re.compile(
            r"^supervised_(lda|lsa|nmf|plsa)_logreg-k_(\d+)-cv_leave_one_out$"
        ),
        lambda m: f"{m.group(1).upper()} (k={m.group(2)}) + LogReg (LOOCV)",
    ),
    # Zero-shot / unsupervised --------------------------------------------
    (re.compile(r"^majority$"), "Majority class"),
    (
        re.compile(r"^prototype_similarity-emb_(.+)$"),
        lambda m: f"Prototype similarity ({_clean_encoder(m.group(1))})",
    ),
    (
        re.compile(
            r"^nli_zero_shot-(.+)-src_(.+)-thr_([\d.]+)$"
        ),
        lambda m: f"NLI zero-shot ({_clean_encoder(m.group(1))}, src={m.group(2)}, thr={m.group(3)})",
    ),
    # EpiScope full system (results come from outputs/output/) ---------------
    (
        re.compile(r"^gemini-([\d]+)-([\d]+)-(flash|pro)$"),
        lambda m: f"EpiScope (Gemini {m.group(1)}.{m.group(2)} {m.group(3).capitalize()})",
    ),
    # Unsupervised topic models (label-mapped via LLM) --------------------
    (
        re.compile(r"^(lda|lsa|nmf|plsa)-k_(\d+)$"),
        lambda m: f"{m.group(1).upper()} (k={m.group(2)})",
    ),
    (
        re.compile(r"^bertopic-k_(\d+)$"),
        lambda m: f"BERTopic (k={m.group(1)})",
    ),
    (
        re.compile(r"^bertopic_guided-k_(\d+)$"),
        lambda m: f"BERTopic guided (k={m.group(1)})",
    ),
    (
        re.compile(r"^bertopic_semisupervised-k_(\d+)-labels_([\d.]+)$"),
        lambda m: f"BERTopic semi-sup. (k={m.group(1)}, labels={m.group(2)})",
    ),
]


_ENCODER_SHORTNAMES = {
    "allenai-specter": "SPECTER",
    "allenai-scibert-scivocab-uncased": "SciBERT",
    "moritzlaurer-deberta-v3-base-mnli-fever-anli": "DeBERTa-v3-mnli",
}


def _clean_encoder(raw: str) -> str:
    """Map raw encoder slugs to short names; otherwise return as-is."""
    return _ENCODER_SHORTNAMES.get(raw, raw)


def display_label(model: str) -> str:
    """Render a path-encoded model string into a human-readable label."""
    for pattern, replacement in _LABEL_RULES:
        match = pattern.match(model)
        if match is None:
            continue
        if callable(replacement):
            return replacement(match)
        return match.expand(replacement)
    return model


# --- ordering --------------------------------------------------------------

_FAMILY_ORDER = {
    "zero_shot": 0,
    "unsupervised": 1,
    "supervised": 2,
    "llm": 3,
    "external": 4,
}

_TASK_DISPLAY = {
    "geo": "Geography",
    "data-type": "Data type",
    "data-accessibility": "Data accessibility",
    "paper-type": "Paper type",
}

_TASK_ORDER = ["geo", "data-type", "data-accessibility", "paper-type"]


# --- pivot construction ----------------------------------------------------

def _fmt_cell(mean: float, std: float, n: int) -> str:
    if pd.isna(mean):
        return ""
    if n <= 1 or pd.isna(std) or std == 0:
        return f"{mean:.3f} (n={n})"
    return f"{mean:.3f} ± {std:.3f} (n={n})"


def enrich_per_run(per_run: pd.DataFrame) -> pd.DataFrame:
    """Attach family + display_label columns to a per_run_metrics frame.

    When ``temperature`` is set to something other than ``"unknown"`` (i.e. an
    actual sampling temperature recorded in the path), it is appended to the
    display label so that, e.g., EpiScope T=0 and T=1 appear as separate rows
    rather than being silently averaged together.
    """
    if per_run.empty:
        return per_run.assign(family=[], display_label=[])
    out = per_run.copy()
    out["family"] = out["source_file"].map(extract_family)
    base_labels = out["model"].map(display_label)
    temps = out["temperature"].astype(str)
    out["display_label"] = [
        f"{label} (T={temp})" if temp not in ("", "unknown", "nan") else label
        for label, temp in zip(base_labels, temps)
    ]
    return out


def build_pivot(
    per_run: pd.DataFrame,
    *,
    metric: str = "jaccard_samples",
    task_order: Sequence[str] = _TASK_ORDER,
) -> pd.DataFrame:
    """Return a wide pivot: rows=(family, baseline), cols=tasks, cells="mean ± std (n)".

    Aggregates across all rows that share (family, display_label, model, task),
    which collapses multi-run baselines (e.g. random-chunk with 2 seeds) into a
    single row while keeping distinct configurations apart.
    """
    if per_run.empty:
        return pd.DataFrame()

    enriched = enrich_per_run(per_run)
    agg = (
        enriched.groupby(
            ["family", "display_label", "model", "temperature", "task"], as_index=False
        )
        .agg(
            mean=(metric, "mean"),
            std=(metric, lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
            n=(metric, "count"),
        )
    )
    agg["cell"] = [
        _fmt_cell(m, s, int(n)) for m, s, n in zip(agg["mean"], agg["std"], agg["n"])
    ]

    pivot = (
        agg.pivot_table(
            index=["family", "display_label", "model", "temperature"],
            columns="task",
            values="cell",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None

    # Reorder task columns and apply human-readable headers.
    index_cols = ["family", "display_label", "model", "temperature"]
    task_cols = [t for t in task_order if t in pivot.columns]
    other_cols = [c for c in pivot.columns if c not in {*index_cols, *task_cols}]
    pivot = pivot[[*index_cols, *task_cols, *other_cols]]
    pivot = pivot.rename(columns={t: _TASK_DISPLAY.get(t, t) for t in task_cols})

    pivot["__family_order"] = pivot["family"].map(lambda f: _FAMILY_ORDER.get(f, 99))
    pivot = (
        pivot.sort_values(["__family_order", "family", "display_label"])
        .drop(columns="__family_order")
        .reset_index(drop=True)
    )
    return pivot


def render_markdown(pivot: pd.DataFrame, *, title: str | None = None) -> str:
    """Render a pivot table to compact Markdown grouped by family.

    Only the display columns are kept (the raw ``model`` slug is dropped from
    the rendered output). The pivot returned by :func:`build_pivot` is the
    expected input.
    """
    if pivot.empty:
        return "(no data)\n"

    task_cols = [
        c for c in pivot.columns
        if c not in {"family", "display_label", "model", "temperature"}
    ]
    header = ["Baseline", *task_cols]
    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    current_family: str | None = None
    for _, row in pivot.iterrows():
        family = row["family"]
        if family != current_family:
            lines.append(f"| **{_FAMILY_HEADER.get(family, family)}** |" + " |" * len(task_cols))
            current_family = family
        cells = [row["display_label"], *[str(row[c]) if pd.notna(row[c]) else "" for c in task_cols]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


_FAMILY_HEADER = {
    "zero_shot": "Zero-shot / heuristic",
    "unsupervised": "Unsupervised topic models",
    "supervised": "Supervised (on this corpus, LOOCV)",
    "llm": "LLM ablations of EpiScope",
    "external": "EpiScope (RAG-based, full system)",
}
