"""Paired significance testing for classification baselines.

Adds the missing piece flagged in `eval/classification/PAPER_NARRATIVE.md`:
paper §6.1.2 (positioning across the design space) and §6.1.3 (component
ablations) report differences between methods, but without paired
significance testing those differences are point estimates only. This
module provides:

  - per-paper score vectors aligned across methods (Jaccard and exact-match
    indicators, averaged or majority-voted across repeats);
  - paired bootstrap confidence intervals on the mean per-paper score
    difference, using BCa intervals (Efron, 1987) with a percentile
    fallback;
  - McNemar's test on the per-paper exact-match indicators, with continuity
    correction by default and an exact binomial fallback when discordant
    pairs are few;
  - Holm-Bonferroni correction across the pairs reported in a single panel.

The orchestrator `pairwise_significance` returns a tidy DataFrame consumed
by `eval/scripts/run_classification_significance.py` and (optionally) by the
existing classification eval pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Per-paper score primitives
# ---------------------------------------------------------------------------


def _jaccard(pred: frozenset, gt: frozenset) -> float:
    """Jaccard similarity between two frozensets. Two empty sets score 1."""
    if not pred and not gt:
        return 1.0
    union = pred | gt
    if not union:
        return 1.0
    return len(pred & gt) / len(union)


def _exact_match(pred: frozenset, gt: frozenset) -> bool:
    return pred == gt


def per_paper_method_scores(
    merged: pd.DataFrame,
    *,
    method_cols: Sequence[str] = ("model",),
    correctness_aggregator: str = "majority",
) -> pd.DataFrame:
    """Aggregate the merged predictions DataFrame to one row per paper/method.

    Parameters
    ----------
    merged
        Output of ``eval.classification.error_analysis.load_merged_predictions``.
        Must contain columns ``paper_id, task, pred_set, gt_set`` plus the
        columns named in ``method_cols``.
    method_cols
        Columns that together identify a "method". Default ``("model",)``;
        pass ``("model", "temperature")`` to treat each (model, temperature)
        as its own method.
    correctness_aggregator
        How to collapse per-run exact-match indicators into a single
        per-paper boolean. ``"majority"`` (default) takes the majority vote
        across runs; ``"all"`` requires every run to be correct; ``"any"``
        accepts any correct run.

    Returns
    -------
    DataFrame with columns ``task, paper_id, <method_cols>, n_runs,
    jaccard_mean, correct``. One row per (task, method, paper_id).
    """
    if merged.empty:
        return pd.DataFrame(
            columns=["task", "paper_id", *method_cols, "n_runs", "jaccard_mean", "correct"]
        )

    df = merged.copy()
    df["_jaccard"] = [
        _jaccard(p, g) for p, g in zip(df["pred_set"], df["gt_set"])
    ]
    df["_correct"] = [
        _exact_match(p, g) for p, g in zip(df["pred_set"], df["gt_set"])
    ]

    group_cols = ["task", *method_cols, "paper_id"]

    if correctness_aggregator == "majority":
        def _agg_correct(values: pd.Series) -> bool:
            return bool(values.sum() * 2 >= len(values))
    elif correctness_aggregator == "all":
        def _agg_correct(values: pd.Series) -> bool:
            return bool(values.all())
    elif correctness_aggregator == "any":
        def _agg_correct(values: pd.Series) -> bool:
            return bool(values.any())
    else:
        raise ValueError(
            f"correctness_aggregator must be one of majority/all/any; "
            f"got {correctness_aggregator!r}."
        )

    grouped = df.groupby(group_cols, as_index=False).agg(
        n_runs=("_jaccard", "count"),
        jaccard_mean=("_jaccard", "mean"),
        correct=("_correct", _agg_correct),
    )
    return grouped


# ---------------------------------------------------------------------------
# Paired bootstrap CI on the mean difference
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    method: str
    B: int
    alpha: float
    n: int


def paired_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    B: int = 10_000,
    alpha: float = 0.05,
    method: str = "bca",
    random_state: int = 13,
) -> BootstrapResult:
    """Paired bootstrap on the mean difference ``mean(a) - mean(b)``.

    Both inputs are length-N vectors of *paired* per-paper scores (same paper
    in the same position). NaNs in either vector drop the pair.

    The bootstrap resamples paper indices (paired) with replacement B times
    and recomputes the mean difference each time. Two interval methods:

    - ``"bca"`` (default): bias-corrected and accelerated interval (Efron 1987).
      Uses jackknife for the acceleration constant. Falls back to percentile
      if the BCa adjustment is undefined (e.g., all bootstrap diffs equal).
    - ``"percentile"``: empirical quantiles of the bootstrap distribution.

    The two-sided p-value is computed as ``2 * min(P(diff <= 0), P(diff >= 0))``
    over the bootstrap distribution, clipped to ``[0, 1]``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}.")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = len(a)
    if n == 0:
        return BootstrapResult(
            mean_a=float("nan"),
            mean_b=float("nan"),
            mean_diff=float("nan"),
            ci_lo=float("nan"),
            ci_hi=float("nan"),
            p_value=float("nan"),
            method=method,
            B=B,
            alpha=alpha,
            n=0,
        )

    rng = np.random.default_rng(random_state)
    diffs = a - b
    observed = float(diffs.mean())

    indices = rng.integers(0, n, size=(B, n))
    boot_diffs = diffs[indices].mean(axis=1)

    if method == "percentile":
        ci_lo = float(np.quantile(boot_diffs, alpha / 2))
        ci_hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
        used_method = "percentile"
    elif method == "bca":
        ci_lo, ci_hi, used_method = _bca_interval(
            diffs=diffs,
            boot_diffs=boot_diffs,
            observed=observed,
            alpha=alpha,
        )
    else:
        raise ValueError(f"method must be 'bca' or 'percentile'; got {method!r}.")

    # Two-sided bootstrap p-value via the achieved significance level.
    p_le = float(np.mean(boot_diffs <= 0))
    p_ge = float(np.mean(boot_diffs >= 0))
    p_value = min(1.0, 2.0 * min(p_le, p_ge))

    return BootstrapResult(
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=observed,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        p_value=p_value,
        method=used_method,
        B=B,
        alpha=alpha,
        n=n,
    )


def _bca_interval(
    *,
    diffs: np.ndarray,
    boot_diffs: np.ndarray,
    observed: float,
    alpha: float,
) -> Tuple[float, float, str]:
    """BCa interval; fall back to percentile if the adjustment is undefined."""
    # Bias-correction z0: inverse-CDF of fraction of bootstrap reps below the
    # observed statistic.
    frac_below = float(np.mean(boot_diffs < observed))
    if frac_below in (0.0, 1.0):
        ci_lo = float(np.quantile(boot_diffs, alpha / 2))
        ci_hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
        return ci_lo, ci_hi, "bca-fallback-percentile"
    z0 = stats.norm.ppf(frac_below)

    # Acceleration via jackknife on the paired differences (the statistic is
    # the mean of a single vector, so jackknifing diffs is appropriate).
    n = len(diffs)
    jack_means = (diffs.sum() - diffs) / (n - 1)
    jack_mean = jack_means.mean()
    num = float(np.sum((jack_mean - jack_means) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jack_means) ** 2))) ** 1.5
    if den == 0.0:
        ci_lo = float(np.quantile(boot_diffs, alpha / 2))
        ci_hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
        return ci_lo, ci_hi, "bca-fallback-percentile"
    accel = num / den

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    def _adjust(z: float) -> float:
        denom = 1.0 - accel * (z0 + z)
        if denom == 0.0:
            return float("nan")
        return float(stats.norm.cdf(z0 + (z0 + z) / denom))

    alpha_lo = _adjust(z_lo)
    alpha_hi = _adjust(z_hi)
    if not (0.0 < alpha_lo < 1.0 and 0.0 < alpha_hi < 1.0):
        ci_lo = float(np.quantile(boot_diffs, alpha / 2))
        ci_hi = float(np.quantile(boot_diffs, 1 - alpha / 2))
        return ci_lo, ci_hi, "bca-fallback-percentile"
    return (
        float(np.quantile(boot_diffs, alpha_lo)),
        float(np.quantile(boot_diffs, alpha_hi)),
        "bca",
    )


# ---------------------------------------------------------------------------
# McNemar's test on paired correctness indicators
# ---------------------------------------------------------------------------


@dataclass
class McNemarResult:
    n: int
    n_a_only: int  # papers where A correct and B wrong (the "b" cell)
    n_b_only: int  # papers where B correct and A wrong (the "c" cell)
    n_both: int
    n_neither: int
    statistic: float
    p_value: float
    test: str  # "chi2-continuity" or "binomial-exact"


def mcnemar(
    a_correct: np.ndarray,
    b_correct: np.ndarray,
    *,
    continuity: bool = True,
    exact_threshold: int = 25,
) -> McNemarResult:
    """Paired McNemar's test on two boolean correctness vectors.

    Uses the chi-square approximation with continuity correction by default.
    When the number of discordant pairs ``b + c`` is small (default <= 25),
    an exact two-sided binomial test is used instead.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"a and b must have the same shape; got {a.shape} vs {b.shape}.")
    n_a_only = int(np.sum(a & ~b))
    n_b_only = int(np.sum(~a & b))
    n_both = int(np.sum(a & b))
    n_neither = int(np.sum(~a & ~b))
    discordant = n_a_only + n_b_only
    n = n_a_only + n_b_only + n_both + n_neither

    if discordant == 0:
        return McNemarResult(
            n=n,
            n_a_only=n_a_only,
            n_b_only=n_b_only,
            n_both=n_both,
            n_neither=n_neither,
            statistic=0.0,
            p_value=1.0,
            test="chi2-continuity" if continuity else "chi2",
        )

    if discordant <= exact_threshold:
        # Exact two-sided binomial on the smaller cell vs total discordants
        # under H0: P(A wrong, B correct) = P(A correct, B wrong) = 0.5.
        result = stats.binomtest(min(n_a_only, n_b_only), discordant, p=0.5, alternative="two-sided")
        return McNemarResult(
            n=n,
            n_a_only=n_a_only,
            n_b_only=n_b_only,
            n_both=n_both,
            n_neither=n_neither,
            statistic=float(min(n_a_only, n_b_only)),
            p_value=float(result.pvalue),
            test="binomial-exact",
        )

    if continuity:
        chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / discordant
        test_label = "chi2-continuity"
    else:
        chi2 = (n_a_only - n_b_only) ** 2 / discordant
        test_label = "chi2"
    p = float(stats.chi2.sf(chi2, df=1))
    return McNemarResult(
        n=n,
        n_a_only=n_a_only,
        n_b_only=n_b_only,
        n_both=n_both,
        n_neither=n_neither,
        statistic=float(chi2),
        p_value=p,
        test=test_label,
    )


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def holm_correct(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values.

    NaN inputs are preserved (treated as missing rather than 0 or 1) and do
    not consume a step.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    valid_p = p[valid]
    order = np.argsort(valid_p)
    m = len(valid_p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * valid_p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    out[valid] = adjusted
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def pairwise_significance(
    merged: pd.DataFrame,
    *,
    pairs: Sequence[Tuple[Any, Any]],
    method_cols: Sequence[str] = ("model",),
    tasks: Optional[Iterable[str]] = None,
    correctness_aggregator: str = "majority",
    B: int = 10_000,
    alpha: float = 0.05,
    bootstrap_method: str = "bca",
    correction: str = "holm",
    random_state: int = 13,
) -> pd.DataFrame:
    """Compute paired significance statistics for a set of method pairs.

    For each (task, pair) combination, this function:

    1. Selects per-paper Jaccard and correctness vectors for the two methods,
       restricted to the *intersection* of paper_ids (so the bootstrap and
       McNemar see paired observations only).
    2. Runs the paired bootstrap on the mean Jaccard difference.
    3. Runs McNemar's test on the exact-match indicators.
    4. Applies Holm-Bonferroni correction across the pairs reported on each
       task (the panel-level correction recommended in §6.1 of the paper).

    Each element of ``pairs`` is a 2-tuple ``(method_a_key, method_b_key)``
    where each key matches the value(s) of ``method_cols``. If
    ``method_cols == ("model",)``, the keys are bare strings. For multi-column
    method identifiers, pass tuples — e.g.,
    ``pairs=[(("gemini-2-5-flash", "0.0"), ("gemini-2-5-flash", "1.0"))]``
    with ``method_cols=("model", "temperature")``.

    Returns a tidy DataFrame with one row per (task, pair) and columns:
    ``task, method_a, method_b, n, mean_a, mean_b, mean_diff,
    ci_lo, ci_hi, ci_method, boot_p, mcnemar_n_a_only, mcnemar_n_b_only,
    mcnemar_statistic, mcnemar_test, mcnemar_p, mcnemar_p_adj, boot_p_adj``.
    """
    per_paper = per_paper_method_scores(
        merged,
        method_cols=method_cols,
        correctness_aggregator=correctness_aggregator,
    )
    method_col_tuple = tuple(method_cols)

    def _key_to_tuple(key: Any) -> tuple:
        if isinstance(key, tuple):
            return key
        if len(method_col_tuple) == 1:
            return (key,)
        raise ValueError(
            f"pair entry {key!r} must be a tuple when method_cols has length "
            f"{len(method_col_tuple)}."
        )

    available_tasks = (
        list(tasks)
        if tasks is not None
        else sorted(per_paper["task"].dropna().unique().tolist())
    )

    rows: list[dict[str, Any]] = []
    for task in available_tasks:
        task_df = per_paper[per_paper["task"] == task]
        if task_df.empty:
            continue
        for pair in pairs:
            key_a = _key_to_tuple(pair[0])
            key_b = _key_to_tuple(pair[1])
            mask_a = np.logical_and.reduce(
                [task_df[col] == val for col, val in zip(method_col_tuple, key_a)]
            )
            mask_b = np.logical_and.reduce(
                [task_df[col] == val for col, val in zip(method_col_tuple, key_b)]
            )
            sub_a = task_df[mask_a][["paper_id", "jaccard_mean", "correct"]]
            sub_b = task_df[mask_b][["paper_id", "jaccard_mean", "correct"]]
            merged_pair = sub_a.merge(
                sub_b, on="paper_id", how="inner", suffixes=("_a", "_b")
            )
            if merged_pair.empty:
                rows.append(
                    {
                        "task": task,
                        "method_a": pair[0],
                        "method_b": pair[1],
                        "n": 0,
                        "mean_a": float("nan"),
                        "mean_b": float("nan"),
                        "mean_diff": float("nan"),
                        "ci_lo": float("nan"),
                        "ci_hi": float("nan"),
                        "ci_method": "n/a",
                        "boot_p": float("nan"),
                        "mcnemar_n_a_only": 0,
                        "mcnemar_n_b_only": 0,
                        "mcnemar_statistic": float("nan"),
                        "mcnemar_test": "n/a",
                        "mcnemar_p": float("nan"),
                    }
                )
                continue
            boot = paired_bootstrap_diff(
                merged_pair["jaccard_mean_a"].to_numpy(),
                merged_pair["jaccard_mean_b"].to_numpy(),
                B=B,
                alpha=alpha,
                method=bootstrap_method,
                random_state=random_state,
            )
            mc = mcnemar(
                merged_pair["correct_a"].to_numpy(dtype=bool),
                merged_pair["correct_b"].to_numpy(dtype=bool),
            )
            rows.append(
                {
                    "task": task,
                    "method_a": pair[0],
                    "method_b": pair[1],
                    "n": boot.n,
                    "mean_a": boot.mean_a,
                    "mean_b": boot.mean_b,
                    "mean_diff": boot.mean_diff,
                    "ci_lo": boot.ci_lo,
                    "ci_hi": boot.ci_hi,
                    "ci_method": boot.method,
                    "boot_p": boot.p_value,
                    "mcnemar_n_a_only": mc.n_a_only,
                    "mcnemar_n_b_only": mc.n_b_only,
                    "mcnemar_statistic": mc.statistic,
                    "mcnemar_test": mc.test,
                    "mcnemar_p": mc.p_value,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    if correction == "holm":
        result["boot_p_adj"] = np.nan
        result["mcnemar_p_adj"] = np.nan
        for task, frame in result.groupby("task"):
            result.loc[frame.index, "boot_p_adj"] = holm_correct(frame["boot_p"].to_numpy())
            result.loc[frame.index, "mcnemar_p_adj"] = holm_correct(
                frame["mcnemar_p"].to_numpy()
            )
    elif correction == "none":
        result["boot_p_adj"] = result["boot_p"]
        result["mcnemar_p_adj"] = result["mcnemar_p"]
    else:
        raise ValueError(f"correction must be 'holm' or 'none'; got {correction!r}.")

    return result.sort_values(["task", "method_a", "method_b"]).reset_index(drop=True)
