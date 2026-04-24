from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .models import RagCaseRunResult, RagasEvaluatorConfig


def _require_ragas():
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError as exc:
        raise ImportError(
            "ragas is not installed. Install the optional eval dependencies from eval/requirements.txt."
        ) from exc

    return EvaluationDataset, SingleTurnSample, evaluate


def _metric_classes():
    try:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            FactualCorrectness,
        )

        return {
            "faithfulness": Faithfulness,
            "context_precision": ContextPrecision,
            "context_recall": ContextRecall,
            "response_relevancy": AnswerRelevancy,
            "factual_correctness": FactualCorrectness,
        }
    except ImportError:
        from ragas.metrics import (  # type: ignore
            FactualCorrectness,
            ResponseRelevancy,
        )
        from ragas.metrics import Faithfulness as LegacyFaithfulness  # type: ignore
        from ragas.metrics import LLMContextPrecisionWithReference  # type: ignore
        from ragas.metrics import LLMContextRecall  # type: ignore

        return {
            "faithfulness": LegacyFaithfulness,
            "context_precision": LLMContextPrecisionWithReference,
            "context_recall": LLMContextRecall,
            "response_relevancy": ResponseRelevancy,
            "factual_correctness": FactualCorrectness,
        }


def _prepare_google_env(config: RagasEvaluatorConfig) -> None:
    if config.llm_provider == "google" and config.api_key:
        os.environ.setdefault("GOOGLE_API_KEY", config.api_key)
    if config.embedding_provider == "google" and config.api_key:
        os.environ.setdefault("GOOGLE_API_KEY", config.api_key)
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _build_evaluator_llm(config: RagasEvaluatorConfig):
    from ragas.llms import llm_factory

    provider = config.llm_provider
    model = config.llm_model
    if not provider or not model:
        return None

    _prepare_google_env(config)

    kwargs: dict[str, Any] = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_version:
        kwargs["api_version"] = config.api_version

    return llm_factory(model, provider=provider, **kwargs)


def _build_evaluator_embeddings(config: RagasEvaluatorConfig):
    provider = config.embedding_provider
    model = config.embedding_model
    if not provider or not model:
        return None

    _prepare_google_env(config)

    if provider == "openai":
        from openai import OpenAI
        from ragas.embeddings import OpenAIEmbeddings

        client = OpenAI(api_key=config.api_key) if config.api_key else OpenAI()
        return OpenAIEmbeddings(client=client, model=model)

    if provider == "google":
        from ragas.embeddings import GoogleEmbeddings

        return GoogleEmbeddings(model=model)

    if provider in {"litellm", "ollama", "openai_compatible"}:
        from ragas.embeddings import LiteLLMEmbeddings

        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["api_base"] = config.api_base
        if config.base_url:
            kwargs["api_base"] = config.base_url
        if config.api_version:
            kwargs["api_version"] = config.api_version
        return LiteLLMEmbeddings(model=model, **kwargs)

    raise ValueError(
        f"Unsupported evaluator embedding provider {provider!r}. "
        "Use one of: openai, google, litellm, ollama, openai_compatible."
    )


def _build_metrics(config: RagasEvaluatorConfig, llm: Any, embeddings: Any) -> list[Any]:
    classes = _metric_classes()
    metrics: list[Any] = []
    for name in config.metric_names:
        key = name.strip().lower()
        if key not in classes:
            supported = ", ".join(sorted(classes))
            raise ValueError(f"Unknown metric {name!r}. Supported metrics: {supported}")
        metric_cls = classes[key]
        kwargs: dict[str, Any] = {}
        if key in {"faithfulness", "context_precision", "context_recall", "factual_correctness"} and llm is not None:
            kwargs["llm"] = llm
        if key == "response_relevancy":
            if llm is not None:
                kwargs["llm"] = llm
            if embeddings is not None:
                kwargs["embeddings"] = embeddings
        metrics.append(metric_cls(**kwargs))
    return metrics


def evaluate_case_runs(
    runs: list[RagCaseRunResult],
    config: RagasEvaluatorConfig,
) -> pd.DataFrame:
    EvaluationDataset, SingleTurnSample, evaluate = _require_ragas()

    eligible_runs = [run for run in runs if run.run_error is None and run.response is not None]
    if not eligible_runs:
        return pd.DataFrame()

    llm = _build_evaluator_llm(config)
    embeddings = _build_evaluator_embeddings(config)
    metrics = _build_metrics(config, llm=llm, embeddings=embeddings)

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=run.user_input,
                retrieved_contexts=list(run.retrieved_contexts),
                response=run.response or "",
                reference=run.reference,
                reference_contexts=list(run.reference_contexts),
            )
            for run in eligible_runs
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=config.raise_exceptions,
    )
    frame = result.to_pandas()
    frame.insert(0, "case_id", [run.case_id for run in eligible_runs])
    frame.insert(1, "paper_path", [run.paper_path for run in eligible_runs])
    frame.insert(2, "paper_id", [run.paper_id for run in eligible_runs])
    return frame


def summarize_ragas_scores(scores: pd.DataFrame, metric_names: list[str]) -> dict[str, Any]:
    if scores.empty:
        return {}
    summary: dict[str, Any] = {"row_count": int(len(scores))}
    for metric in metric_names:
        if metric in scores.columns:
            series = pd.to_numeric(scores[metric], errors="coerce")
            summary[metric] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)) if len(series) > 1 else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
            }
    return summary
