from __future__ import annotations

import os
from typing import Any

import pandas as pd

from .models import RagCaseRunResult, RagasEvaluatorConfig

_GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

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


def _normalize_provider_alias(provider: str | None) -> str | None:
    if provider is None:
        return None
    normalized = provider.strip().lower()
    if normalized == "gemini":
        return "google"
    return normalized


def _prepare_google_env(config: RagasEvaluatorConfig) -> None:
    llm_provider = _normalize_provider_alias(config.llm_provider)
    embedding_provider = _normalize_provider_alias(config.embedding_provider)
    if llm_provider == "google" and config.api_key:
        os.environ.setdefault("GOOGLE_API_KEY", config.api_key)
    if embedding_provider == "google" and config.api_key:
        os.environ.setdefault("GOOGLE_API_KEY", config.api_key)
    if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _resolve_gemini_openai_base_url(config: RagasEvaluatorConfig) -> str:
    return config.base_url or config.api_base or _GEMINI_OPENAI_BASE_URL


def _build_ragas_client(provider: str, config: RagasEvaluatorConfig):
    if provider == "openai":
        from openai import OpenAI

        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        elif config.api_base:
            kwargs["base_url"] = config.api_base
        return OpenAI(**kwargs)

    if provider == "google":
        from google import genai
        from google.genai import types
        import httpx

        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")

        timeout_config = httpx.Timeout(
            connect=10.0,
            read=30.0,
            write=10.0,
            pool=10.0,
        )
        limits_config = httpx.Limits(
            max_keepalive_connections=5,
            max_connections=10,
        )

        http_options = types.HttpOptions(
            api_version=config.api_version,
            base_url=config.base_url or config.api_base,
            client_args={
                "timeout": timeout_config,
                "limits": limits_config,
            },
            async_client_args={
                "timeout": timeout_config,
                "limits": limits_config,
            }
        )

        return genai.Client(
            api_key=api_key,
            http_options=http_options
        )

    return None


def _build_ragas_llm_client_and_provider(
    provider: str,
    config: RagasEvaluatorConfig,
) -> tuple[Any | None, str]:
    if provider == "google":
        from openai import OpenAI

        api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        kwargs: dict[str, Any] = {
            "base_url": _resolve_gemini_openai_base_url(config),
        }
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAI(**kwargs), "openai"

    return _build_ragas_client(provider, config), provider


def _build_evaluator_llm(config: RagasEvaluatorConfig):
    from ragas.llms import llm_factory

    provider = _normalize_provider_alias(config.llm_provider)
    model = config.llm_model
    if not provider or not model:
        return None

    _prepare_google_env(config)
    client, ragas_provider = _build_ragas_llm_client_and_provider(provider, config)

    kwargs: dict[str, Any] = {}
    if config.api_version and ragas_provider != "openai":
        kwargs["api_version"] = config.api_version
    if config.base_url and ragas_provider not in {"openai"}:
        kwargs["base_url"] = config.base_url
    elif config.api_base and ragas_provider not in {"openai"}:
        kwargs["base_url"] = config.api_base
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort

    if client is not None:
        return llm_factory(model, provider=ragas_provider, client=client, **kwargs)

    if ragas_provider in {"litellm", "ollama", "openai_compatible"}:
        raise ValueError(
            f"Unsupported evaluator LLM provider {provider!r} with the installed ragas version. "
            "Please use openai or gemini/google, or extend _build_ragas_client()."
        )

    raise ValueError(
        f"Unsupported evaluator LLM provider {provider!r}. "
        "Use one of: openai, gemini, google."
    )


def _build_evaluator_embeddings(config: RagasEvaluatorConfig):
    from ragas.embeddings.base import embedding_factory

    provider = _normalize_provider_alias(config.embedding_provider)
    model = config.embedding_model
    if not provider or not model:
        return None

    _prepare_google_env(config)
    client = _build_ragas_client(provider, config)

    if provider in {"openai", "google"}:
        kwargs: dict[str, Any] = {}
        if config.base_url and provider != "openai":
            kwargs["base_url"] = config.base_url
        elif config.api_base and provider != "openai":
            kwargs["base_url"] = config.api_base
        return embedding_factory(provider=provider, model=model, client=client, **kwargs)

    raise ValueError(
        f"Unsupported evaluator embedding provider {provider!r}. "
        "Use one of: openai, gemini, google."
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
