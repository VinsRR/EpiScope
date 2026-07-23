"""Tests for provider structured-output / constrained-decoding translation.

Covers the shared schema helpers plus each client's mapping of
``response_schema`` to its native mechanism, the best-effort fallback when a
provider rejects the feature, and the generator/runner plumbing that carries a
schema down to the client.
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

import episcope.clients as clients_mod
from episcope.clients import (
    AnthropicClient,
    GeminiClient,
    OllamaClient,
    OpenAIClient,
    _is_open_map,
    _normalize_response_schema,
    _openai_response_format,
    _schema_name,
    _strictify_json_schema,
)
from episcope.rag.generation.llm_generator import structured_output_kwargs


class _Weather(BaseModel):
    reasoning: str
    confidence: Optional[float] = None
    probs: Optional[Dict[str, float]] = None
    label: str


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def test_normalize_accepts_model_class_and_dict() -> None:
    from_model = _normalize_response_schema(_Weather)
    assert from_model["properties"]["label"]["type"] == "string"
    passthrough = _normalize_response_schema({"type": "object"})
    assert passthrough == {"type": "object"}
    assert _normalize_response_schema(None) is None


def test_normalize_rejects_other_types() -> None:
    with pytest.raises(TypeError):
        _normalize_response_schema(123)


def test_strictify_marks_all_required_and_closes_objects() -> None:
    strict = _strictify_json_schema(_normalize_response_schema(_Weather))
    assert strict["additionalProperties"] is False
    # every remaining property is required...
    assert set(strict["required"]) == set(strict["properties"])
    # ...optional scalar stays nullable via anyOf/null rather than being dropped
    assert {"type": "null"} in strict["properties"]["confidence"]["anyOf"]


def test_strictify_drops_open_maps_and_unsupported_keywords() -> None:
    strict = _strictify_json_schema(_normalize_response_schema(_Weather))
    # Dict[str, float] is an open map -> dropped entirely
    assert "probs" not in strict["properties"]
    assert "probs" not in strict["required"]
    # unsupported validation keywords stripped everywhere
    dumped = json.dumps(strict)
    for banned in ("minimum", "maximum", "default", "title"):
        assert banned not in dumped


def test_is_open_map() -> None:
    assert _is_open_map({"type": "object", "additionalProperties": {"type": "number"}})
    assert not _is_open_map({"type": "object", "properties": {}})
    assert not _is_open_map({"type": "string"})


def test_openai_response_format_shape() -> None:
    rf = _openai_response_format(_normalize_response_schema(_Weather))
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "_Weather"


def test_schema_name_sanitizes() -> None:
    assert _schema_name({"title": "Bad Name!!"}) == "Bad_Name__"
    assert _schema_name({}) == "response"


# ---------------------------------------------------------------------------
# structured_output_kwargs mapping (config mode -> generate kwargs)
# ---------------------------------------------------------------------------


def test_structured_output_kwargs_modes() -> None:
    assert structured_output_kwargs("schema", _Weather) == {"response_schema": _Weather}
    assert structured_output_kwargs("json", _Weather) == {"format": "json"}
    assert structured_output_kwargs("off", _Weather) == {}
    # schema mode with no schema falls through to unconstrained
    assert structured_output_kwargs("schema", None) == {}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class _FakeOllamaResp:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def __enter__(self) -> "_FakeOllamaResp":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Dict[str, Any]:
        return self._data


@pytest.fixture()
def ollama_capture(monkeypatch):
    captured: Dict[str, Any] = {}

    def fake_post(url, json=None, timeout=None, stream=None):  # noqa: A002
        captured["payload"] = json
        return _FakeOllamaResp(
            {"message": {"content": "{}"}, "prompt_eval_count": 1, "eval_count": 1}
        )

    monkeypatch.setattr(clients_mod.requests, "post", fake_post)
    return captured


def test_ollama_sends_schema_as_format(ollama_capture) -> None:
    client = OllamaClient(stream=False)
    client.chat([{"role": "user", "content": "hi"}], model="llama3.2:1b", response_schema=_Weather)
    assert ollama_capture["payload"]["format"] == _Weather.model_json_schema()


def test_ollama_schema_overrides_json_format_kwarg(ollama_capture) -> None:
    client = OllamaClient(stream=False)
    client.chat(
        [{"role": "user", "content": "hi"}],
        model="llama3.2:1b",
        response_schema=_Weather,
        format="json",
    )
    assert ollama_capture["payload"]["format"] == _Weather.model_json_schema()


def test_ollama_plain_json_mode_without_schema(ollama_capture) -> None:
    client = OllamaClient(stream=False)
    client.chat([{"role": "user", "content": "hi"}], model="llama3.2:1b", format="json")
    assert ollama_capture["payload"]["format"] == "json"


# ---------------------------------------------------------------------------
# OpenAI / OpenRouter (shared response_format path)
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, fail_with_schema: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail_with_schema = fail_with_schema

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail_with_schema and "response_format" in kwargs:
            raise RuntimeError("model does not support response_format")
        msg = types.SimpleNamespace(content="{}")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)


def _make_openai(monkeypatch, completions: _FakeCompletions) -> OpenAIClient:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient()
    client._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=completions)
    )
    return client


def test_openai_sets_response_format_json_schema(monkeypatch) -> None:
    completions = _FakeCompletions()
    client = _make_openai(monkeypatch, completions)
    client.chat([{"role": "user", "content": "hi"}], model="gpt-x", response_schema=_Weather)
    rf = completions.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True


def test_openai_no_schema_omits_response_format(monkeypatch) -> None:
    completions = _FakeCompletions()
    client = _make_openai(monkeypatch, completions)
    client.chat([{"role": "user", "content": "hi"}], model="gpt-x")
    assert "response_format" not in completions.calls[0]


def test_openai_falls_back_when_schema_rejected(monkeypatch) -> None:
    completions = _FakeCompletions(fail_with_schema=True)
    client = _make_openai(monkeypatch, completions)
    answer = client.chat(
        [{"role": "user", "content": "hi"}], model="gpt-x", response_schema=_Weather
    )
    assert answer == "{}"
    # first attempt carried the schema and raised; retry dropped it
    assert "response_format" in completions.calls[0]
    assert "response_format" not in completions.calls[1]


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class _FakeModels:
    def __init__(self, fail_with_schema: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail_with_schema = fail_with_schema

    def generate_content(self, *, model: str, contents: Any, config: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "config": config})
        if self.fail_with_schema and getattr(config, "response_json_schema", None):
            raise RuntimeError("schema not supported for this model")
        return types.SimpleNamespace(text='{"ok": 1}', usage_metadata=None)


def _make_gemini(monkeypatch, models: _FakeModels) -> GeminiClient:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = GeminiClient(prompt_cache=False)
    client._client = types.SimpleNamespace(models=models)
    return client


def test_gemini_sets_response_json_schema_in_config(monkeypatch) -> None:
    models = _FakeModels()
    client = _make_gemini(monkeypatch, models)
    client.chat([{"role": "user", "content": "hi"}], model="gemini-2.5-flash", response_schema=_Weather)
    config = models.calls[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema is not None


def test_gemini_falls_back_when_schema_rejected(monkeypatch) -> None:
    models = _FakeModels(fail_with_schema=True)
    client = _make_gemini(monkeypatch, models)
    answer = client.chat(
        [{"role": "user", "content": "hi"}], model="gemini-2.5-flash", response_schema=_Weather
    )
    assert answer == '{"ok": 1}'
    assert getattr(models.calls[0]["config"], "response_json_schema", None) is not None
    assert getattr(models.calls[1]["config"], "response_json_schema", None) is None


# ---------------------------------------------------------------------------
# Anthropic (forced tool use)
# ---------------------------------------------------------------------------


@dataclass
class _FakeToolUseBlock:
    input: Dict[str, Any]
    type: str = "tool_use"


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int = 1
    output_tokens: int = 1
    cache_read_input_tokens: int = 0


class _FakeAnthropicMessages:
    def __init__(self, fail_with_tools: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail_with_tools = fail_with_tools

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail_with_tools and "tools" in kwargs:
            raise RuntimeError("tools not supported")
        if "tools" in kwargs:
            content: List[Any] = [_FakeToolUseBlock(input={"label": "A"})]
        else:
            content = [_FakeTextBlock(text="plain text")]
        return types.SimpleNamespace(content=content, usage=_FakeUsage())


def _make_anthropic(monkeypatch, messages: _FakeAnthropicMessages) -> AnthropicClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient()
    client._client = types.SimpleNamespace(messages=messages)
    return client


def test_anthropic_forces_tool_and_serializes_input(monkeypatch) -> None:
    messages = _FakeAnthropicMessages()
    client = _make_anthropic(monkeypatch, messages)
    answer = client.chat(
        [{"role": "user", "content": "hi"}], model="claude-sonnet-5", response_schema=_Weather
    )
    call = messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "_Weather"}
    assert call["tools"][0]["input_schema"] == _Weather.model_json_schema()
    assert json.loads(answer) == {"label": "A"}


def test_anthropic_falls_back_to_text_when_tools_rejected(monkeypatch) -> None:
    messages = _FakeAnthropicMessages(fail_with_tools=True)
    client = _make_anthropic(monkeypatch, messages)
    answer = client.chat(
        [{"role": "user", "content": "hi"}], model="claude-sonnet-5", response_schema=_Weather
    )
    assert answer == "plain text"
    assert "tools" in messages.calls[0]
    assert "tools" not in messages.calls[1]


# ---------------------------------------------------------------------------
# Generator plumbing
# ---------------------------------------------------------------------------


@dataclass
class _RecordingClient:
    seen: Dict[str, Any] = field(default_factory=dict)

    def chat(self, messages, *, model, temperature=0.0, max_tokens=None, response_schema=None, **kwargs):
        self.seen["response_schema"] = response_schema
        self.seen["messages"] = list(messages)
        return "{}"

    def embed(self, texts, *, model, **kwargs):  # pragma: no cover - unused
        return [[0.0] for _ in texts]


def test_generator_forwards_response_schema_without_leaking_to_builder() -> None:
    from episcope.rag.generation.llm_generator import LLMGenerator

    client = _RecordingClient()
    gen = LLMGenerator(client=client, model="m")

    seen_builder_kwargs: Dict[str, Any] = {}

    def builder(*, contexts, question, **kwargs):
        seen_builder_kwargs.update(kwargs)
        return [{"role": "user", "content": "hi"}]

    gen.generate(contexts=[], message_builder=builder, response_schema=_Weather)
    assert client.seen["response_schema"] is _Weather
    assert "response_schema" not in seen_builder_kwargs
