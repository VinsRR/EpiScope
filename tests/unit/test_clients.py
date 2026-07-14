from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

import pytest

from episcope.clients import AnthropicClient


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeUsage:
    input_tokens: int = 12
    output_tokens: int = 34
    cache_read_input_tokens: int = 0


@dataclass
class _FakeMessage:
    content: List[_FakeTextBlock]
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeMessagesResource:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def create(self, **kwargs: Any) -> _FakeMessage:
        self.calls.append(kwargs)
        return _FakeMessage(content=[_FakeTextBlock(text="the answer")])


class _FakeAnthropicSDKClient:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.messages = _FakeMessagesResource()


@pytest.fixture()
def client(monkeypatch) -> AnthropicClient:
    """An AnthropicClient wired to a fake SDK so chat() never hits the network."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c = AnthropicClient()
    c._client = _FakeAnthropicSDKClient()
    return c


def test_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_requires_anthropic_package(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("episcope.clients.anthropic", None)
    with pytest.raises(ImportError, match="anthropic package"):
        AnthropicClient()


def test_chat_extracts_system_message_and_applies_cache_control(client) -> None:
    answer = client.chat(
        [
            {"role": "system", "content": "You are careful."},
            {"role": "user", "content": "Summarize the paper."},
        ],
        model="claude-sonnet-5",
    )

    assert answer == "the answer"
    call = client._client.messages.calls[0]
    assert call["messages"] == [{"role": "user", "content": "Summarize the paper."}]
    assert call["system"] == [
        {
            "type": "text",
            "text": "You are careful.",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_chat_without_prompt_cache_sends_plain_system_string(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    c = AnthropicClient(prompt_cache=False)
    c._client = _FakeAnthropicSDKClient()

    c.chat(
        [
            {"role": "system", "content": "You are careful."},
            {"role": "user", "content": "Hi"},
        ],
        model="claude-sonnet-5",
    )

    assert c._client.messages.calls[0]["system"] == "You are careful."


def test_chat_omits_system_param_when_no_system_message(client) -> None:
    client.chat([{"role": "user", "content": "Hi"}], model="claude-sonnet-5")
    assert "system" not in client._client.messages.calls[0]


def test_chat_defaults_max_tokens_when_not_supplied(client) -> None:
    client.chat([{"role": "user", "content": "Hi"}], model="claude-sonnet-5")
    assert client._client.messages.calls[0]["max_tokens"] == client.default_max_tokens


def test_chat_honors_explicit_max_tokens(client) -> None:
    client.chat(
        [{"role": "user", "content": "Hi"}], model="claude-sonnet-5", max_tokens=256
    )
    assert client._client.messages.calls[0]["max_tokens"] == 256


def test_chat_records_usage(client) -> None:
    client.chat([{"role": "user", "content": "Hi"}], model="claude-sonnet-5")
    usage = client.usage_snapshot()
    assert usage.prompt_tokens == 12
    assert usage.completion_tokens == 34
    assert usage.call_count == 1


def test_embed_is_not_supported(client) -> None:
    with pytest.raises(NotImplementedError, match="does not provide an embeddings API"):
        client.embed(["hello"], model="whatever")
