from __future__ import annotations

import pytest

from episcope.clients import AnthropicClient
from episcope.services.runtime import (
    EpiScopeRuntime,
    RuntimeConfig,
    _coerce_llm_provider,
)


def test_build_llm_client_dispatches_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    runtime = EpiScopeRuntime(RuntimeConfig(llm_provider="anthropic"))
    assert isinstance(runtime.build_llm_client(), AnthropicClient)


def test_coerce_llm_provider_accepts_anthropic() -> None:
    assert _coerce_llm_provider("Anthropic") == "anthropic"


def test_coerce_llm_provider_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported llm_provider"):
        _coerce_llm_provider("bogus")
