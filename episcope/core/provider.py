"""
core/provider.py

Abstractions for large language model (LLM) providers.

The provider layer encapsulates the details of talking to a language
model.  By abstracting this interaction behind a common interface the
rest of the system can remain agnostic to whether a local model or a
hosted API is used.  See ``providers/`` for concrete implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List


class AbstractProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: Iterable[dict[str, str]], **kwargs: Any) -> str:
        """Send a sequence of messages to the provider and return the response.

        Parameters
        ----------
        messages: Iterable[dict[str, str]]
            A sequence of chat messages, where each message is a dict
            with keys ``"role"`` and ``"content"``.  The provider is
            responsible for converting these messages into the format
            expected by the underlying model.
        **kwargs: Any
            Additional provider‑specific keyword arguments.

        Returns
        -------
        str
            The generated text response.
        """


class ProviderFactory:
    """Factory for instantiating provider classes.

    Concrete provider implementations should register themselves via
    ``ProviderFactory.register_provider``.  At runtime the selected
    provider can be instantiated via ``ProviderFactory.create`` using
    the name configured in ``settings.CONFIG``.
    """

    _providers: dict[str, type[AbstractProvider]] = {}

    @classmethod
    def register_provider(cls, name: str, provider_cls: type[AbstractProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AbstractProvider:
        if name not in cls._providers:
            raise ValueError(f"Unknown provider '{name}'. Available: {list(cls._providers)}")
        return cls._providers[name](**kwargs)  # type: ignore[return-value]

