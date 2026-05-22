from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

from episcope.rag.provenance import Provenance


class Generator(ABC):
    """Abstract base class for generators."""

    @abstractmethod
    def generate(
        self,
        contexts: Sequence[Any],
        *,
        question: Optional[str] = None,
        message_builder: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Provenance:
        """Generate a response based on the given inputs."""
        pass
