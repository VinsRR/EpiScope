from abc import ABC, abstractmethod
from typing import Any


class Generator(ABC):
    """Abstract base class for generators."""

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Generate a response based on the given inputs."""
        pass
