"""
Configuration modules for EpiScope.

This package exposes configuration objects for the various indexing and retrieval
strategies supported by EpiScope.  Each configuration file defines sensible
defaults for embedding models, index parameters, prompts and connection
settings.  See individual modules for details.
"""

# Explicitly import common configs so they are discoverable via
# `from episcope.configs import text_only, colpali, graph`.
from .text_only import *  # noqa: F401,F403
from .ColPali import *  # noqa: F401,F403
from .graph import *  # noqa: F401,F403
from .CLIP import *  # noqa: F401,F403

__all__ = []
