"""Top-level classification experiments and baselines.

This contains research
baselines, exploratory topic-modeling utilities, and runner code that depends on
EpiScope data structures without being part of the production EpiScope package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _candidate in (_ROOT, _SRC):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
