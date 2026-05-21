from __future__ import annotations

# Set parallelism guards before any library that spawns thread-pools is imported.
# HuggingFace tokenizers (Rust) + UMAP's internal forking causes segfaults otherwise.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.runners.frozen import main


if __name__ == "__main__":
    main()
