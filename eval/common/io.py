from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_results_tsv(path: str | Path) -> pd.DataFrame:
    """Read one classification results TSV and verify core columns."""
    df = pd.read_csv(path, sep="\t", dtype={"paper_id": str})
    required = {"paper_id", "classification"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {path!r}: {sorted(missing)}. Found: {list(df.columns)}"
        )
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    return df


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"paper_id": str})
    df["paper_id"] = df["paper_id"].astype(str).str.strip()
    return df


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
