"""I/O helpers: format-agnostic table reading, parquet stage writes, and a run
manifest for reproducibility (config hash + input hashes + seed)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a tabular file, inferring format from suffix (.csv, .parquet, .pq).

    Deliberately permissive: Phase 0 points this at an unknown panel and we want it
    to just load so we can inventory the columns.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".gz"} and path.stem.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.name!r} (suffix {suffix!r})")


def write_stage(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a pipeline-stage output as parquet, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def file_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes, for the run manifest."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON run manifest (config hash, input hashes, seed, timestamps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
