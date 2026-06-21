"""Hand-collected reference data (ratings, call schedules, seniority).

PIT-ready schema with effective_date / thru_date so a static v1 fill can later be
backfilled to point-in-time without restructuring. The file does not exist yet
(candidate-pool-now): every loader tolerates a missing/empty file and returns an
empty frame, which makes the curated universe filters no-ops.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REFERENCE_COLUMNS = [
    "cusip",
    "effective_date",
    "thru_date",
    "rating",
    "rating_num",      # numeric scale, e.g. AAA=1 ... for tier comparisons
    "is_bullet",       # True for bullets (kept in v1 universe)
    "seniority",       # e.g. SR_UNSECURED, SR_SECURED, SUBORDINATED
    "next_call_date",
    "call_price",
]


def empty_reference() -> pd.DataFrame:
    return pd.DataFrame(columns=REFERENCE_COLUMNS)


def load_reference(path: str | Path) -> pd.DataFrame:
    """Load the ratings/calls CSV, or an empty frame if it's absent/empty."""
    path = Path(path)
    if not path.exists():
        return empty_reference()
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    if df.empty:
        return empty_reference()
    for col in ("effective_date", "thru_date", "next_call_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def pit_reference_asof(ref: pd.DataFrame, asof) -> pd.DataFrame:
    """Rows in effect at `asof` (effective_date <= asof < thru_date).

    Open-ended thru_date (NaT) means still in effect. Empty in -> empty out.
    """
    if ref.empty:
        return ref
    asof = pd.Timestamp(asof)
    eff = ref["effective_date"].fillna(pd.Timestamp.min)
    thru = ref["thru_date"].fillna(pd.Timestamp.max)
    return ref[(eff <= asof) & (asof < thru)].copy()
