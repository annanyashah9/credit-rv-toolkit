"""FRED ingest: Treasury constant-maturity yield curve + SOFR.

Uses the keyless `fredgraph.csv` endpoint (no API key required), caching each series
to data/raw/fred/. Treasury CMT yields are published in percent; we convert to
decimal so they line up with the panel's `ytm` (also decimal).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from crv.config import Config

FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Treasury constant-maturity series -> tenor in years. This grid defines the curve
# we interpolate G-spreads against.
TREASURY_TENORS: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}


def fetch_fred_series(series_id: str, cache_dir: Path, refresh: bool = False) -> pd.Series:
    """Download one FRED series as a date-indexed Series (percent units).

    Cached as CSV under cache_dir; pass refresh=True to re-download.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{series_id}.csv"
    if refresh or not cache.exists():
        resp = requests.get(FREDGRAPH.format(series=series_id), timeout=60)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    raw = pd.read_csv(cache, na_values=["."])
    # fredgraph uses "observation_date" (newer) or "DATE" (older) for the date col.
    date_col = "observation_date" if "observation_date" in raw.columns else raw.columns[0]
    val_col = [c for c in raw.columns if c != date_col][0]
    s = pd.Series(
        pd.to_numeric(raw[val_col], errors="coerce").values,
        index=pd.to_datetime(raw[date_col]),
        name=series_id,
    )
    return s.dropna()


def load_treasury_curve(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Treasury par-yield curve as a (date x tenor_years) DataFrame in DECIMAL yield.

    Forward-filled across non-publication days so any bond date can be priced.
    Columns are tenor floats (sorted); index is daily dates.
    """
    cache_dir = cfg.paths.raw / "fred"
    cols = {}
    for series, tenor in TREASURY_TENORS.items():
        cols[tenor] = fetch_fred_series(series, cache_dir, refresh=refresh) / 100.0  # %->decimal
    curve = pd.DataFrame(cols).sort_index(axis=1).sort_index()
    # Daily grid + ffill so weekends/holidays inherit the last published curve.
    full = pd.date_range(curve.index.min(), curve.index.max(), freq="D")
    return curve.reindex(full).ffill()


def load_sofr(cfg: Config, refresh: bool = False) -> pd.Series:
    """SOFR overnight rate (decimal). Not used by Phase 1 G-spread; kept for Phase 2+."""
    s = fetch_fred_series("SOFR", cfg.paths.raw / "fred", refresh=refresh) / 100.0
    return s.rename("sofr")
