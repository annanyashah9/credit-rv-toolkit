"""Open Source Bond Asset Pricing (OBAP) panel loader.

Phase 0 reality: we don't yet know the panel's exact schema (column names, units,
frequency). So this module does two things:

  1. `load_raw_panel` -- read the file untouched, for inventory.
  2. `classify_columns` -- heuristically tag each column into a canonical role
     (date / id / price / yield / spread / duration / volume / bid_ask / rating)
     so the Phase 0 inventory can report what's present and what's missing.

Once the user fills `ingest.obap_column_map` in the config (informed by the
inventory), `load_panel` applies the rename to produce canonical columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crv.config import Config
from crv.io import read_table

# Canonical roles we care about downstream, with substring hints for the
# heuristic classifier. Hints are lowercase; matched against lowercased column
# names. Order matters: earlier roles win ties, so SPECIFIC roles come before the
# greedy generic ones ("date" with its "dt" hint must be last, or it steals
# columns like "maturity_dt").
ROLE_HINTS: dict[str, tuple[str, ...]] = {
    "identifier": ("cusip", "isin", "bond_id", "bondid", "issue_id", "permno", "figi"),
    "issuer": ("issuer", "ticker", "company", "obligor", "permco"),
    "maturity": ("maturity", "matur", "mat_dt", "mat_date"),
    "coupon": ("coupon", "cpn"),
    "size": ("issue_size", "amt_out", "amount_out", "offering", "face"),
    "rating": ("rating", "rtg", "grade"),
    "sector": ("sector", "industry", "gics"),
    "duration": ("duration", "dur", "mod_dur", "moddur", "dts"),
    "spread": ("spread", "oas", "gspread", "zspread", "sprd"),
    "yield": ("yield", "ytm", "ytw", "yld"),
    "price": ("price", "prc", "prclean", "prc_clean", "dirty", "clean_prc"),
    "volume": ("volume", "vol", "par_volume", "trd_size", "qty", "amount_traded"),
    "bid_ask": ("bid", "ask", "bidask", "bid_ask", "spread_bid"),
    # Generic date hint last: "dt" matches many names, so let specifics win first.
    "date": ("date", "dt", "month", "period", "asof"),
}


def load_raw_panel(cfg: Config) -> pd.DataFrame:
    """Read the OBAP panel exactly as stored. Raises a clear error if not configured."""
    path = cfg.paths.obap_panel
    if path is None:
        raise FileNotFoundError(
            "paths.obap_panel is not set. Download the OBAP panel into data/raw/ and "
            "set paths.obap_panel in your config (csv or parquet)."
        )
    if not Path(path).exists():
        raise FileNotFoundError(f"OBAP panel not found at {path}")
    return read_table(path)


def classify_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map each column name -> best-guess canonical role (or 'unknown').

    Pure name-based heuristic; the Phase 0 inventory pairs this with dtype/coverage
    stats so a human can confirm or correct it via obap_column_map.
    """
    out: dict[str, str] = {}
    for col in df.columns:
        name = str(col).lower()
        role = "unknown"
        for candidate, hints in ROLE_HINTS.items():
            if any(h in name for h in hints):
                role = candidate
                break
        out[str(col)] = role
    return out


def load_panel(cfg: Config) -> pd.DataFrame:
    """Load the panel and apply the user-confirmed column map to canonical names.

    Used by downstream stages (Phase 1+), once the inventory has informed the map.
    """
    df = load_raw_panel(cfg)
    if cfg.ingest.obap_column_map:
        df = df.rename(columns=cfg.ingest.obap_column_map)
    return df
