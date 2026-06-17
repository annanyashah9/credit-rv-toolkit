"""Point-in-time universe construction.

Membership is evaluated as-of each rebalance date (month-end by default) from
panel-derivable rules only -- survivorship-safe because each date sees only data up
to that date. Curated filters (bullet-only, rating-tier, seniority) plug in via the
hand-collected reference and are no-ops until that CSV exists (candidate-pool-now).

Canonical column names are assumed (see ingest.obap.load_panel): cusip, date, ytm,
mod_duration, maturity, issue_size, sector_ff30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import Config
from crv.ingest.reference import pit_reference_asof

REQUIRED = ["cusip", "date", "ytm", "mod_duration", "maturity", "issue_size", "sector_ff30"]


def rebalance_dates(panel: pd.DataFrame, freq: str) -> pd.DatetimeIndex:
    """Period-end dates spanning the panel, snapped to the freq grid."""
    d = pd.to_datetime(panel["date"])
    return pd.date_range(d.min(), d.max(), freq=freq)


def _window_slice(
    panel_sorted: pd.DataFrame, date_vals: np.ndarray, asof: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    """Rows with date in (asof - window_days, asof], via searchsorted on sorted dates."""
    lo = np.datetime64(asof - pd.Timedelta(days=window_days))
    hi = np.datetime64(asof)
    i = np.searchsorted(date_vals, lo, side="right")
    j = np.searchsorted(date_vals, hi, side="right")
    return panel_sorted.iloc[i:j]


def build_candidate_universe(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Membership table: one row per (cusip, rebalance_date) that PASSES panel-only
    rules, with the as-of attributes used downstream.

    Columns: cusip, rebalance_date, n_days, trade_freq, issue_size, ttm,
    sector_ff30, ytm, mod_duration.
    """
    u = cfg.universe
    missing = [c for c in REQUIRED if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing canonical columns {missing}; set ingest.obap_column_map")

    ps = panel.sort_values("date", kind="stable")
    date_vals = pd.to_datetime(ps["date"]).to_numpy()
    rebals = rebalance_dates(panel, u.rebalance_freq)

    rows = []
    for asof in rebals:
        win = _window_slice(ps, date_vals, asof, u.window_days)
        if win.empty:
            continue
        n_bdays = max(len(pd.bdate_range(asof - pd.Timedelta(days=u.window_days), asof)), 1)
        # As-of attributes = each bond's most recent obs within the window.
        last = win.groupby("cusip", sort=False).agg(
            n_days=("date", "nunique"),
            issue_size=("issue_size", "last"),
            ttm=("maturity", "last"),
            sector_ff30=("sector_ff30", "last"),
            ytm=("ytm", "last"),
            mod_duration=("mod_duration", "last"),
        )
        last["trade_freq"] = last["n_days"] / n_bdays
        keep = (
            (last["issue_size"] >= u.size_floor)
            & (last["trade_freq"] >= u.min_trade_freq)
            & last["ttm"].between(u.min_ttm, u.max_ttm)
            & last["ytm"].notna()
            & last["mod_duration"].notna()
            & last["sector_ff30"].notna()
        )
        sel = last[keep].reset_index()
        sel.insert(1, "rebalance_date", asof)
        rows.append(sel)

    if not rows:
        return pd.DataFrame(
            columns=["cusip", "rebalance_date", "n_days", "trade_freq", "issue_size",
                     "ttm", "sector_ff30", "ytm", "mod_duration"]
        )
    return pd.concat(rows, ignore_index=True)


def apply_curated_filters(
    candidates: pd.DataFrame,
    reference: pd.DataFrame,
    max_rating_num: float | None = None,
) -> pd.DataFrame:
    """Apply bullet-only / rating-tier / seniority filters using the reference.

    No-op when `reference` is empty (candidate-pool-now). When present, applied
    per rebalance_date so rating/seniority are point-in-time.

    max_rating_num : if set, drop bonds whose numeric rating exceeds it (worse than
    the distressed cutoff on an increasing AAA=1.. scale). None => no rating filter.
    """
    if reference.empty or candidates.empty:
        return candidates

    out = []
    for asof, grp in candidates.groupby("rebalance_date", sort=False):
        ref = pit_reference_asof(reference, asof)
        if ref.empty:
            continue
        merged = grp.merge(ref[["cusip", "is_bullet", "rating_num", "seniority"]], on="cusip")
        merged = merged[merged["is_bullet"].fillna(False)]
        if max_rating_num is not None:
            merged = merged[merged["rating_num"].notna() & (merged["rating_num"] <= max_rating_num)]
        out.append(merged)
    return pd.concat(out, ignore_index=True) if out else candidates.iloc[0:0]
