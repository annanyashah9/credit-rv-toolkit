"""Transaction-cost table from measured bid/ask.

One-way half-spread (return units) = (ask - bid) / (ask + bid). Computed on the daily
panel, smoothed (trailing mean), sampled backward to each rebalance date, then a
bucketed median fallback fills names without a quote (~30% of rows). Kept methodologically
DISTINCT from the Bao-gamma liquidity control so costs can't launder alpha (Risk #3).

The portfolio charges `|Δweight| * half_spread` per name per rebalance (a round trip
emerges over a buy and a later sell).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import Config


def _daily_half_spread(panel: pd.DataFrame, window: int, min_obs: int) -> pd.DataFrame:
    p = panel[["cusip", "date", "bid", "ask"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    valid = (p["bid"] > 0) & (p["ask"] > 0) & (p["ask"] >= p["bid"])
    p["hs"] = np.where(valid, (p["ask"] - p["bid"]) / (p["ask"] + p["bid"]), np.nan)
    p = p.sort_values(["cusip", "date"])
    p["hs_smooth"] = (
        p.groupby("cusip", sort=False)["hs"]
        .rolling(window=window, min_periods=min_obs).mean()
        .reset_index(level=0, drop=True)
    )
    return p[["cusip", "date", "hs_smooth"]].dropna(subset=["hs_smooth"])


def _asof(daily: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Backward as-of of daily hs_smooth onto (cusip, rebalance_date)."""
    d = daily.sort_values("date")
    m = members[["cusip", "rebalance_date"]].sort_values("rebalance_date")
    merged = pd.merge_asof(m, d, left_on="rebalance_date", right_on="date", by="cusip",
                           direction="backward")
    return merged.set_index(members.index)["hs_smooth"]


def build_cost_table(panel: pd.DataFrame, universe: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return [cusip, rebalance_date, half_spread, cost_source].

    Measured where a trailing quote exists; otherwise a bucketed median fallback
    (rebalance_date x sector x duration-tertile, backing off to date, then global).
    """
    lc = cfg.liquidity
    daily = _daily_half_spread(panel, lc.window_days, lc.min_obs)

    u = universe.reset_index(drop=True).copy()
    u["half_spread"] = _asof(daily, u).values
    u["cost_source"] = np.where(u["half_spread"].notna(), "measured", "fallback")

    # Duration tertile per date for the fallback bucket.
    u["dur_bucket"] = (
        u.groupby("rebalance_date")["mod_duration"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop"))
    )
    for keys in (["rebalance_date", "sector_ff30", "dur_bucket"],
                 ["rebalance_date", "sector_ff30"], ["rebalance_date"]):
        med = u.groupby(keys)["half_spread"].transform("median")
        u["half_spread"] = u["half_spread"].fillna(med)
    u["half_spread"] = u["half_spread"].fillna(u["half_spread"].median())

    return u[["cusip", "rebalance_date", "half_spread", "cost_source"]]
