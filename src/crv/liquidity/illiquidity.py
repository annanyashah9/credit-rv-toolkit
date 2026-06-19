"""Daily illiquidity measures, computed on the daily panel and sampled (backward) to
rebalance dates so they are strictly trailing (no look-ahead).

- Bao-Pan-Wang gamma: gamma = -Cov(ΔP_t, ΔP_{t-1}); transitory price reversals from
  illiquidity make consecutive price changes negatively autocorrelated, so gamma > 0
  for illiquid bonds. Needs only daily prices.
- Amihud: average |daily return| per unit dollar volume; high = illiquid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_by_cusip(df: pd.DataFrame, col: str, window: int, min_obs: int) -> pd.Series:
    """Rolling mean of `col` within each cusip over `window` rows (trading days)."""
    return (
        df.groupby("cusip", sort=False)[col]
        .rolling(window=window, min_periods=min_obs)
        .mean()
        .reset_index(level=0, drop=True)
    )


def daily_illiquidity(
    panel: pd.DataFrame, window: int, min_obs: int,
    price_col: str = "price", dvol_col: str = "dvolume", date_col: str = "date",
) -> pd.DataFrame:
    """Return [cusip, date, bao_gamma, amihud] on the daily grid.

    Both are trailing rolling means within each cusip; rows before `min_obs` history
    are NaN.
    """
    p = panel[["cusip", date_col, price_col, dvol_col]].sort_values(["cusip", date_col])
    p = p.rename(columns={date_col: "date"})

    dp = p.groupby("cusip", sort=False)[price_col].diff()
    dp_lag = dp.groupby(p["cusip"], sort=False).shift(1)
    p["_neg_autocov"] = -(dp * dp_lag)            # per-day product; rolling-mean -> -Cov

    ret = p.groupby("cusip", sort=False)[price_col].pct_change()
    dvol = p[dvol_col].replace(0, np.nan)
    p["_amihud_daily"] = (ret.abs() / dvol)

    out = pd.DataFrame({"cusip": p["cusip"], "date": p["date"]})
    out["bao_gamma"] = _rolling_by_cusip(p, "_neg_autocov", window, min_obs).values
    out["amihud"] = _rolling_by_cusip(p, "_amihud_daily", window, min_obs).values
    return out


def sample_asof(daily: pd.DataFrame, rebal_dates, by_cusip: pd.Series) -> pd.DataFrame:
    """Backward as-of sample of the daily illiquidity onto (cusip, rebalance_date).

    Implemented as a per-cusip merge_asof so each membership row gets the most recent
    trailing measure at or before its rebalance date.
    """
    daily = daily.sort_values("date")
    members = pd.DataFrame({"cusip": by_cusip.values,
                            "rebalance_date": pd.to_datetime(rebal_dates)}).sort_values(
        "rebalance_date")
    merged = pd.merge_asof(
        members, daily, left_on="rebalance_date", right_on="date", by="cusip",
        direction="backward",
    )
    return merged[["cusip", "rebalance_date", "bao_gamma", "amihud"]]
