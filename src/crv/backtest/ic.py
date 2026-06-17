"""Information coefficient: per-date cross-sectional correlation of signal vs forward
outcome, summarized over time with HAC standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.forward import forward_spread_return
from crv.backtest.stats import newey_west_mean


def cross_sectional_ic(
    df: pd.DataFrame,
    signal_col: str = "z",
    ret_col: str = "r_fwd",
    method: str = "spearman",
    date_col: str = "rebalance_date",
    min_names: int = 20,
) -> pd.Series:
    """IC per rebalance date (date-indexed Series). Cross-sections thinner than
    min_names are skipped."""
    out = {}
    for date, g in df.groupby(date_col, sort=True):
        sub = g[[signal_col, ret_col]].dropna()
        if len(sub) < min_names or sub[signal_col].nunique() < 2:
            continue
        out[date] = sub[signal_col].corr(sub[ret_col], method=method)
    return pd.Series(out, name="ic").sort_index()


def _winsorize(s: pd.Series, cap: float) -> pd.Series:
    return s.clip(lower=-cap, upper=cap)


def ic_table(
    signal: pd.DataFrame,
    horizons=(1, 3, 6),
    method: str = "spearman",
    winsor_z: float | None = None,
) -> pd.DataFrame:
    """One row per horizon: mean IC, HAC SE, t-stat, n_periods, hit_rate, and a
    winsorized-signal robustness IC.

    HAC lag = horizon - 1 (overlap length of the forward windows).
    """
    rows = []
    for h in horizons:
        fwd = forward_spread_return(signal, h)
        ic = cross_sectional_ic(fwd, method=method)
        mean, se, t = newey_west_mean(ic, lags=h - 1)
        row = {
            "horizon_m": h,
            "mean_ic": mean,
            "hac_se": se,
            "t_stat": t,
            "n_periods": int(ic.notna().sum()),
            "hit_rate": float((ic > 0).mean()) if len(ic) else np.nan,
        }
        if winsor_z is not None:
            fwd_w = fwd.assign(z=_winsorize(fwd["z"], winsor_z))
            ic_w = cross_sectional_ic(fwd_w, method=method)
            row["mean_ic_winsor"] = float(ic_w.mean()) if len(ic_w) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def ic_timeseries(signal: pd.DataFrame, horizon: int, method: str = "spearman") -> pd.Series:
    """The per-date IC series for one horizon (for plotting)."""
    return cross_sectional_ic(forward_spread_return(signal, horizon), method=method)
