"""Thin quintile diagnostic (Phase 1.5 only).

Sort each cross-section into n quantiles by signal; report mean AND median forward
return per quantile plus the top-minus-bottom long-short, and the fraction of the
cheapest bucket that blows up. The mean-vs-median gap exposes distressed-tail
contamination of the cheap bucket.

INTENTIONALLY NAIVE: no duration/DTS/sector neutralization, no transaction costs, no
default handling. Those are Phase 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.forward import forward_spread_return


def quintile_returns(
    signal: pd.DataFrame, horizon: int, n_quantiles: int = 5, min_names: int = 25
) -> pd.DataFrame:
    """Per-quantile mean & median forward return + an 'LS' (top-minus-bottom) row.

    Quantile 1 = richest .. n = cheapest. Means/medians are averaged over dates.
    """
    fwd = forward_spread_return(signal, horizon)
    mean_parts, med_parts, ls_mean, ls_med = [], [], [], []
    for _date, g in fwd.groupby("rebalance_date", sort=True):
        sub = g[["z", "r_fwd"]].dropna()
        if len(sub) < min_names or sub["z"].nunique() < n_quantiles:
            continue
        q = pd.qcut(sub["z"].rank(method="first"), n_quantiles, labels=False) + 1
        m = sub["r_fwd"].groupby(q).mean()
        md = sub["r_fwd"].groupby(q).median()
        mean_parts.append(m)
        med_parts.append(md)
        if {1, n_quantiles} <= set(m.index):
            ls_mean.append(m[n_quantiles] - m[1])
            ls_med.append(md[n_quantiles] - md[1])

    if not mean_parts:
        return pd.DataFrame(columns=["mean_r", "median_r"])

    out = pd.DataFrame(
        {
            "mean_r": pd.concat(mean_parts, axis=1).mean(axis=1),
            "median_r": pd.concat(med_parts, axis=1).mean(axis=1),
        }
    )
    out.index.name = "quantile"
    out.loc["LS"] = [np.mean(ls_mean) if ls_mean else np.nan,
                     np.mean(ls_med) if ls_med else np.nan]
    return out


def tail_loss_fraction(
    signal: pd.DataFrame, horizon: int, n_quantiles: int = 5, thresh: float = -0.05,
    min_names: int = 25,
) -> float:
    """Fraction of the cheapest quantile whose forward return is below `thresh`
    (the distressed blow-up tail that contaminates the cheap bucket)."""
    fwd = forward_spread_return(signal, horizon)
    top = []
    for _date, g in fwd.groupby("rebalance_date", sort=True):
        sub = g[["z", "r_fwd"]].dropna()
        if len(sub) < min_names or sub["z"].nunique() < n_quantiles:
            continue
        q = pd.qcut(sub["z"].rank(method="first"), n_quantiles, labels=False) + 1
        top.append(sub.loc[q == n_quantiles, "r_fwd"])
    if not top:
        return float("nan")
    allr = pd.concat(top)
    return float((allr < thresh).mean())
