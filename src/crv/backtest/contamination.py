"""Contamination tests: is the signal idiosyncratic RV alpha, or is it just predicting
defaults / harvesting an illiquidity premium?

All three are DIAGNOSTICS, not tradeable backtests — `impending_default` in particular
uses knowledge of who eventually defaults (look-ahead), so it only measures how much the
signal's behavior depends on the distressed tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.defaults import DEFAULTED
from crv.backtest.ic import cross_sectional_ic


def _flag_impending_default(signal: pd.DataFrame, exits: pd.DataFrame, window_m: int) -> pd.Series:
    """Boolean per signal row: does this name default within `window_m` months of the
    formation date?"""
    d = exits.loc[exits["exit_type"] == DEFAULTED, ["cusip", "default_date"]]
    s = signal.merge(d, on="cusip", how="left")
    dd = pd.to_datetime(s["default_date"])
    rb = pd.to_datetime(s["rebalance_date"])
    months = (dd - rb).dt.days / 30.44
    return (months >= 0) & (months <= window_m)


def impending_default_test(signal: pd.DataFrame, fwd: pd.DataFrame, exits: pd.DataFrame,
                           cfg) -> dict:
    """Compare rank IC (z vs forward real return) on the full sample vs survivors-only
    (drop names that default within the window at formation)."""
    flag = _flag_impending_default(signal, exits, cfg.backtest.default_window_m)
    merged = signal.assign(_imp=flag.values).merge(fwd, on=["cusip", "rebalance_date"])
    ic_full = cross_sectional_ic(merged).mean()
    ic_surv = cross_sectional_ic(merged[~merged["_imp"]]).mean()
    return {
        "ic_full": float(ic_full),
        "ic_survivors": float(ic_surv),
        "frac_impending": float(flag.mean()),
    }


def default_odds_by_quantile(signal: pd.DataFrame, exits: pd.DataFrame, cfg) -> pd.DataFrame:
    """Fraction of names that default within the window, by z-quintile (does cheap
    predict default?)."""
    flag = _flag_impending_default(signal, exits, cfg.backtest.default_window_m)
    s = signal.assign(_imp=flag.values).dropna(subset=["z"])
    nq = cfg.backtest.n_quantiles
    s = s.assign(q=s.groupby("rebalance_date")["z"].transform(
        lambda x: pd.qcut(x.rank(method="first"), nq, labels=False) if x.nunique() >= nq else np.nan
    ))
    return s.groupby("q")["_imp"].mean().rename("default_rate").to_frame()


def liquidity_split_test(signal: pd.DataFrame, fwd: pd.DataFrame, liquidity: pd.DataFrame,
                         cfg) -> dict:
    """Rank IC within liquid vs illiquid halves (per-date median split on the chosen
    liquidity feature). If the signal only works in illiquid names, it's a premium."""
    metric = cfg.backtest.liquidity_split
    s = signal.merge(liquidity[["cusip", "rebalance_date", metric]],
                     on=["cusip", "rebalance_date"], how="left")
    s = s.merge(fwd, on=["cusip", "rebalance_date"])
    med = s.groupby("rebalance_date")[metric].transform("median")
    illiquid = s[metric] > med            # higher gamma/amihud = more illiquid
    return {
        "ic_liquid": float(cross_sectional_ic(s[~illiquid]).mean()),
        "ic_illiquid": float(cross_sectional_ic(s[illiquid]).mean()),
        "metric": metric,
    }
