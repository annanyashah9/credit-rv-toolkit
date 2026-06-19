"""Neutralized long-short portfolio construction and P&L.

Pipeline per rebalance date:
  1. neutralize the signal: residualize z on [sector dummies, mod_dur, dts] so the book
     is (approximately) duration/DTS/sector neutral by construction.
  2. dollar-neutral quintile long-short: long top quintile, short bottom, equal-weight,
     sum(long)=+1, sum(short)=-1.
  3. overlapping book (Jegadeesh-Titman): the active book = mean of the last
     `holding_months` monthly weight vectors, so a name is held for h months.
  4. monthly P&L: gross = book . r1 ; cost = |Δbook| . half_spread ; net = gross - cost.

Realized net duration/DTS/sector exposures are reported so neutrality is verified, not
assumed. A long-only variant (long minus universe) is also returned, since shorting
corporates is operationally hard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def neutralize_signal(df: pd.DataFrame, axes: list[str]) -> pd.Series:
    """Cross-sectional residual of z on the neutralization axes (within one date)."""
    y = pd.to_numeric(df["z"], errors="coerce")
    cols = []
    if "sector" in axes:
        cols.append(pd.get_dummies(df["sector_ff30"].astype("category"), drop_first=True,
                                   dtype=float))
    num = []
    if "duration" in axes:
        num.append("mod_duration")
    if "dts" in axes:
        num.append("dts")
    parts = [pd.DataFrame({"const": 1.0}, index=df.index)]
    if num:
        parts.append(df[num].astype(float))
    parts.extend(cols)
    X = pd.concat(parts, axis=1)
    ok = X.notna().all(axis=1) & y.notna()
    resid = pd.Series(np.nan, index=df.index)
    if ok.sum() < X.shape[1] + 2:
        return y  # too thin to neutralize; fall back to raw z
    beta, *_ = np.linalg.lstsq(X[ok].to_numpy(), y[ok].to_numpy(), rcond=None)
    resid[ok] = y[ok].to_numpy() - X[ok].to_numpy() @ beta
    return resid


def long_short_weights(zn: pd.Series, n_quantiles: int) -> pd.Series:
    """Dollar-neutral equal-weight quintile L/S weights (sum long=+1, short=-1)."""
    w = pd.Series(0.0, index=zn.index)
    valid = zn.dropna()
    if valid.nunique() < n_quantiles:
        return w
    q = pd.qcut(valid.rank(method="first"), n_quantiles, labels=False)
    top, bot = q == (n_quantiles - 1), q == 0
    if top.sum():
        w.loc[valid.index[top]] = 1.0 / top.sum()
    if bot.sum():
        w.loc[valid.index[bot]] = -1.0 / bot.sum()
    return w


def long_only_weights(zn: pd.Series, n_quantiles: int) -> pd.Series:
    """Long top-quintile (sum=+1) minus equal-weight universe (sum=-1): a market-neutral
    long-cheap tilt that avoids name-level shorting."""
    w = pd.Series(0.0, index=zn.index)
    valid = zn.dropna()
    if valid.nunique() < n_quantiles:
        return w
    q = pd.qcut(valid.rank(method="first"), n_quantiles, labels=False)
    top = q == (n_quantiles - 1)
    w.loc[valid.index] = -1.0 / len(valid)
    if top.sum():
        w.loc[valid.index[top]] += 1.0 / top.sum()
    return w


def run_portfolio(df: pd.DataFrame, r1: pd.DataFrame, costs: pd.DataFrame, cfg, style: str):
    """Build the overlapping book and monthly P&L for one style ('ls' or 'long_only').

    df: per (cusip, rebalance_date) with z, sector_ff30, mod_duration, dts.
    Returns a date-indexed DataFrame: gross, cost, net, turnover, breadth, net_dur, net_dts.
    """
    bt = cfg.backtest
    h, nq, axes = bt.holding_months, bt.n_quantiles, bt.neutralize
    weight_fn = long_short_weights if style == "ls" else long_only_weights

    r1_by = {d: g.set_index("cusip")["r1"] for d, g in r1.groupby("rebalance_date")}
    hs_by = {d: g.set_index("cusip")["half_spread"] for d, g in costs.groupby("rebalance_date")}

    dates = sorted(df["rebalance_date"].unique())
    targets, attrs = {}, {}
    for d, g in df.groupby("rebalance_date"):
        gi = g.set_index("cusip")
        zn = neutralize_signal(gi.assign(z=gi["z"]), axes)
        targets[d] = weight_fn(zn, nq)
        attrs[d] = gi[["mod_duration", "dts"]]

    recent: list[pd.Series] = []
    rows = []
    prev_book = pd.Series(dtype=float)
    for d in dates:
        recent.append(targets[d])
        if len(recent) > h:
            recent.pop(0)
        book = pd.concat(recent, axis=1).fillna(0.0).mean(axis=1)
        book = book[book != 0.0]

        r1_t = r1_by.get(d, pd.Series(dtype=float))
        gross = float((book * r1_t.reindex(book.index)).fillna(0.0).sum())

        all_names = book.index.union(prev_book.index)
        dbook = book.reindex(all_names).fillna(0.0) - prev_book.reindex(all_names).fillna(0.0)
        hs_t = hs_by.get(d, pd.Series(dtype=float)).reindex(all_names).fillna(0.0)
        cost = float((dbook.abs() * hs_t).sum())

        at = attrs[d].reindex(book.index)
        rows.append({
            "rebalance_date": d,
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
            "turnover": float(dbook.abs().sum()),
            "breadth": int((book != 0).sum()),
            "net_dur": float((book * at["mod_duration"]).sum()),
            "net_dts": float((book * at["dts"]).sum()),
        })
        prev_book = book

    return pd.DataFrame(rows).set_index("rebalance_date")
