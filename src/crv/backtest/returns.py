"""Monthly excess return earned by a position formed at each rebalance date.

Excess return over the next month (rates-neutral; no coupon needed):
    r1 = s_t * (1/12)  -  D_t * (s_{t+1} - s_t)
with s in decimal (gspread_bp/1e4) and D = spread duration (mod_duration). Carry +
spread-change P&L.

Default carry-through: if a held name defaults in the month after formation, its return
is the realized recovery loss `(recovery*par - price_t)/price_t`, and it earns nothing
after (it has left the universe). Defaults are never silently dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.defaults import DEFAULTED
from crv.backtest.forward import add_month_ordinal
from crv.config import Config

BPS = 1e4
PAR = 100.0


def monthly_excess_return(
    spreads: pd.DataFrame, exits: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Return [cusip, rebalance_date, r1] (next-month excess return at each formation date)."""
    rec = cfg.backtest.recovery
    s = add_month_ordinal(spreads)
    ordinal_to_date = dict(enumerate(np.sort(pd.to_datetime(s["rebalance_date"]).unique())))

    base = s[["cusip", "rebalance_date", "t_idx", "gspread_bp", "mod_duration", "price"]].copy()
    fut = s[["cusip", "t_idx", "gspread_bp"]].rename(
        columns={"gspread_bp": "gspread_fwd", "t_idx": "t_idx_fwd"})
    base["t_idx_fwd"] = base["t_idx"] + 1
    m = base.merge(fut, on=["cusip", "t_idx_fwd"], how="left")

    s_t = m["gspread_bp"] / BPS
    ds = (m["gspread_fwd"] - m["gspread_bp"]) / BPS
    m["r1"] = s_t * (1.0 / 12.0) - m["mod_duration"] * ds

    # Default override at the formation month whose forward window contains the default.
    deflt = exits.loc[exits["exit_type"] == DEFAULTED, ["cusip", "default_date"]]
    m = m.merge(deflt, on="cusip", how="left")
    m["next_date"] = (m["t_idx"] + 1).map(ordinal_to_date)
    is_formation = (
        m["default_date"].notna()
        & (m["rebalance_date"] < m["default_date"])
        & (m["default_date"] <= m["next_date"])
    )
    loss = (rec * PAR - m["price"]) / m["price"]
    m.loc[is_formation, "r1"] = loss[is_formation]

    # Drop rows at/after the default month (no position survives), keep the loss row.
    post_default = m["default_date"].notna() & (m["rebalance_date"] >= m["default_date"])
    m = m[~post_default]

    out = m[["cusip", "rebalance_date", "r1"]].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=["r1"])


def forward_excess_return(r1: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """h-month forward realized excess return per (cusip, formation date) = sum of the
    next `horizon` monthly r1 (compounding ≈ summing for small monthly returns).

    Aligns on the global month ordinal (t_idx), so calendar gaps don't get summed as if
    consecutive. Leakage-free: a window survives only if all `horizon` months exist for
    that cusip (a default truncates the series ⇒ that window drops to NaN and is removed).
    """
    s = add_month_ordinal(r1)
    ordinal_to_date = dict(enumerate(np.sort(pd.to_datetime(s["rebalance_date"]).unique())))
    wide = s.pivot_table(index="t_idx", columns="cusip", values="r1").sort_index()
    # Reverse rolling sum so row i = sum of rows i..i+horizon-1 (require all present).
    fwd = wide[::-1].rolling(horizon, min_periods=horizon).sum()[::-1]
    long = fwd.stack().rename("r_fwd").reset_index()
    long["rebalance_date"] = long["t_idx"].map(ordinal_to_date)
    return long[["cusip", "rebalance_date", "r_fwd"]].dropna(subset=["r_fwd"])
