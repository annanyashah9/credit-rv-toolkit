"""Default / exit detection (free-data heuristic).

No clean default flag exists, so we classify each bond's exit from the panel itself,
using the maturity field to disambiguate the ambiguous "left the panel" event:

- MATURED  : last observed remaining-maturity ~ 0 (reached maturity).
- DEFAULTED: clean price fell below `distress_floor` AND the bond left the panel well
             before maturity (early, distressed exit). default_date = first sub-floor date.
- OTHER    : early exit without distress (called/tendered/data gap) — no default loss.
- ACTIVE   : still trading at the panel end.

Defaulted names are assigned recovery and CARRIED through returns (handled in
backtest.returns), never silently dropped — the survivorship-bias fix.
"""

from __future__ import annotations

import pandas as pd

from crv.config import Config

MATURED, DEFAULTED, OTHER, ACTIVE = "MATURED", "DEFAULTED", "OTHER", "ACTIVE"


def detect_exits(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return [cusip, exit_type, last_date, default_date] (one row per cusip)."""
    floor = cfg.backtest.distress_floor
    p = panel[["cusip", "date", "price", "maturity"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    panel_end = p["date"].max()

    g = p.groupby("cusip", sort=False)
    agg = g.agg(last_date=("date", "max"), last_ttm=("maturity", "min"),
                min_price=("price", "min"))
    # First date the clean price breaches the distress floor (NaT if never).
    first_distress = (
        p[p["price"] < floor].groupby("cusip")["date"].min().rename("default_date")
    )
    agg = agg.join(first_distress)

    early_exit = agg["last_date"] < (panel_end - pd.Timedelta(days=90))
    matured = agg["last_ttm"] <= 0.15
    distressed = agg["min_price"] < floor

    exit_type = pd.Series(ACTIVE, index=agg.index)
    exit_type[early_exit] = OTHER
    exit_type[early_exit & distressed & (agg["last_ttm"] > 0.5)] = DEFAULTED
    exit_type[matured] = MATURED

    out = agg.reset_index()[["cusip", "last_date", "default_date"]]
    out["exit_type"] = exit_type.values
    # default_date only meaningful for DEFAULTED rows.
    out.loc[out["exit_type"] != DEFAULTED, "default_date"] = pd.NaT
    return out
