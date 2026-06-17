"""Phase-1 signal orchestration: universe membership -> spreads -> naive residual.

Everything runs on the monthly membership grid (cusip x rebalance_date), which keeps
the cross-sectional fits small and point-in-time. The output `z` is the naive
cheap/rich signal that feeds the Phase 1.5 thin backtest.
"""

from __future__ import annotations

import pandas as pd

from crv.config import Config
from crv.fairvalue.linear import fit_predict_fair_spread
from crv.fairvalue.residual import standardized_residual
from crv.spreads.dts import dts
from crv.spreads.gspread import g_spread


def compute_spreads(universe: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Add gspread_bp and dts to the universe membership table.

    Uses the as-of ytm/ttm carried by the membership row and the Treasury curve at the
    rebalance date.
    """
    df = universe.copy()
    df["gspread_bp"] = g_spread(
        df, curve, ytm_col="ytm", ttm_col="ttm", date_col="rebalance_date"
    )
    df["dts"] = dts(df["gspread_bp"], df["mod_duration"])
    return df


def make_naive_signal(spreads: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-date OLS fair value + standardized residual.

    Skips cross-sections with fewer than cfg.signal.min_names_per_date usable bonds.
    Returns: cusip, rebalance_date, ttm, sector_ff30, gspread_bp, dts, fair_bp,
    resid_bp, z.
    """
    out = []
    for _asof, grp in spreads.groupby("rebalance_date", sort=True):
        usable = grp["gspread_bp"].notna().sum()
        if usable < cfg.signal.min_names_per_date:
            continue
        fair = fit_predict_fair_spread(grp, y_col="gspread_bp")
        res = standardized_residual(grp["gspread_bp"], fair, robust=cfg.signal.robust_scale)
        block = grp[["cusip", "rebalance_date", "ttm", "sector_ff30", "gspread_bp", "dts"]].copy()
        block["fair_bp"] = fair
        block = pd.concat([block, res], axis=1)
        out.append(block)

    cols = ["cusip", "rebalance_date", "ttm", "sector_ff30", "gspread_bp", "dts",
            "fair_bp", "resid_bp", "z"]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)
