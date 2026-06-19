"""Signal orchestration: universe membership -> spreads -> standardized residual.

Two fair-value models share the monthly (cusip x rebalance_date) grid and the same
per-date, point-in-time cross-sectional fit:
- 'naive'       (Phase 1):   OLS on term+sector in bp space.
- 'peer_shrunk' (Phase 2a):  OLS on term+sector+liquidity in asinh space + empirical-
  Bayes issuer shrinkage; the residual is liquidity-neutral by construction.

Output columns are identical across models so the backtest is model-agnostic:
cusip, rebalance_date, ttm, sector_ff30, gspread_bp, dts, fair_bp, resid_bp, z.
"""

from __future__ import annotations

import pandas as pd

from crv.config import Config
from crv.fairvalue.linear import fit_predict_fair_spread, peer_shrunk_fair_spread
from crv.fairvalue.residual import standardized_residual
from crv.fairvalue.transform import from_model_space, to_model_space
from crv.spreads.dts import dts
from crv.spreads.gspread import g_spread

_OUT_COLS = ["cusip", "rebalance_date", "ttm", "sector_ff30", "gspread_bp", "dts",
             "fair_bp", "resid_bp", "z"]


def compute_spreads(universe: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    """Add gspread_bp and dts to the universe membership table."""
    df = universe.copy()
    df["gspread_bp"] = g_spread(df, curve, ytm_col="ytm", ttm_col="ttm",
                                date_col="rebalance_date")
    df["dts"] = dts(df["gspread_bp"], df["mod_duration"])
    return df


def make_naive_signal(spreads: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-date naive OLS fair value + standardized residual (Phase 1)."""
    out = []
    for _asof, grp in spreads.groupby("rebalance_date", sort=True):
        if grp["gspread_bp"].notna().sum() < cfg.signal.min_names_per_date:
            continue
        fair = fit_predict_fair_spread(grp, y_col="gspread_bp")
        res = standardized_residual(grp["gspread_bp"], fair, robust=cfg.signal.robust_scale)
        block = grp[["cusip", "rebalance_date", "ttm", "sector_ff30", "gspread_bp", "dts"]].copy()
        block["fair_bp"] = fair
        block = pd.concat([block, res], axis=1)
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=_OUT_COLS)


def make_peer_shrunk_signal(
    spreads: pd.DataFrame, liquidity: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Per-date peer-shrunk, liquidity-controlled fair value + residual (Phase 2a)."""
    scale = cfg.model.asinh_scale
    df = spreads.merge(liquidity, on=["cusip", "rebalance_date"], how="left")
    df["y_model"] = to_model_space(df["gspread_bp"], scale)

    out = []
    for _asof, grp in df.groupby("rebalance_date", sort=True):
        if grp["gspread_bp"].notna().sum() < cfg.signal.min_names_per_date:
            continue
        fair_model = peer_shrunk_fair_spread(grp, y_col="y_model", issuer_col="issuer",
                                             k=cfg.model.shrink_k)
        # Standardize the residual in model (asinh) space -> z.
        res = standardized_residual(grp["y_model"], fair_model, robust=cfg.signal.robust_scale)
        fair_bp = from_model_space(fair_model, scale)
        block = grp[["cusip", "rebalance_date", "ttm", "sector_ff30", "gspread_bp", "dts"]].copy()
        block["fair_bp"] = fair_bp
        block["resid_bp"] = block["gspread_bp"] - fair_bp
        block["z"] = res["z"].values
        out.append(block)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=_OUT_COLS)


def make_signal(spreads: pd.DataFrame, liquidity: pd.DataFrame | None, cfg: Config) -> pd.DataFrame:
    """Dispatch to the configured fair-value model."""
    if cfg.model.kind == "peer_shrunk":
        if liquidity is None:
            raise ValueError("peer_shrunk model requires the liquidity feature table")
        return make_peer_shrunk_signal(spreads, liquidity, cfg)
    return make_naive_signal(spreads, cfg)
