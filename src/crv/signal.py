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

import numpy as np
import pandas as pd

from crv.backtest.windows import walk_forward_windows
from crv.config import Config
from crv.fairvalue.features import build_design_matrix
from crv.fairvalue.linear import fit_predict_fair_spread, peer_shrunk_fair_spread
from crv.fairvalue.models import make_model_factory
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


def _standardize_block(grp: pd.DataFrame, fair_model: np.ndarray, scale: float,
                       robust: bool) -> pd.DataFrame:
    """Assemble a signal block from a cross-section + model-space fair prediction."""
    res = standardized_residual(grp["y_model"], pd.Series(fair_model, index=grp.index),
                                robust=robust)
    fair_bp = from_model_space(fair_model, scale)
    block = grp[["cusip", "rebalance_date", "ttm", "sector_ff30", "gspread_bp", "dts"]].copy()
    block["fair_bp"] = fair_bp
    block["resid_bp"] = block["gspread_bp"] - fair_bp
    block["z"] = res["z"].values
    return block


def make_walkforward_signal(
    spreads: pd.DataFrame, liquidity: pd.DataFrame, cfg: Config, kind: str
) -> pd.DataFrame:
    """Out-of-sample walk-forward signal for a pooled model (ridge_wf / gbm_wf).

    A model is refit on a trailing window of PAST cross-sections and predicts each
    current cross-section (no look-ahead). The design matrix is built once over the
    full panel so train/predict feature columns are identical.
    """
    m = cfg.model
    scale = m.asinh_scale
    df = spreads.merge(liquidity, on=["cusip", "rebalance_date"], how="left")
    df["y_model"] = to_model_space(df["gspread_bp"], scale)
    df = df.sort_values("rebalance_date").reset_index(drop=True)

    X = build_design_matrix(df, feature_set="full")
    y = df["y_model"]
    ok = X.notna().all(axis=1) & y.notna()
    Xv, yv = X.to_numpy(), y.to_numpy()
    dates = df["rebalance_date"]

    factory = make_model_factory(kind, cfg)
    windows = walk_forward_windows(
        dates, m.train_window_months, m.min_train_months, m.refit_every_months, m.train_scheme
    )

    out = []
    for train_dates, predict_dates in windows:
        tr = ok & dates.isin(train_dates)
        if tr.sum() < cfg.signal.min_names_per_date:
            continue
        model = factory()
        model.fit(Xv[tr.to_numpy()], yv[tr.to_numpy()])
        for asof in predict_dates:
            sel = ok & (dates == asof)
            if sel.sum() < cfg.signal.min_names_per_date:
                continue
            grp = df[sel]
            fair_model = model.predict(Xv[sel.to_numpy()])
            out.append(_standardize_block(grp, fair_model, scale, cfg.signal.robust_scale))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=_OUT_COLS)


def make_signal(
    spreads: pd.DataFrame, liquidity: pd.DataFrame | None, cfg: Config
) -> pd.DataFrame:
    """Dispatch to the configured fair-value model."""
    kind = cfg.model.kind
    if kind in ("ridge_wf", "gbm_wf"):
        if liquidity is None:
            raise ValueError(f"{kind} requires the liquidity feature table")
        return make_walkforward_signal(spreads, liquidity, cfg, kind)
    if kind == "peer_shrunk":
        if liquidity is None:
            raise ValueError("peer_shrunk model requires the liquidity feature table")
        return make_peer_shrunk_signal(spreads, liquidity, cfg)
    return make_naive_signal(spreads, cfg)
