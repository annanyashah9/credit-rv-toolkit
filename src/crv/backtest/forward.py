"""Forward outcome construction (leakage-free).

The Phase-1.5 outcome is the h-month-ahead long-cheap spread-return proxy:

    r_fwd = -(gspread_{t+h} - gspread_t) / 1e4 * mod_dur_t

i.e. spread tightening times duration = price gain on a long position. Rates-neutral
by construction (it's a spread). Carry, costs, defaults are Phase 3.

Alignment is by a dense integer month-ordinal over the sorted rebalance dates, so the
t->t+h match never reaches backward in time and bonds absent at t+h are simply dropped
(no forward fill, no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BPS = 1e4


def add_month_ordinal(signal: pd.DataFrame, date_col: str = "rebalance_date") -> pd.DataFrame:
    """Attach `t_idx`: a dense 0..N-1 index over the sorted unique rebalance dates."""
    dates = np.sort(pd.to_datetime(signal[date_col]).unique())
    ordinal = {d: i for i, d in enumerate(dates)}
    out = signal.copy()
    out["t_idx"] = pd.to_datetime(out[date_col]).map(ordinal).astype("int64")
    return out


def forward_spread_return(
    signal: pd.DataFrame, horizon_m: int, mod_dur_col: str = "dts"
) -> pd.DataFrame:
    """Return [cusip, rebalance_date, t_idx, z, gspread_bp, r_fwd] for one horizon.

    mod_dur is recovered as dts / gspread_bp (DTS = spread_bp * duration); this avoids
    needing to thread mod_duration through the signal table. Rows without a t+h match
    (bond gone) or with unusable inputs are dropped.
    """
    s = add_month_ordinal(signal)
    base_cols = ["cusip", "rebalance_date", "t_idx", "z", "gspread_bp", "dts"]
    base = s[base_cols].copy()

    fut = s[["cusip", "t_idx", "gspread_bp"]].rename(
        columns={"gspread_bp": "gspread_fwd", "t_idx": "t_idx_fwd"}
    )
    # Match each row to its own cusip h steps ahead.
    base["t_idx_fwd"] = base["t_idx"] + horizon_m
    merged = base.merge(
        fut, left_on=["cusip", "t_idx_fwd"], right_on=["cusip", "t_idx_fwd"], how="inner"
    )

    mod_dur = merged["dts"] / merged["gspread_bp"].replace(0, np.nan)
    dspread = (merged["gspread_fwd"] - merged["gspread_bp"]) / BPS
    merged["r_fwd"] = -dspread * mod_dur

    out = merged[["cusip", "rebalance_date", "t_idx", "z", "gspread_bp", "r_fwd"]]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["z", "r_fwd"])
