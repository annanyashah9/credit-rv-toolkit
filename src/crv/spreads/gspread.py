"""G-spread: bond yield minus the maturity-matched Treasury par yield, in bps.

G-spread is the Phase-1 signal substrate (Z-spread is off — no coupon column).
`bond_maturity` in the panel is verified to be REMAINING years-to-maturity, so it is
used directly as the term to interpolate the Treasury curve against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.spreads.discount import interp_curve_yield

BPS = 1e4


def g_spread(
    panel: pd.DataFrame,
    curve: pd.DataFrame,
    ytm_col: str = "ytm",
    ttm_col: str = "maturity",
    date_col: str = "date",
) -> pd.Series:
    """Return G-spread in bps for each row of the (canonical-column) panel.

    g_spread = (ytm - treasury_yield(ttm)) * 1e4, with ytm and the curve both in
    decimal. NaN where curve/inputs are missing.
    """
    tsy = interp_curve_yield(curve, panel[date_col], panel[ttm_col].to_numpy())
    gs = (panel[ytm_col].to_numpy() - tsy) * BPS
    return pd.Series(gs, index=panel.index, name="gspread_bp")


def validate_against_panel(gspread_bp: pd.Series, credit_spread_decimal: pd.Series) -> float:
    """Sanity correlation between our G-spread and the panel's own credit_spread.

    credit_spread is decimal; convert to bps. Returns Pearson r over rows where both
    are finite. Expect > ~0.9 if our computation is sound.
    """
    a = pd.to_numeric(gspread_bp, errors="coerce")
    b = pd.to_numeric(credit_spread_decimal, errors="coerce") * BPS
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])
