"""Treasury-curve interpolation.

Given the (date x tenor) curve from ingest.fred, interpolate the par yield at an
arbitrary time-to-maturity. Linear in tenor with flat extrapolation beyond the grid
ends (ttm clipped into [shortest, longest] tenor).

All bond-days on the same date share one curve, so we interpolate per date-group
with a vectorized np.interp — fast and memory-light even at ~30M rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def interp_curve_yield(
    curve: pd.DataFrame, dates: pd.Series, ttm_years: pd.Series | np.ndarray
) -> np.ndarray:
    """For each (date, ttm) pair return the interpolated decimal yield.

    Parameters
    ----------
    curve : DataFrame indexed by date, columns = tenor floats (decimal yields),
            daily-ffilled (see fred.load_treasury_curve).
    dates : observation dates (length n).
    ttm_years : times-to-maturity in years (length n).

    Returns an array of decimal yields; NaN where the date is outside curve history.
    """
    tenors = np.asarray(curve.columns, dtype=float)
    work = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(dates).to_numpy()),
            "ttm": np.clip(np.asarray(ttm_years, dtype=float), tenors[0], tenors[-1]),
        }
    )
    out = np.full(len(work), np.nan)

    # Curve rows indexed by date for O(1) lookup; dates absent from the curve -> skip.
    cmat = curve.to_numpy()
    pos = {d: i for i, d in enumerate(curve.index)}

    for date, grp in work.groupby("date", sort=False):
        i = pos.get(pd.Timestamp(date))
        if i is None:
            continue
        out[grp.index.to_numpy()] = np.interp(grp["ttm"].to_numpy(), tenors, cmat[i])
    return out
