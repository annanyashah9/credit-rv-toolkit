"""Standardized residual = the cheap/rich signal.

residual = observed - fair (bps). Standardized per cross-section by a robust scale
(MAD) so the z-score is comparable across dates. Positive z => cheap (spread wider
than fair => higher yield => candidate long); negative z => rich.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_TO_SIGMA = 1.4826  # MAD * this ~= std for normal data


def standardized_residual(
    observed: pd.Series, fair: pd.Series, robust: bool = True
) -> pd.DataFrame:
    """Return a frame with resid_bp and z (standardized within this cross-section)."""
    resid = pd.to_numeric(observed, errors="coerce") - pd.to_numeric(fair, errors="coerce")
    if robust:
        med = resid.median()
        scale = _MAD_TO_SIGMA * (resid - med).abs().median()
    else:
        med = resid.mean()
        scale = resid.std(ddof=0)
    z = (resid - med) / scale if scale and np.isfinite(scale) and scale > 0 else resid * np.nan
    return pd.DataFrame({"resid_bp": resid, "z": z})
