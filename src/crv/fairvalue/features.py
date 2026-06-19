"""Design-matrix construction for the fair-value model.

- 'naive'  (Phase 1): intercept + term + term^2 + sector dummies.
- 'full'   (Phase 2a): naive + liquidity controls, so fair value prices the
  illiquidity premium and the residual is liquidity-neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.liquidity.controls import LIQ_FEATURES


def build_design_matrix(
    df: pd.DataFrame,
    feature_set: str = "naive",
    term_col: str = "ttm",
    sector_col: str = "sector_ff30",
) -> pd.DataFrame:
    """Return numeric design matrix X for one cross-section.

    Term structure (term, term^2), one-hot sector (drop-first), and — for 'full' —
    the liquidity controls in `LIQ_FEATURES`.
    """
    term = pd.to_numeric(df[term_col], errors="coerce")
    X = pd.DataFrame({"const": 1.0, "term": term, "term2": term**2}, index=df.index)
    sectors = pd.get_dummies(df[sector_col].astype("category"), prefix="sec", drop_first=True,
                             dtype=float)
    X = pd.concat([X, sectors], axis=1)

    if feature_set == "full":
        liq = [c for c in LIQ_FEATURES if c in df.columns]
        X = pd.concat([X, df[liq].astype(float)], axis=1)

    return X.replace([np.inf, -np.inf], np.nan)
