"""Design-matrix construction for the fair-value model.

Phase-1 (naive) features are term structure + sector only: a bond is "fair" relative
to peers of similar maturity and sector. Liquidity controls, rating, and seniority
enter in Phase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_design_matrix(df: pd.DataFrame, term_col: str = "ttm",
                        sector_col: str = "sector_ff30") -> pd.DataFrame:
    """Return numeric design matrix X for one cross-section.

    Columns: intercept, term, term^2, and one-hot sector dummies (drop-first to avoid
    collinearity with the intercept).
    """
    term = pd.to_numeric(df[term_col], errors="coerce")
    X = pd.DataFrame(
        {"const": 1.0, "term": term, "term2": term**2},
        index=df.index,
    )
    sectors = pd.get_dummies(df[sector_col].astype("category"), prefix="sec", drop_first=True,
                             dtype=float)
    X = pd.concat([X, sectors], axis=1)
    return X.replace([np.inf, -np.inf], np.nan)
