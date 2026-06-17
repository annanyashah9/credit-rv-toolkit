"""Golden-path tests for Phase 0 inventory: column classification + frequency."""

from __future__ import annotations

import pandas as pd

from crv.ingest.obap import classify_columns


def test_classify_roles_specific_beats_generic():
    df = pd.DataFrame(
        columns=[
            "CUSIP", "issuer_ticker", "trd_exctn_dt", "prc_clean", "yield",
            "gspread_bp", "mod_dur", "par_volume", "rating", "sector",
            "coupon", "maturity_dt", "issue_size_mm",
        ]
    )
    roles = classify_columns(df)
    # The greedy "dt" date hint must NOT steal maturity_dt.
    assert roles["maturity_dt"] == "maturity"
    assert roles["trd_exctn_dt"] == "date"
    assert roles["CUSIP"] == "identifier"
    assert roles["prc_clean"] == "price"
    assert roles["gspread_bp"] == "spread"
    assert roles["mod_dur"] == "duration"
    assert roles["par_volume"] == "volume"
    assert roles["sector"] == "sector"
    assert roles["issue_size_mm"] == "size"


def test_unknown_column_is_unknown():
    df = pd.DataFrame(columns=["some_random_field"])
    assert classify_columns(df)["some_random_field"] == "unknown"
