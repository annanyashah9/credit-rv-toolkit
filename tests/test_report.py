"""Smoke test for the consolidated report and the run manifest."""

from __future__ import annotations

import json

from crv.io import file_hash, write_manifest


def test_manifest_roundtrip(tmp_path):
    f = tmp_path / "data.txt"
    f.write_bytes(b"hello")
    out = write_manifest(tmp_path / "m.json", {"seed": 42, "hash": file_hash(f)})
    payload = json.loads(out.read_text())
    assert payload["seed"] == 42
    assert len(payload["hash"]) == 64  # sha256 hex


def test_report_renders_sections_from_artifacts():
    """If the interim artifacts exist, the report builds and contains the key sections.
    Skipped on a clean checkout where the pipeline hasn't been run."""
    import pytest

    from crv.config import load_config

    cfg = load_config("configs/base.yaml")
    if not (cfg.paths.interim / "signal-peer_shrunk.parquet").exists():
        pytest.skip("interim artifacts not present; run `crv phase2a` first")
    from crv.report.consolidated import build_report

    md = build_report(cfg)
    for header in ("## Net-of-cost performance", "## Predictive content on realised returns",
                   "## Is the edge genuine", "## Conclusion"):
        assert header in md
