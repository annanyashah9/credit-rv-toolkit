.PHONY: install dev phase0 fred phase1 lint test fixture clean

install:
	.venv/bin/pip install -e .

dev:
	.venv/bin/pip install -e ".[dev]"

# Phase 0 data-truth inventory -> docs/phase0-findings.md
phase0:
	.venv/bin/crv phase0 --config configs/base.yaml

# Cache the Treasury curve from FRED (keyless)
fred:
	.venv/bin/crv fred --config configs/base.yaml

# Phase 1: universe -> spreads -> naive residual signal
phase1:
	.venv/bin/crv phase1 --config configs/base.yaml

# Phase 2a: universe -> spreads -> liquidity -> peer-shrunk signal
phase2a:
	.venv/bin/crv phase2a --config configs/base.yaml

# Phase 2b: ML-vs-linear walk-forward comparison -> docs/phase2b-results.md
phase2b:
	.venv/bin/crv phase2b --config configs/base.yaml

# Phase 1.5: thin IC/HAC backtest gate -> reports/ + docs/phase1_5-results.md
backtest:
	.venv/bin/crv backtest --config configs/base.yaml

# Generate a tiny synthetic panel so the pipeline is runnable without real data.
fixture:
	.venv/bin/python scripts/make_fixture.py

lint:
	.venv/bin/ruff check src tests

test:
	.venv/bin/pytest -q

clean:
	rm -rf data/interim/* data/processed/* **/__pycache__
