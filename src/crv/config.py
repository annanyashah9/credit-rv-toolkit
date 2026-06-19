"""Typed, config-driven settings loaded from YAML.

Every stage of the pipeline reads its parameters from here so that a run is fully
reproducible from one config file. Phase 0's findings (docs/phase0-findings.md) are
what *freeze* the values in these configs; until then, fields carry conservative
defaults and many are intentionally optional.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Filesystem layout. Relative paths are resolved against the project root."""

    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    reference: Path = Path("data/reference")
    docs: Path = Path("docs")
    reports: Path = Path("reports")

    # The raw OBAP panel file the user drops in. Format inferred from suffix
    # (.csv / .parquet). Left as None until the user points us at it.
    obap_panel: Path | None = None


class IngestConfig(BaseModel):
    """Ingestion knobs. `obap_column_map` lets us rename the panel's real columns
    onto our canonical names once Phase 0 inventory reveals them."""

    obap_column_map: dict[str, str] = Field(default_factory=dict)
    fred_series: dict[str, str] = Field(
        default_factory=lambda: {
            # canonical_name -> FRED series id; filled/confirmed in Phase 0.
            "sofr": "SOFR",
        }
    )


class UniverseConfig(BaseModel):
    """Point-in-time inclusion thresholds (Moderate strictness, frozen by Phase 0)."""

    size_floor: float = 500_000.0       # amt outstanding in $thousands => $500mm
    min_trade_freq: float = 0.50        # fraction of trailing business days traded
    window_days: int = 63               # ~3 trading months
    min_ttm: float = 1.0                # years
    max_ttm: float = 30.0
    rebalance_freq: str = "ME"          # pandas offset alias; ME = month-end
    max_rating_num: float | None = None  # distressed cutoff on AAA=1.. scale; None=off (no CSV yet)


class SignalConfig(BaseModel):
    """Naive Phase-1 fair-value / residual settings."""

    min_names_per_date: int = 20        # skip cross-sections too thin to fit
    robust_scale: bool = True           # MAD-based standardization


class LiquidityConfig(BaseModel):
    """Phase-2a liquidity-control settings."""

    window_days: int = 252              # trailing daily window for Bao gamma / Amihud
    min_obs: int = 40                   # min daily obs in window to trust a measure
    winsor_pct: float = 0.01            # two-sided winsorization of liquidity features


class ModelConfig(BaseModel):
    """Fair-value model selection (Phase 2a/2b)."""

    kind: str = "peer_shrunk"           # 'naive'|'peer_shrunk'|'ridge_wf'|'gbm_wf'
    asinh_scale: float = 100.0          # bp scale inside asinh transform
    shrink_k: float | None = None       # EB issuer shrinkage; None => estimate per cross-section

    # Walk-forward training (ridge_wf / gbm_wf)
    train_scheme: str = "rolling"       # 'rolling' | 'expanding'
    train_window_months: int = 60
    min_train_months: int = 24
    refit_every_months: int = 3
    ridge_alpha: float = 1.0
    gbm: dict = Field(default_factory=lambda: {
        "max_depth": 3, "learning_rate": 0.05, "max_iter": 300,
        "min_samples_leaf": 200, "l2_regularization": 1.0,
    })


class BacktestConfig(BaseModel):
    """Backtest settings (Phase 1.5 thin gate + Phase 3 P&L)."""

    horizons: list[int] = Field(default_factory=lambda: [1, 3, 6])  # months ahead
    n_quantiles: int = 5
    ic_method: str = "spearman"          # rank IC; robust to z's right-skew
    winsor_z: float = 5.0                # cap |z| for the robustness line

    # Phase 3 P&L
    holding_months: int = 3              # overlapping-portfolio hold length
    recovery: float = 0.40               # senior-unsecured default recovery (par fraction)
    distress_floor: float = 55.0         # clean price below which an early exit = default
    neutralize: list[str] = Field(default_factory=lambda: ["sector", "duration", "dts"])
    ann_factor: int = 12                 # monthly -> annual

    # Phase 3b
    holding_grid: list[int] = Field(default_factory=lambda: [3, 6, 12])
    band_enter_q: int = 4                # enter long at top quintile (0-indexed: 4 of 5)
    band_exit_q: int = 3                 # exit only when it drops below quintile 3 (hysteresis)
    default_window_m: int = 6            # horizon for "impending default" tagging
    liquidity_split: str = "bao_gamma"   # liquidity feature for the premium test


class Config(BaseModel):
    """Root config object passed through the whole pipeline."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # Resolved at load time; not read from YAML.
    project_root: Path = Field(default=Path.cwd(), exclude=True)

    def resolve_paths(self) -> Config:
        """Make all paths absolute relative to project_root (idempotent)."""
        root = self.project_root
        p = self.paths
        for field in ("raw", "interim", "processed", "reference", "docs", "reports"):
            val = getattr(p, field)
            if not val.is_absolute():
                setattr(p, field, root / val)
        if p.obap_panel is not None and not p.obap_panel.is_absolute():
            p.obap_panel = root / p.obap_panel
        return self


def load_config(path: str | Path, project_root: str | Path | None = None) -> Config:
    """Load a YAML config file into a validated Config.

    project_root defaults to the config file's parent's parent (configs/ lives at
    the repo root), so paths resolve correctly regardless of CWD.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    root = Path(project_root) if project_root else path.resolve().parent.parent
    cfg = Config(**data)
    cfg.project_root = root
    return cfg.resolve_paths()
