"""Phase 0 data-truth inventory.

Runs the *automatable* checks from docs/phase0-gate.md against the raw OBAP panel
and writes docs/phase0-findings.md with, for each check, the auto-detected value and
a suggested PASS / BRANCH. Human-judgment checks (price convention, distress-floor
calibration) are emitted with the supporting statistics but left for a person to
resolve.

This is the first gate deliverable. It does NOT modify any config; it informs the
config freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crv.config import Config
from crv.ingest.obap import classify_columns, load_panel


@dataclass
class CheckResult:
    check_id: str
    title: str
    finding: str
    verdict: str  # "PASS" | "BRANCH" | "MANUAL"
    detail: str = ""


@dataclass
class InventoryReport:
    n_rows: int
    n_cols: int
    roles: dict[str, str]
    checks: list[CheckResult] = field(default_factory=list)


def _pick(roles: dict[str, str], role: str) -> str | None:
    """First column assigned to `role`, if any."""
    for col, r in roles.items():
        if r == role:
            return col
    return None


def _infer_frequency(df: pd.DataFrame, date_col: str, id_col: str | None) -> tuple[str, float]:
    """Return (label, median_gap_days). Gaps measured within-id when an id exists."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    work = df.assign(_d=dates).dropna(subset=["_d"])
    if id_col and id_col in work.columns:
        gaps = work.sort_values([id_col, "_d"]).groupby(id_col)["_d"].diff().dropna()
    else:
        uniq = pd.Series(sorted(work["_d"].unique()))
        gaps = uniq.diff().dropna()
    if len(gaps) == 0:
        return "unknown", float("nan")
    median_days = float(pd.to_timedelta(gaps).dt.days.median())
    if median_days <= 4:
        label = "daily"
    elif 20 <= median_days <= 40:
        label = "monthly"
    elif 5 <= median_days <= 10:
        label = "weekly"
    else:
        label = f"~{median_days:.0f}d (irregular)"
    return label, median_days


def build_inventory(cfg: Config) -> InventoryReport:
    # Apply the column map if the user has set one: on the first (discovery) run the
    # map is empty so we classify the raw schema; on later (verification) runs the
    # canonical names classify cleanly and stats use the intended columns.
    df = load_panel(cfg)
    roles = classify_columns(df)
    rep = InventoryReport(n_rows=len(df), n_cols=df.shape[1], roles=roles)

    date_col = _pick(roles, "date")
    id_col = _pick(roles, "identifier")

    # --- A1 frequency ---------------------------------------------------------
    if date_col:
        label, gap = _infer_frequency(df, date_col, id_col)
        verdict = "PASS" if label == "daily" else "BRANCH"
        rep.checks.append(
            CheckResult(
                "A1",
                "Panel frequency",
                f"date col = {date_col!r}; median gap = {gap:.1f}d -> **{label}**",
                verdict,
                "" if verdict == "PASS"
                else "Monthly/irregular is PLAN-ALTERING: revisit horizons, Amihud, HAC lag.",
            )
        )
    else:
        rep.checks.append(
            CheckResult("A1", "Panel frequency", "no date column detected", "MANUAL",
                        "Could not auto-detect a date column; set it manually.")
        )

    # --- A2 column inventory --------------------------------------------------
    role_counts: dict[str, int] = {}
    for r in roles.values():
        role_counts[r] = role_counts.get(r, 0) + 1
    core = {"price", "yield", "spread", "duration"}
    present_core = {r for r in core if r in roles.values()}
    rep.checks.append(
        CheckResult(
            "A2",
            "Column inventory",
            f"{df.shape[1]} cols; roles found: {sorted(set(roles.values()))}",
            "PASS" if present_core == core else "BRANCH",
            f"core present: {sorted(present_core)}; missing: {sorted(core - present_core)}",
        )
    )

    # --- A3 identifiers -------------------------------------------------------
    rep.checks.append(
        CheckResult(
            "A3",
            "Identifiers & join keys",
            f"id col = {id_col!r}; issuer col = {_pick(roles, 'issuer')!r}",
            "PASS" if id_col else "BRANCH",
            "Need a stable per-bond key to join the ratings/calls CSV and FRED dates.",
        )
    )

    # --- A4 history length ----------------------------------------------------
    if date_col:
        d = pd.to_datetime(df[date_col], errors="coerce").dropna()
        span_years = (d.max() - d.min()).days / 365.25 if len(d) else 0.0
        rep.checks.append(
            CheckResult(
                "A4",
                "History length & coverage",
                f"{d.min().date()} -> {d.max().date()} ({span_years:.1f} yrs)",
                "PASS" if span_years >= 5 else "BRANCH",
                "Short history (<5y): expanding-window only; fewer horizons; flag low power.",
            )
        )

    # --- A6 missingness -------------------------------------------------------
    null_frac = df.isna().mean()
    worst = null_frac.sort_values(ascending=False).head(5)
    rep.checks.append(
        CheckResult(
            "A6",
            "Missingness",
            "top null fractions: "
            + ", ".join(f"{c}={v:.0%}" for c, v in worst.items()),
            "PASS" if null_frac.max() < 0.5 else "BRANCH",
            "Heavy gaps -> minimum-observations filter in the universe rule.",
        )
    )

    # --- B1 volume present ----------------------------------------------------
    vol_col = _pick(roles, "volume")
    if vol_col:
        cov = 1 - df[vol_col].isna().mean()
        rep.checks.append(
            CheckResult("B1", "Volume present (Amihud)",
                        f"volume col = {vol_col!r}, coverage {cov:.0%}",
                        "PASS" if cov > 0.5 else "BRANCH",
                        "Enables Amihud alongside Bao gamma."))
    else:
        rep.checks.append(
            CheckResult("B1", "Volume present (Amihud)", "no volume column detected", "BRANCH",
                        "Expected fallback: Bao gamma + trade-freq + age + issue-size."))

    # --- B2 bid/ask -----------------------------------------------------------
    ba_col = _pick(roles, "bid_ask")
    rep.checks.append(
        CheckResult("B2", "Bid-ask / high-low", f"bid_ask col = {ba_col!r}",
                    "PASS" if ba_col else "BRANCH",
                    "If absent (expected): cost model = bucketed half-spread schedule."))

    # --- B3 cashflow fields (Z-spread, optional) ------------------------------
    has_cpn = _pick(roles, "coupon") is not None
    has_mat = _pick(roles, "maturity") is not None
    rep.checks.append(
        CheckResult("B3", "Cashflow fields (Z-spread, optional)",
                    f"coupon={has_cpn}, maturity={has_mat}",
                    "PASS" if (has_cpn and has_mat) else "BRANCH",
                    "Missing is fine in refined scope: G-spread is the substrate."))

    # --- A5 / D3 manual flags -------------------------------------------------
    price_col = _pick(roles, "price")
    if price_col:
        s = pd.to_numeric(df[price_col], errors="coerce").dropna()
        rep.checks.append(
            CheckResult("A5", "Price convention (manual)",
                        f"{price_col!r}: min={s.min():.2f}, median={s.median():.2f}, "
                        f"max={s.max():.2f}",
                        "MANUAL",
                        "Confirm clean vs dirty & par scaling (100 vs 1.0). Sets distress floor."))

    return rep


def render_findings_md(cfg: Config, rep: InventoryReport) -> str:
    """Render the inventory as the docs/phase0-findings.md record."""
    lines: list[str] = []
    lines.append("# Phase 0 — Data-Truth Findings (auto-generated)")
    lines.append("")
    lines.append("> Generated by `crv.ingest.inventory`. Cross-check against "
                 "`docs/phase0-gate.md`. Resolve every MANUAL item by hand, then freeze configs.")
    lines.append("")
    lines.append(f"- Panel rows: **{rep.n_rows:,}**, columns: **{rep.n_cols}**")
    lines.append("")
    lines.append("## Automated check results")
    lines.append("")
    lines.append("| Check | Title | Finding | Verdict |")
    lines.append("|---|---|---|---|")
    for c in rep.checks:
        finding = c.finding.replace("|", "\\|")
        lines.append(f"| {c.check_id} | {c.title} | {finding} | **{c.verdict}** |")
    lines.append("")
    lines.append("## Notes / branch actions (items needing a decision)")
    lines.append("")
    for c in rep.checks:
        if c.detail and c.verdict in {"BRANCH", "MANUAL"}:
            lines.append(f"- **{c.check_id} ({c.verdict})** — {c.detail}")
    lines.append("")
    lines.append("## Column role guesses (confirm, then fill `ingest.obap_column_map`)")
    lines.append("")
    lines.append("| Column | Guessed role |")
    lines.append("|---|---|")
    for col, role in rep.roles.items():
        lines.append(f"| {col} | {role} |")
    lines.append("")
    return "\n".join(lines)


def run(cfg: Config) -> str:
    """Build the inventory and write docs/phase0-findings.md. Returns the path."""
    rep = build_inventory(cfg)
    out = cfg.paths.docs / "phase0-findings.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_findings_md(cfg, rep))
    return str(out)
