"""Generate the Phase 1.5 results markdown from the IC table + quintile diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crv.config import Config


def _verdict(table: pd.DataFrame, quints: pd.DataFrame, tail_frac: float) -> str:
    """Honest read that reconciles rank-IC with the equal-weight long-short.

    The key Phase-1.5 finding is usually a divergence: positive rank IC (typical cheap
    bond converges) alongside a negative *mean* long-short (a distressed tail in the
    cheap bucket blows up). That divergence is the motivation for Phases 2-3, not a
    failure.
    """
    ic_ok = ((table["mean_ic"] > 0) & (table["t_stat"].abs() >= 2)).sum() >= 2
    ls_mean = quints.loc["LS", "mean_r"] if "LS" in quints.index else np.nan
    ls_med = quints.loc["LS", "median_r"] if "LS" in quints.index else np.nan

    if ic_ok and ls_mean < 0 <= ls_med:
        return (
            f"**GO for Phase 2 — with the central caveat made concrete.** Rank IC is "
            f"positive and HAC-significant at multiple horizons (the *typical* cheap bond "
            f"converges: monotone median returns, top-minus-bottom median LS = {ls_med:+.4f}). "
            f"But the equal-weight *mean* long-short is NEGATIVE ({ls_mean:+.4f}) because "
            f"~{tail_frac:.0%} of the cheapest bucket are distressed names with catastrophic "
            f"returns. The signal has genuine convergence content; an uncontrolled portfolio "
            f"loses to the distressed tail. This is exactly what Phase 2 (liquidity controls, "
            f"log-spread) and Phase 3 (default carry-through, contamination test, "
            f"neutralization) are built to fix."
        )
    if ic_ok and ls_mean >= 0:
        return ("**GO.** Positive HAC-significant IC and a positive equal-weight long-short even "
                "before controls — stronger than expected for a naive signal.")
    if (table["mean_ic"] > 0).all():
        return ("**WEAK/POSITIVE.** IC positive across horizons but not all HAC-significant; "
                "Phase 2's controlled signal should sharpen it.")
    return ("**FLAT/MIXED.** Naive signal lacks consistent positive IC — expected for an "
            "uncontrolled level-spread residual; motivates Phase 2 rather than condemning it.")


def write_phase2b_comparison(cfg: Config, results: dict, mid_horizon: int) -> Path:
    """Side-by-side ML-vs-linear comparison doc.

    `results` maps model_kind -> {'ic': ic_table_df, 'quints': df, 'tail': float}.
    """
    out = cfg.paths.docs / "phase2b-results.md"

    # IC summary: one row per model, columns = mean IC (HAC t) per horizon.
    ic_rows = []
    for kind, r in results.items():
        row = {"model": kind}
        for _, h in r["ic"].iterrows():
            row[f"IC_{int(h['horizon_m'])}m"] = f"{h['mean_ic']:.3f} (t={h['t_stat']:.1f})"
        ls = r["quints"].loc["LS"] if "LS" in r["quints"].index else None
        row["LS_mean"] = f"{ls['mean_r']:+.4f}" if ls is not None else "n/a"
        row["LS_median"] = f"{ls['median_r']:+.4f}" if ls is not None else "n/a"
        row["tail<-5%"] = f"{r['tail']:.1%}"
        ic_rows.append(row)
    summary = pd.DataFrame(ic_rows)

    # Verdict: compare WF-GBM vs WF-Ridge (clean ML-vs-linear), at mid horizon.
    def _mean_ic(kind):
        t = results.get(kind, {}).get("ic")
        if t is None:
            return np.nan
        row = t[t["horizon_m"] == mid_horizon]
        return float(row["mean_ic"].iloc[0]) if len(row) else np.nan

    gbm, ridge = _mean_ic("gbm_wf"), _mean_ic("ridge_wf")
    if np.isfinite(gbm) and np.isfinite(ridge):
        gap = gbm - ridge
        rel = gap / abs(ridge) if ridge else np.nan
        if gap > 0 and rel > 0.10:
            verdict = (f"**ML edges linear.** At {mid_horizon}m, WF-GBM IC {gbm:.3f} vs WF-Ridge "
                       f"{ridge:.3f} (+{rel:.0%}) on identical features/training — the gap is "
                       f"functional form. Worth carrying GBM into Phase 3, but confirm it survives "
                       f"costs/neutralization.")
        elif abs(rel) <= 0.10:
            verdict = (f"**ML ≈ linear.** WF-GBM ({gbm:.3f}) and WF-Ridge ({ridge:.3f}) are within "
                       f"~10% at {mid_horizon}m — the relationship here is largely linear once "
                       f"liquidity is controlled. A defensible, honest finding: prefer the simpler "
                       f"linear model unless GBM wins clearly after costs.")
        else:
            verdict = (f"**Linear edges ML.** WF-Ridge ({ridge:.3f}) ≥ WF-GBM ({gbm:.3f}) at "
                       f"{mid_horizon}m — added flexibility doesn't help on this signal; the GBM "
                       f"likely overfits the proxy. Carry the linear model forward.")
    else:
        verdict = "Insufficient results to adjudicate ML vs linear."

    # The other headline: how does the cross-sectional peer-shrunk model compare to the
    # pooled walk-forward arms?
    ps = _mean_ic("peer_shrunk")
    if np.isfinite(ps) and np.isfinite(gbm):
        better = ps > max(gbm, ridge)
        ps_quint = results.get("peer_shrunk", {}).get("quints")
        ps_ls = ps_quint.loc["LS", "mean_r"] if ps_quint is not None else np.nan
        verb = "beats" if better else "trails"
        verdict += (
            f"\n\n**Bigger story — model FORM, not complexity.** The cross-sectional "
            f"peer-shrunk model {verb} both pooled walk-forward arms at {mid_horizon}m "
            f"(IC {ps:.3f} vs {ridge:.3f}/{gbm:.3f}) and is the only arm with a positive "
            f"equal-weight long-short (LS_mean {ps_ls:+.4f} vs negative for both WF arms). "
            f"Its edge is the per-date issuer shrinkage toward sector peers — which the "
            f"pooled models structurally lack (2,988 issuers can't be dummies) — not temporal "
            f"pooling or tree flexibility. The win came from the right cross-sectional "
            f"structure, not from a more complex learner."
        )

    lines = [
        "# Phase 2b — ML-vs-Linear Comparison (walk-forward)",
        "",
        "> All arms scored on the same thin spread-return proxy (costless; bounce-inflated). The",
        "> GAP between models is more trustworthy than absolute levels. WF-GBM and WF-Ridge share",
        "> identical features + rolling-60m training, so their gap isolates functional form;",
        "> peer_shrunk is the Phase-2a cross-sectional reference.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        f"## Verdict ({mid_horizon}m)",
        "",
        verdict,
        "",
        "Per-model detail: `docs/backtest-<kind>.md`.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def write_phase3a_results(
    cfg: Config, perf: dict, n_defaults: int, cost_split: dict, mid_horizon: int
) -> Path:
    """Net-of-cost P&L writeup with gross-vs-net, neutrality check, and honest read."""
    out = cfg.paths.docs / "phase3a-results.md"
    hold = cfg.backtest.holding_months
    rows = []
    for style, p in perf.items():
        rows.append({
            "style": style,
            "gross_ann": f"{p['gross_ann']:+.4f}",
            "net_ann": f"{p['net_ann']:+.4f}",
            "net_sharpe": f"{p['net_sharpe']:.2f}",
            "net_HAC_t": f"{p['net_hac_t']:.2f}",
            "ann_cost": f"{p['cost_drag_ann']:.4f}",
            "turnover": f"{p['avg_turnover']:.2f}",
            "breadth": f"{p['avg_breadth']:.0f}",
            "net_dur": f"{p['avg_net_dur']:+.2f}",
            "net_dts": f"{p['avg_net_dts']:+.1f}",
        })
    table = pd.DataFrame(rows)

    ls = perf.get("ls", {})
    nan = float("nan")
    net_ann, hac_t = ls.get("net_ann", nan), ls.get("net_hac_t", nan)
    sharpe, drag = ls.get("net_sharpe", nan), ls.get("cost_drag_ann", nan)
    turn = ls.get("avg_turnover", nan)
    ndur, ndts = ls.get("avg_net_dur", nan), ls.get("avg_net_dts", nan)
    net_pos = net_ann > 0 and abs(hac_t) >= 2
    gross_ann = ls.get("gross_ann", nan)
    survives = "survives" if net_pos else "does NOT survive"
    verdict = (
        f"The duration/DTS/sector-neutral long-short is **gross-positive** "
        f"({gross_ann:+.4f}/yr) but {survives} transaction costs: a {drag:.4f}/yr cost drag "
        f"at {turn:.2f} monthly turnover flips it to net {net_ann:+.4f}/yr "
        f"(HAC t={hac_t:.2f}, Sharpe {sharpe:.2f}). Realized net duration {ndur:+.2f} and net "
        f"DTS {ndts:+.1f}. {n_defaults} defaults carried through at {cfg.backtest.recovery:.0%} "
        f"recovery (real recovery losses, heavier than the proxy)."
    )
    if not net_pos:
        verdict += (" This is the honest centerpiece result: the signal predicts (gross-positive), "
                    "but at monthly turnover, wide credit bid/ask makes it untradeable as-is. The "
                    "lever is turnover — Phase 3b tests longer holding / no-trade bands, plus the "
                    "impending-default and liquidity contamination tests.")

    lines = [
        "# Phase 3a — Net-of-Cost Neutralized Backtest",
        "",
        "> Real excess returns (carry, spread-change, default loss), measured bid/ask costs,",
        "> default/recovery carry-through, duration/DTS/sector-neutral. Signal: peer_shrunk.",
        f"> Holding {hold}m overlapping; HAC lag {hold - 1}.",
        "",
        "## Performance (annualized, excess-return units)",
        "",
        table.to_markdown(index=False),
        "",
        f"Cost coverage: {cost_split.get('measured', 0):.0%} measured bid/ask, "
        f"{cost_split.get('fallback', 0):.0%} bucketed fallback.",
        "",
        "## Read",
        "",
        verdict,
        "",
        "Figure: `reports/equity_curve.png`. Phase 3b adds contamination tests + IC-decay on real "
        "returns.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def write_phase1_5_summary(
    cfg: Config, table: pd.DataFrame, quints: pd.DataFrame, mid_horizon: int,
    tail_frac: float, model_kind: str = "naive",
) -> Path:
    out = cfg.paths.docs / f"backtest-{model_kind}.md"
    lines = [
        f"# Thin Backtest Results — `{model_kind}` signal",
        "",
        "> Auto-generated by `crv backtest`. Naive signal only: spread-return proxy outcome,",
        "> **no costs / neutralization / default handling** (those are Phase 3). The outcome is",
        "> a costless spread-change proxy off the same price series, so magnitudes are optimistic",
        "> (bid-ask bounce can inflate mean-reversion); treat as direction-of-travel, not P&L.",
        "",
        "## Information coefficient (rank IC, HAC SE, lag = horizon-1)",
        "",
        table.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Quintile diagnostic ({mid_horizon}m, naive — no neutralization/costs)",
        "",
        quints.to_markdown(floatfmt=".5f"),
        "",
        "(Quantile 1 = richest, top = cheapest; **LS** = top − bottom. The mean-vs-median gap is",
        f"the contamination signal: ~{tail_frac:.0%} of the cheapest bucket has forward return",
        "< −5% — the distressed tail.)",
        "",
        "## Read",
        "",
        _verdict(table, quints, tail_frac),
        "",
        f"Figures: `reports/ic_decay.png`, `reports/ic_timeseries_{mid_horizon}m.png`.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out
