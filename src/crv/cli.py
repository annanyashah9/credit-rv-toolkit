"""Command-line entry point: `crv <stage> --config <yaml>`.

Stages are added as phases land. Today: `phase0` (data-truth inventory).
"""

from __future__ import annotations

import argparse
import sys

from crv.config import load_config


def _cmd_phase0(args: argparse.Namespace) -> int:
    from crv.ingest import inventory

    cfg = load_config(args.config)
    out = inventory.run(cfg)
    print(f"Phase 0 findings written to: {out}")
    return 0


def _cmd_fred(args: argparse.Namespace) -> int:
    from crv.ingest.fred import load_treasury_curve

    cfg = load_config(args.config)
    curve = load_treasury_curve(cfg, refresh=args.refresh)
    print(f"Treasury curve cached: {curve.shape[0]} days x {curve.shape[1]} tenors, "
          f"{curve.index.min().date()} -> {curve.index.max().date()}")
    return 0


def _cmd_universe(args: argparse.Namespace) -> int:
    from crv.ingest.obap import load_panel
    from crv.ingest.reference import load_reference
    from crv.io import write_stage
    from crv.universe import apply_curated_filters, build_candidate_universe

    cfg = load_config(args.config)
    panel = load_panel(cfg)
    cand = build_candidate_universe(panel, cfg)
    ref = load_reference(cfg.paths.reference / "ratings_calls.csv")
    final = apply_curated_filters(cand, ref, max_rating_num=cfg.universe.max_rating_num)
    out = write_stage(final, cfg.paths.interim / "universe.parquet")
    n_dates = final["rebalance_date"].nunique() if not final.empty else 0
    print(f"Universe: {len(final):,} (cusip x date) rows over {n_dates} rebalances "
          f"({'curated' if not ref.empty else 'candidate-pool, no CSV yet'}) -> {out}")
    return 0


def _cmd_spreads(args: argparse.Namespace) -> int:
    from crv.ingest.fred import load_treasury_curve
    from crv.io import read_table, write_stage
    from crv.signal import compute_spreads

    cfg = load_config(args.config)
    universe = read_table(cfg.paths.interim / "universe.parquet")
    curve = load_treasury_curve(cfg)
    spreads = compute_spreads(universe, curve)
    out = write_stage(spreads, cfg.paths.interim / "spreads.parquet")
    cov = spreads["gspread_bp"].notna().mean()
    print(f"Spreads: gspread+dts on {len(spreads):,} rows, gspread coverage {cov:.0%} -> {out}")
    return 0


def _cmd_liquidity(args: argparse.Namespace) -> int:
    from crv.ingest.obap import load_panel
    from crv.io import read_table, write_stage
    from crv.liquidity.controls import build_liquidity_features

    cfg = load_config(args.config)
    panel = load_panel(cfg)
    universe = read_table(cfg.paths.interim / "universe.parquet")
    feats = build_liquidity_features(panel, universe, cfg)
    out = write_stage(feats, cfg.paths.interim / "liquidity.parquet")
    cov = feats[["bao_gamma", "amihud"]].notna().all(axis=1).mean()
    print(f"Liquidity: {len(feats):,} rows, bao+amihud coverage {cov:.0%} -> {out}")
    return 0


def _cmd_signal(args: argparse.Namespace) -> int:
    from crv.io import read_table, write_stage
    from crv.signal import make_signal

    cfg = load_config(args.config)
    spreads = read_table(cfg.paths.interim / "spreads.parquet")
    liquidity = None
    if cfg.model.kind == "peer_shrunk":
        liquidity = read_table(cfg.paths.interim / "liquidity.parquet")
    sig = make_signal(spreads, liquidity, cfg)
    write_stage(sig, cfg.paths.interim / f"signal-{cfg.model.kind}.parquet")
    out = write_stage(sig, cfg.paths.interim / "signal.parquet")  # active
    n_dates = sig["rebalance_date"].nunique() if not sig.empty else 0
    print(f"Signal ({cfg.model.kind}): {len(sig):,} residuals over {n_dates} "
          f"cross-sections -> {out}")
    return 0


def _cmd_phase1(args: argparse.Namespace) -> int:
    """Run the Phase-1 chain: universe -> spreads -> signal (use a naive-model config)."""
    for fn in (_cmd_universe, _cmd_spreads, _cmd_signal):
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


def _cmd_phase2a(args: argparse.Namespace) -> int:
    """Run the Phase-2a chain: universe -> spreads -> liquidity -> signal."""
    for fn in (_cmd_universe, _cmd_spreads, _cmd_liquidity, _cmd_signal):
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


def _cmd_phase2b(args: argparse.Namespace) -> int:
    """Generate signals for peer_shrunk / ridge_wf / gbm_wf, backtest each, and write
    the ML-vs-linear comparison."""
    from crv.backtest.ic import ic_table
    from crv.backtest.quintiles import quintile_returns, tail_loss_fraction
    from crv.io import read_table, write_stage
    from crv.report.summary import write_phase1_5_summary, write_phase2b_comparison
    from crv.signal import make_signal

    cfg = load_config(args.config)
    # Reuse universe/spreads/liquidity if present, else build them.
    for name, fn in (("universe", _cmd_universe), ("spreads", _cmd_spreads),
                     ("liquidity", _cmd_liquidity)):
        if not (cfg.paths.interim / f"{name}.parquet").exists():
            fn(args)

    spreads = read_table(cfg.paths.interim / "spreads.parquet")
    liquidity = read_table(cfg.paths.interim / "liquidity.parquet")
    bt = cfg.backtest
    mid = bt.horizons[len(bt.horizons) // 2]

    results = {}
    for kind in ("peer_shrunk", "ridge_wf", "gbm_wf"):
        cfg.model.kind = kind
        sig = make_signal(spreads, liquidity, cfg)
        write_stage(sig, cfg.paths.interim / f"signal-{kind}.parquet")
        table = ic_table(sig, horizons=tuple(bt.horizons), method=bt.ic_method,
                         winsor_z=bt.winsor_z)
        quints = quintile_returns(sig, mid, n_quantiles=bt.n_quantiles)
        tail = tail_loss_fraction(sig, mid, n_quantiles=bt.n_quantiles)
        write_phase1_5_summary(cfg, table, quints, mid, tail, model_kind=kind)
        results[kind] = {"ic": table, "quints": quints, "tail": tail}
        ic_mid = table.loc[table["horizon_m"] == mid, "mean_ic"].iloc[0]
        print(f"  {kind:12s} IC{mid}m={ic_mid:.3f}  LS_mean={quints.loc['LS','mean_r']:+.4f}  "
              f"tail={tail:.1%}")

    out = write_phase2b_comparison(cfg, results, mid)
    print(f"\nComparison: {out}")
    return 0


def _cmd_phase3a(args: argparse.Namespace) -> int:
    """Net-of-cost neutralized backtest on the peer_shrunk signal."""
    import pandas as pd

    from crv.backtest.costs import build_cost_table
    from crv.backtest.defaults import DEFAULTED, detect_exits
    from crv.backtest.performance import summarize
    from crv.backtest.portfolio import run_portfolio
    from crv.backtest.returns import monthly_excess_return
    from crv.ingest.obap import load_panel
    from crv.io import read_table
    from crv.report.figures import plot_equity_curves
    from crv.report.summary import write_phase3a_results

    cfg = load_config(args.config)
    inter = cfg.paths.interim
    for name, fn in (("universe", _cmd_universe), ("spreads", _cmd_spreads),
                     ("liquidity", _cmd_liquidity)):
        if not (inter / f"{name}.parquet").exists():
            fn(args)
    # Ensure the peer_shrunk signal exists.
    if not (inter / "signal-peer_shrunk.parquet").exists():
        cfg.model.kind = "peer_shrunk"
        _cmd_signal(args)

    panel = load_panel(cfg)
    spreads = read_table(inter / "spreads.parquet")
    signal = read_table(inter / "signal-peer_shrunk.parquet")

    exits = detect_exits(panel, cfg)
    n_def = int((exits["exit_type"] == DEFAULTED).sum())
    costs = build_cost_table(panel, spreads, cfg)
    cost_split = costs["cost_source"].value_counts(normalize=True).to_dict()
    r1 = monthly_excess_return(spreads, exits, cfg)

    # Working frame: signal z + neutralization attributes from spreads.
    df = signal.merge(spreads[["cusip", "rebalance_date", "mod_duration"]],
                      on=["cusip", "rebalance_date"], how="left")

    books, perf = {}, {}
    for style in ("ls", "long_only"):
        book = run_portfolio(df, r1, costs, cfg, style)
        books[style] = book
        perf[style] = summarize(book, cfg)
        print(f"  {style:10s} net_ann={perf[style]['net_ann']:+.4f} "
              f"sharpe={perf[style]['net_sharpe']:.2f} HACt={perf[style]['net_hac_t']:.2f} "
              f"turnover={perf[style]['avg_turnover']:.2f}")

    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(books, cfg.paths.reports / "equity_curve.png")
    pd.concat({s: b for s, b in books.items()}, names=["style"]).to_csv(
        cfg.paths.reports / "phase3a_perf.csv")
    out = write_phase3a_results(cfg, perf, n_def, cost_split, cfg.backtest.horizons[1])
    print(f"\nDefaults carried: {n_def} | cost coverage: "
          f"{cost_split.get('measured', 0):.0%} measured\nResults: {out}")
    return 0


def _cmd_phase3b(args: argparse.Namespace) -> int:
    """Turnover frontier + contamination tests + real-return IC-decay."""
    import pandas as pd

    from crv.backtest import contamination as cont
    from crv.backtest.costs import build_cost_table
    from crv.backtest.defaults import detect_exits
    from crv.backtest.ic import cross_sectional_ic
    from crv.backtest.performance import summarize
    from crv.backtest.portfolio import run_portfolio
    from crv.backtest.returns import forward_excess_return, monthly_excess_return
    from crv.backtest.stats import newey_west_mean
    from crv.ingest.obap import load_panel
    from crv.io import read_table
    from crv.report.figures import plot_ic_decay, plot_turnover_frontier
    from crv.report.summary import write_phase3b_results

    cfg = load_config(args.config)
    inter = cfg.paths.interim
    for name, fn in (("universe", _cmd_universe), ("spreads", _cmd_spreads),
                     ("liquidity", _cmd_liquidity)):
        if not (inter / f"{name}.parquet").exists():
            fn(args)
    if not (inter / "signal-peer_shrunk.parquet").exists():
        cfg.model.kind = "peer_shrunk"
        _cmd_signal(args)

    panel = load_panel(cfg)
    spreads = read_table(inter / "spreads.parquet")
    signal = read_table(inter / "signal-peer_shrunk.parquet")
    liquidity = read_table(inter / "liquidity.parquet")
    exits = detect_exits(panel, cfg)
    costs = build_cost_table(panel, spreads, cfg)
    r1 = monthly_excess_return(spreads, exits, cfg)
    df = signal.merge(spreads[["cusip", "rebalance_date", "mod_duration"]],
                      on=["cusip", "rebalance_date"], how="left")

    # --- turnover frontier: holding sweep (band off) + a banded variant ----------
    def _row(variant, book):
        p = summarize(book, cfg)
        turn = book["turnover"].mean()
        be = (book["gross"].mean() / turn) if turn else float("nan")
        return {"variant": variant, "net_ann": p["net_ann"], "gross_ann": p["gross_ann"],
                "turnover": p["avg_turnover"], "sharpe": p["net_sharpe"],
                "hac_t": p["net_hac_t"], "breakeven_hs": be}

    rows = []
    for h in cfg.backtest.holding_grid:
        cfg.backtest.holding_months = h
        rows.append(_row(f"hold{h}m", run_portfolio(df, r1, costs, cfg, "ls", band=False)))
    cfg.backtest.holding_months = cfg.backtest.holding_grid[0]
    rows.append(_row(f"hold{cfg.backtest.holding_grid[0]}m+band",
                     run_portfolio(df, r1, costs, cfg, "ls", band=True)))
    frontier = pd.DataFrame(rows)

    # --- real-return IC-decay -----------------------------------------------------
    ic_rows = []
    for h in (1, 3, 6, 12):
        fwd = forward_excess_return(r1, h)
        m = signal.merge(fwd, on=["cusip", "rebalance_date"])
        ic = cross_sectional_ic(m)
        mean, se, t = newey_west_mean(ic, lags=h - 1)
        ic_rows.append({"horizon_m": h, "mean_ic": mean, "hac_t": t,
                        "n_periods": int(ic.notna().sum())})
    real_ic = pd.DataFrame(ic_rows)

    # --- contamination tests ------------------------------------------------------
    fwd3 = forward_excess_return(r1, cfg.backtest.default_window_m)
    impending = cont.impending_default_test(signal, fwd3, exits, cfg)
    liq = cont.liquidity_split_test(signal, fwd3, liquidity, cfg)
    odds = cont.default_odds_by_quantile(signal, exits, cfg)

    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    plot_turnover_frontier(frontier, cfg.paths.reports / "turnover_frontier.png")
    plot_ic_decay(real_ic.rename(columns={"hac_t": "t_stat"}).assign(hac_se=0.0),
                  cfg.paths.reports / "real_ic_decay.png")
    out = write_phase3b_results(cfg, frontier, real_ic, impending, liq, odds)
    print(frontier.round(4).to_string(index=False))
    print(f"\nimpending: full IC {impending['ic_full']:.3f} vs "
          f"survivors {impending['ic_survivors']:.3f} | liquidity: "
          f"liquid {liq['ic_liquid']:.3f} vs illiquid {liq['ic_illiquid']:.3f}")
    print(f"Results: {out}")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    """Phase 1.5 thin gate: IC at horizons + HAC + decay plot + quintile diagnostic."""
    from crv.backtest.ic import ic_table, ic_timeseries
    from crv.backtest.quintiles import quintile_returns, tail_loss_fraction
    from crv.io import read_table
    from crv.report.figures import plot_ic_decay, plot_ic_timeseries
    from crv.report.summary import write_phase1_5_summary

    cfg = load_config(args.config)
    signal = read_table(cfg.paths.interim / "signal.parquet")
    bt = cfg.backtest

    table = ic_table(signal, horizons=tuple(bt.horizons), method=bt.ic_method,
                     winsor_z=bt.winsor_z)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cfg.paths.reports.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.paths.reports / "ic_table.csv", index=False)
    decay_png = plot_ic_decay(table, cfg.paths.reports / "ic_decay.png")
    mid = bt.horizons[len(bt.horizons) // 2]
    ts = ic_timeseries(signal, mid, method=bt.ic_method)
    plot_ic_timeseries(ts, mid, cfg.paths.reports / f"ic_timeseries_{mid}m.png")
    quints = quintile_returns(signal, mid, n_quantiles=bt.n_quantiles)
    tail_frac = tail_loss_fraction(signal, mid, n_quantiles=bt.n_quantiles)

    summary = write_phase1_5_summary(cfg, table, quints, mid, tail_frac,
                                     model_kind=cfg.model.kind)
    print(f"\nQuintile {mid}m (mean vs median r):")
    print(quints.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"cheapest-bucket tail (<-5%): {tail_frac:.1%}")
    print(f"\nDecay plot: {decay_png}\nSummary:    {summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Shared options so `--config` works both before and after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/base.yaml", help="path to config YAML")

    p = argparse.ArgumentParser(
        prog="crv", description="Credit relative-value toolkit", parents=[common]
    )
    sub = p.add_subparsers(dest="stage", required=True)

    specs = [
        ("phase0", _cmd_phase0, "run the Phase 0 data-truth inventory"),
        ("fred", _cmd_fred, "download/cache the Treasury yield curve"),
        ("universe", _cmd_universe, "build the point-in-time universe membership"),
        ("spreads", _cmd_spreads, "compute G-spread + DTS on the universe"),
        ("liquidity", _cmd_liquidity, "compute liquidity controls (Bao gamma, Amihud, ...)"),
        ("signal", _cmd_signal, "fit fair value + standardized residual (per config model)"),
        ("phase1", _cmd_phase1, "run universe -> spreads -> signal (naive)"),
        ("phase2a", _cmd_phase2a, "run universe -> spreads -> liquidity -> signal (peer-shrunk)"),
        ("phase2b", _cmd_phase2b, "ML-vs-linear: peer_shrunk / ridge_wf / gbm_wf comparison"),
        ("phase3a", _cmd_phase3a, "net-of-cost neutralized backtest (returns, costs, defaults)"),
        ("phase3b", _cmd_phase3b, "turnover frontier + contamination tests + real-return decay"),
        ("backtest", _cmd_backtest, "Phase 1.5 thin IC/HAC signal-content gate"),
    ]
    for name, fn, help_text in specs:
        sp = sub.add_parser(name, parents=[common], help=help_text)
        if name == "fred":
            sp.add_argument("--refresh", action="store_true", help="re-download FRED series")
        sp.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
