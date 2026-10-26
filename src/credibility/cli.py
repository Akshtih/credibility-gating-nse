"""Command line entry point:  python -m credibility.cli <command>

    data fetch      build the snapshot defined in configs/data.yaml
    data verify     re-hash every file in a snapshot
    data report     print a per-symbol quality table
"""

from __future__ import annotations

import argparse
import sys

from credibility.config import load_data_config
from credibility.data import Snapshot, build_snapshot


def _cmd_fetch(args) -> int:
    cfg = load_data_config(args.config)
    snap = build_snapshot(cfg, overwrite=args.overwrite)
    ok = snap.symbols()
    total = len(cfg.symbols) + 1
    print(f"\nsnapshot '{cfg.snapshot_name}': {len(ok)}/{total} symbols passed the quality gate "
          f"({len(snap.manifest['failed'])} fetch failures)")
    return 0


def _cmd_verify(args) -> int:
    cfg = load_data_config(args.config)
    snap = Snapshot.open(cfg.snapshot_name)
    results = snap.verify_all()
    bad = [s for s, ok in results.items() if not ok]
    print(f"verified {len(results)} files; {len(bad)} corrupted" + (f": {bad}" if bad else ""))
    return 1 if bad else 0


def _cmd_report(args) -> int:
    cfg = load_data_config(args.config)
    snap = Snapshot.open(cfg.snapshot_name)
    print(f"{'symbol':<16}{'rows':>6}  {'start':<11}{'end':<11}{'passed':<7} issues")
    for sym, m in sorted(snap.manifest["symbols"].items()):
        issues = "; ".join(f"{i['severity']}:{i['check']}" for i in m["quality"]["issues"]) or "-"
        print(f"{sym:<16}{m['rows']:>6}  {m['start']:<11}{m['end']:<11}{str(m['quality']['passed']):<7} {issues}")
    for sym, why in snap.manifest["failed"].items():
        print(f"{sym:<16}  FETCH FAILED: {why}")
    return 0


def _cmd_eval_run(args) -> int:
    from credibility.config import load_eval_config
    from credibility.evaluation import run_backtest

    cfg = load_eval_config(args.eval_config)
    run_backtest(
        cfg, args.period, task=args.task, run_name=args.name,
        symbols=args.tickers.split(",") if args.tickers else None,
        models=args.models.split(",") if args.models else None,
        max_origins=args.max_origins, workers=args.workers,
        confirm_holdout=args.confirm_holdout,
    )
    return 0


def _cmd_eval_report(args) -> int:
    import pandas as pd

    from credibility.config import load_eval_config
    from credibility.evaluation import format_table, summarize

    cfg = load_eval_config(args.eval_config)
    run_dir = cfg.results_path / args.run
    tbl = summarize(pd.read_parquet(run_dir / "predictions.parquet"), cfg, args.task)
    tbl.to_csv(run_dir / "summary.csv", index=False)
    print(format_table(tbl))
    print(f"\nfull table: {run_dir / 'summary.csv'}")
    return 0


def _cmd_gate_build(args) -> int:
    from pathlib import Path

    from credibility.config import PROJECT_ROOT, load_eval_config
    from credibility.gate import build_gate

    cfg = load_eval_config(args.eval_config)
    root = Path(args.registry) if args.registry else PROJECT_ROOT / "registry"
    for task, run in (("return", args.ret_run), ("volatility", args.vol_run)):
        if run:
            d = build_gate(cfg, task, cfg.results_path / run, root)
            print(d[["task", "model", "horizon", "skill", "dm_p_adj", "coverage", "promote"]].round(4).to_string(index=False))
    return 0


def _cmd_serve_forecast(args) -> int:
    import json
    from pathlib import Path

    from credibility.config import PROJECT_ROOT, load_eval_config
    from credibility.gate import Registry, serve_forecast

    cfg = load_eval_config(args.eval_config)
    reg = Registry(Path(args.registry) if args.registry else PROJECT_ROOT / "registry")
    print(json.dumps(serve_forecast(args.symbol, args.task, args.horizon, cfg, reg, as_of=args.as_of), indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="credibility")
    p.add_argument("--config", default=None, help="path to data.yaml (default: configs/data.yaml)")
    p.add_argument("--eval-config", default=None, help="path to eval.yaml (default: configs/eval.yaml)")
    sub = p.add_subparsers(dest="group", required=True)
    ev = sub.add_parser("eval").add_subparsers(dest="cmd", required=True)
    r = ev.add_parser("run")
    r.add_argument("--period", choices=["dev", "holdout"], required=True)
    r.add_argument("--task", choices=["return", "volatility"], default="return")
    r.add_argument("--name", default=None)
    r.add_argument("--tickers", default=None, help="comma-separated subset")
    r.add_argument("--models", default=None, help="comma-separated subset")
    r.add_argument("--max-origins", type=int, default=None, help="cap origins per ticker (quick runs)")
    r.add_argument("--workers", type=int, default=None)
    r.add_argument("--confirm-holdout", action="store_true", help="required for --period holdout (runs once)")
    r.set_defaults(func=_cmd_eval_run)
    g = sub.add_parser("gate").add_subparsers(dest="cmd", required=True)
    gb = g.add_parser("build")
    gb.add_argument("--ret-run", default=None, help="dev run directory name for the return task")
    gb.add_argument("--vol-run", default=None, help="dev run directory name for the volatility task")
    gb.add_argument("--registry", default=None)
    gb.set_defaults(func=_cmd_gate_build)
    sv = sub.add_parser("serve").add_subparsers(dest="cmd", required=True)
    sf = sv.add_parser("forecast")
    sf.add_argument("--symbol", required=True)
    sf.add_argument("--task", choices=["return", "volatility"], default="return")
    sf.add_argument("--horizon", type=int, default=5)
    sf.add_argument("--as-of", default=None)
    sf.add_argument("--registry", default=None)
    sf.set_defaults(func=_cmd_serve_forecast)
    rep = ev.add_parser("report")
    rep.add_argument("--run", required=True, help="run directory name under results/")
    rep.add_argument("--task", choices=["return", "volatility"], default="return")
    rep.set_defaults(func=_cmd_eval_report)
    data = sub.add_parser("data").add_subparsers(dest="cmd", required=True)
    f = data.add_parser("fetch")
    f.add_argument("--overwrite", action="store_true", help="rebuild an existing snapshot (breaks reproducibility)")
    f.set_defaults(func=_cmd_fetch)
    data.add_parser("verify").set_defaults(func=_cmd_verify)
    data.add_parser("report").set_defaults(func=_cmd_report)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
