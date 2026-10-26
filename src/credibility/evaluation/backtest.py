"""Rolling-origin backtester (return and volatility tasks).

For every (ticker, origin) the model is fitted on ``df.iloc[:i+1]`` ONLY and predicts each horizon.
The realised value is computed by the task from prices at/after the origin.

Periods
-------
dev      origins whose realised outcome date is strictly BEFORE ``holdout_start``
         (purged: no dev result depends on holdout prices).
holdout  origins on/after ``holdout_start``. Runnable once per task; a lock file blocks re-runs.

Output: ``<results_dir>/<run_name>/predictions.parquet`` (one row per ticker x origin x model x horizon,
with optional model-native interval ``lo``/``hi`` and the causal trailing daily return std ``sigma_d``)
and ``failures.csv``, ``run.json``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from credibility.config import EvalConfig
from credibility.data import Snapshot
from credibility.evaluation.tasks import Task, get_task
from credibility.models import HORIZON_AWARE, make_model

PERIODS = ("dev", "holdout")
SIGMA_WINDOW = 252


class HoldoutLockedError(RuntimeError):
    pass


def forecast_full(model_name: str, df: pd.DataFrame, i: int, horizons: Sequence[int]):
    """Fit on rows 0..i inclusive; return ({h: pred}, {h: (lo, hi) or None}). The ONLY place history is sliced."""
    hist = df.iloc[: i + 1]
    params = {"horizons": tuple(horizons)} if model_name in HORIZON_AWARE else {}
    m = make_model(model_name, **params).fit(hist)
    return m.predict_many(horizons), {h: m.interval(h) for h in horizons}


def forecast_at_origin(model_name: str, df: pd.DataFrame, i: int, horizons: Sequence[int]) -> dict[int, float]:
    return forecast_full(model_name, df, i, horizons)[0]


def origin_grid(n_rows: int, cfg: EvalConfig) -> range:
    """Origin row indices, anchored at min_train_rows-1 so the grid is identical across periods."""
    return range(cfg.min_train_rows - 1, n_rows - 1, cfg.origin_step)


def is_valid(period: str, origin_date: pd.Timestamp, target_date: pd.Timestamp, holdout_start: pd.Timestamp) -> bool:
    if period == "dev":
        return target_date < holdout_start          # purge: outcome must not touch the holdout
    if period == "holdout":
        return origin_date >= holdout_start
    raise ValueError(f"period must be one of {PERIODS}")


def _run_symbol(args) -> tuple[list[dict], list[dict]]:
    snapshot, symbol, cfg, task, period, models, max_origins = args
    df = Snapshot.open(snapshot).load(symbol)
    dates = df["date"].to_numpy()
    logc = np.log(df["close"].to_numpy(dtype=float))
    rets = np.diff(logc, prepend=np.nan)
    n = len(df)
    hold = pd.Timestamp(cfg.holdout_start)
    rows: list[dict] = []
    fails: list[dict] = []
    done = 0
    for i in origin_grid(n, cfg):
        valid_h = [h for h in task.horizons
                   if i + h < n and is_valid(period, pd.Timestamp(dates[i]), pd.Timestamp(dates[i + h]), hold)]
        if not valid_h:
            continue
        if max_origins is not None and done >= max_origins:
            break
        done += 1
        sigma_d = float(np.nanstd(rets[max(1, i - SIGMA_WINDOW + 1): i + 1], ddof=1))
        for name in models:
            try:
                preds, ivals = forecast_full(name, df, i, task.horizons)
            except Exception as exc:  # a model failing must not kill the run
                fails.append({"symbol": symbol, "model": name, "origin_idx": i,
                              "origin_date": pd.Timestamp(dates[i]), "error": repr(exc)[:200]})
                continue
            for h in valid_h:
                iv = ivals.get(h)
                rows.append({
                    "symbol": symbol, "model": name, "origin_idx": i,
                    "origin_date": pd.Timestamp(dates[i]), "target_date": pd.Timestamp(dates[i + h]),
                    "horizon": h, "pred": preds[h], "actual": task.realised(logc, i, h),
                    "lo": iv[0] if iv else np.nan, "hi": iv[1] if iv else np.nan, "sigma_d": sigma_d,
                })
    return rows, fails


def run_backtest(
    cfg: EvalConfig,
    period: str,
    task: str | Task = "return",
    run_name: str | None = None,
    symbols: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    max_origins: int | None = None,
    workers: int | None = None,
    confirm_holdout: bool = False,
    log=print,
) -> Path:
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}")
    task = get_task(cfg, task) if isinstance(task, str) else task
    out_root = cfg.results_path
    out_root.mkdir(parents=True, exist_ok=True)
    lock = out_root / f"holdout_{task.name}.lock"
    if period == "holdout":
        if not confirm_holdout:
            raise HoldoutLockedError("holdout is run once; pass confirm_holdout=True (CLI: --confirm-holdout)")
        if lock.exists():
            raise HoldoutLockedError(f"holdout already run for task '{task.name}': {lock.read_text(encoding='utf-8')[:200]}")
        lock.write_text(json.dumps({"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    "task": task.name, "holdout_start": cfg.holdout_start,
                                    "snapshot": cfg.snapshot}), encoding="utf-8")

    snap = Snapshot.open(cfg.snapshot)
    universe = list(symbols) if symbols else [s for s in snap.symbols() if not s.startswith("^")]
    models = list(models) if models else list(task.models)
    run_name = run_name or f"{task.name}_{period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = out_root / run_name
    out.mkdir(parents=True, exist_ok=False)

    jobs = [(cfg.snapshot, s, cfg, task, period, models, max_origins) for s in universe]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 2))
    log(f"[{run_name}] {task.name}/{period}: {len(universe)} tickers x {len(models)} models, {workers} workers")
    rows: list[dict] = []
    fails: list[dict] = []
    if workers == 1:
        results = map(_run_symbol, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_run_symbol, jobs)
    for job, (r, f) in zip(jobs, results):
        rows += r
        fails += f
        log(f"  {job[1]}: {len(r)} predictions, {len(f)} failures")
    preds = pd.DataFrame(rows)
    preds.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(fails, columns=["symbol", "model", "origin_idx", "origin_date", "error"]).to_csv(out / "failures.csv", index=False)
    (out / "run.json").write_text(json.dumps({
        "task": task.name, "period": period, "snapshot": cfg.snapshot, "models": models,
        "benchmark": task.benchmark, "horizons": list(task.horizons),
        "symbols": universe, "origin_step": cfg.origin_step, "min_train_rows": cfg.min_train_rows,
        "holdout_start": cfg.holdout_start, "max_origins": max_origins,
        "n_predictions": int(len(preds)), "n_failures": len(fails),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    log(f"[{run_name}] wrote {len(preds)} predictions, {len(fails)} failures -> {out}")
    return out
