"""Serve a credibility-checked forecast.

``serve_forecast`` looks up the registered model for (task, horizon), re-fits its spec on snapshot data up to
``as_of`` (no future rows), forecasts, attaches a conformal interval computed from stored out-of-sample
residuals realised before ``as_of``, and returns the evidence behind the verdict. If no candidate passed the
gate the benchmark is served and the response says so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credibility.config import EvalConfig
from credibility.data import Snapshot
from credibility.evaluation import get_task
from credibility.gate.registry import Registry
from credibility.models import HORIZON_AWARE, make_model
from credibility.uncertainty import ConformalConfig, conformal_quantile, series_intervals

VERDICT_TEXT = {
    "TRUSTED": "Model beat the benchmark for this stock (significant, multiple-testing corrected).",
    "CAUTION": "Model was better than the benchmark for this stock, but not significantly.",
    "NOT_BETTER": "Model was not better than the benchmark for this stock.",
    "BENCHMARK_ONLY": "No candidate model passed the credibility gate; the benchmark forecast is served.",
}


class ServeError(RuntimeError):
    pass


def _calibration_series(cfg: EvalConfig, entry: dict, symbol: str) -> pd.DataFrame:
    frames = []
    for run in entry.get("calibration_runs", []):
        path = cfg.results_path / run / "predictions.parquet"
        if path.exists():
            f = pd.read_parquet(path)
            frames.append(f[(f["symbol"] == symbol) & (f["model"] == entry["model"]) & (f["horizon"] == entry["horizon"])])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_values("origin_idx").drop_duplicates(["origin_date"])


def interval_at(cal: pd.DataFrame, task, as_of: pd.Timestamp, horizon: int, conf: ConformalConfig):
    """ACI half-width (link scale) for a forecast made at ``as_of`` from outcomes realised on/before it."""
    if cal.empty:
        return None
    cal = cal[cal["origin_date"] <= as_of].sort_values("origin_date")
    if cal.empty:
        return None
    res = np.abs(task.link(cal["actual"].to_numpy()) - task.link(cal["pred"].to_numpy()))
    o = np.append(cal["origin_date"].to_numpy(), np.datetime64(as_of))
    t = np.append(cal["target_date"].to_numpy(), np.datetime64(as_of + pd.offsets.BDay(horizon)))
    _, q_aci, _ = series_intervals(o, t, np.append(res, 0.0), conf)
    q = q_aci[-1]
    return None if not np.isfinite(q) else float(q)


def serve_forecast(symbol: str, task_name: str, horizon: int, cfg: EvalConfig, registry: Registry,
                   as_of: str | None = None, conf: ConformalConfig = ConformalConfig(),
                   verdicts: pd.DataFrame | None = None) -> dict:
    if verdicts is None:
        vp = registry.root / "evidence" / task_name / "ticker_verdicts.csv"
        verdicts = pd.read_csv(vp) if vp.exists() else None
    entry = registry.served(task_name, horizon)
    if entry is None:
        raise ServeError(f"nothing registered for task '{task_name}' horizon {horizon}; run `gate build` first")
    task = get_task(cfg, task_name)
    snap = Snapshot.open(entry["snapshot"])
    if symbol not in snap.symbols():
        raise ServeError(f"{symbol} is not in snapshot '{entry['snapshot']}' (or failed the quality gate)")
    df = snap.load(symbol)
    if as_of:
        df = df[df["date"] <= pd.Timestamp(as_of)]
    if len(df) < cfg.min_train_rows:
        raise ServeError(f"not enough history for {symbol}")
    as_of_ts = pd.Timestamp(df["date"].iloc[-1])

    params = dict(entry["params"])
    if entry["model"] in HORIZON_AWARE:
        params["horizons"] = tuple(task.horizons)
    model = make_model(entry["model"], **{k: v for k, v in params.items() if k in _accepted(entry["model"])}).fit(df)
    pred = model.predict(horizon)

    q = interval_at(_calibration_series(cfg, entry, symbol), task, as_of_ts, horizon, conf)
    z = float(task.link(pred))
    lo = float(task.inverse_link(z - q)) if q is not None else None
    hi = float(task.inverse_link(z + q)) if q is not None else None

    last_close = float(df["close"].iloc[-1])
    out = {
        "symbol": symbol, "task": task_name, "horizon_days": horizon, "as_of": as_of_ts.strftime("%Y-%m-%d"),
        "served_model": entry["model"], "role": entry["role"], "registry_entry": entry["id"],
        "forecast": pred, "interval_80": [lo, hi], "nominal_coverage": 1 - conf.alpha,
        "evidence": {"dev_skill": entry["metrics"].get("skill"), "dm_p_adj": entry["metrics"].get("dm_p_adj"),
                     "coverage": entry["metrics"].get("coverage"), "reason": entry["reason"]},
    }
    if task_name == "return":
        out["last_close"] = last_close
        out["forecast_price"] = last_close * float(np.exp(pred))
        out["interval_price_80"] = [last_close * float(np.exp(lo)), last_close * float(np.exp(hi))] if q is not None else [None, None]
    else:
        out["forecast_annualised_vol"] = float(np.sqrt(pred * 252))
        out["interval_vol_80"] = [float(np.sqrt(lo * 252)), float(np.sqrt(hi * 252))] if q is not None else [None, None]

    if entry["role"] == "fallback":
        out["verdict"] = "BENCHMARK_ONLY"
    elif verdicts is not None:
        v = verdicts[(verdicts["symbol"] == symbol) & (verdicts["model"] == entry["model"]) & (verdicts["horizon"] == horizon)]
        out["verdict"] = v["verdict"].iloc[0] if len(v) else "CAUTION"
    else:
        out["verdict"] = "CAUTION"
    out["verdict_text"] = VERDICT_TEXT[out["verdict"]]
    return out


def _accepted(model: str) -> set[str]:
    """Constructor kwargs we may pass from a stored spec (get_params keys the constructor accepts)."""
    import inspect

    m = make_model(model)
    return set(inspect.signature(type(m).__init__).parameters) - {"self"}
