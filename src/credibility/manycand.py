"""Many-candidates experiment (protocol: configs/many.yaml, pre-registered).

build_candidates   extra forecasts for the SAME (stock, origin, horizon) keys as a stored development run.
                   Fitted candidates are computed from ``df.iloc[:i+1]`` only; shrunk and blended candidates are
                   linear combinations of stored (leakage-free) predictions.
evaluate_split     per (candidate, horizon) cell: skill and pooled DM p on the SELECTION part and on the VALIDATION part.
apply_rules        which cells each selection rule promotes (decided on SELECTION only).
rule_summary       how the promoted cells fared on VALIDATION.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from credibility.config import CONFIG_DIR
from credibility.data import Snapshot
from credibility.evaluation.metrics import KEYS
from credibility.evaluation.stats import dm_test, holm, overlap_lag
from credibility.evaluation.tasks import Task
from credibility.models import make_model

DEV_END = pd.Timestamp("2024-01-01")


def load_protocol(path=None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "many.yaml").read_text(encoding="utf-8"))


def _rows(key_frame: pd.DataFrame, name: str, pred: np.ndarray) -> pd.DataFrame:
    out = key_frame[["symbol", "origin_idx", "origin_date", "target_date", "horizon", "actual"]].copy()
    out["model"], out["pred"] = name, pred
    return out


def _past_return(logc: np.ndarray, i: int, k: int) -> float:
    return float(logc[i] - logc[i - k])


def return_candidates(base: pd.DataFrame, snapshot: str, spec: dict) -> pd.DataFrame:
    """Drift, momentum, reversal (from prices up to the origin) and shrunk copies of stored predictions."""
    key = base[base["model"] == "naive"].reset_index(drop=True)
    snap = Snapshot.open(snapshot)
    parts = []
    names: dict[str, np.ndarray] = {}
    for sym, g in key.groupby("symbol"):
        logc = np.log(snap.load(sym)["close"].to_numpy(float))
        i = g["origin_idx"].to_numpy()
        h = g["horizon"].to_numpy()
        for k in spec["drift_lookback"]:
            names.setdefault(f"drift_{k}", np.full(len(key), np.nan))[g.index] = [h_ * _past_return(logc, i_, k) / k for i_, h_ in zip(i, h)]
        for k in spec["momentum"]["lookback"]:
            for c in spec["momentum"]["scale"]:
                names.setdefault(f"mom_{k}_{c}", np.full(len(key), np.nan))[g.index] = [c * _past_return(logc, i_, k) for i_ in i]
        for k in spec["reversal"]["lookback"]:
            for c in spec["reversal"]["scale"]:
                names.setdefault(f"rev_{k}_{c}", np.full(len(key), np.nan))[g.index] = [-c * _past_return(logc, i_, k) for i_ in i]
    for name, pred in names.items():
        parts.append(_rows(key, name, pred))
    for b in spec["shrunk"]["base"]:
        src = base[base["model"] == b][KEYS + ["pred"]]
        m = key.merge(src, on=KEYS, how="left", suffixes=("_k", ""))
        for c in spec["shrunk"]["factor"]:
            parts.append(_rows(key, f"{b}_x{c}", c * m["pred"].to_numpy()))
    return pd.concat(parts, ignore_index=True)


def volatility_candidates(base: pd.DataFrame, snapshot: str, spec: dict, horizons) -> pd.DataFrame:
    """EWMA decay grid and alternative trailing windows (fitted on df.iloc[:i+1]), plus blends/scalings of stored predictions."""
    key = base[base["model"] == "hist_vol"].reset_index(drop=True)
    snap = Snapshot.open(snapshot)
    fitted = {f"ewma_{lam}": ("ewma", {"lam": lam}) for lam in spec["ewma_lambda"]}
    fitted.update({f"hist_{L}": ("hist_vol", {"lookback": L}) for L in spec["hist_lookback"]})
    preds = {n: np.full(len(key), np.nan) for n in fitted}
    for sym, g in key.groupby("symbol"):
        df = snap.load(sym)
        for i in g["origin_idx"].unique():
            hist = df.iloc[: i + 1]                                   # the only place history is sliced
            sel = g[g["origin_idx"] == i]
            for n, (model, params) in fitted.items():
                m = make_model(model, **params).fit(hist)
                preds[n][sel.index] = [m.predict(int(h)) for h in sel["horizon"]]
    parts = [_rows(key, n, p) for n, p in preds.items()]

    def stored(name):
        src = base[base["model"] == name][KEYS + ["pred"]]
        return key.merge(src, on=KEYS, how="left", suffixes=("_k", ""))["pred"].to_numpy()

    ew, ga, har = stored("ewma"), stored("garch"), stored("har")
    for w in spec["blend"]["ewma_garch_weight_on_ewma"]:
        parts.append(_rows(key, f"blend_ew_ga_{w}", w * ew + (1 - w) * ga))
    for w in spec["blend"]["ewma_har"]:
        parts.append(_rows(key, f"blend_ew_har_{w}", w * ew + (1 - w) * har))
    for w in spec["blend"]["garch_har"]:
        parts.append(_rows(key, f"blend_ga_har_{w}", w * ga + (1 - w) * har))
    for b in spec["scaled"]["base"]:
        for c in spec["scaled"]["factor"]:
            parts.append(_rows(key, f"{b}_x{c}", c * stored(b)))
    return pd.concat(parts, ignore_index=True)


def _cell_stats(g: pd.DataFrame, task: Task, lag: int) -> dict:
    lm = task.loss(g["pred"].to_numpy(), g["actual"].to_numpy())
    lb = task.loss(g["pb"].to_numpy(), g["actual"].to_numpy())
    skill = float(1 - lm.mean() / lb.mean()) if lb.mean() else float("nan")
    d = pd.Series(lm - lb, index=g.index).groupby(g["origin_date"].to_numpy()).mean().sort_index().to_numpy()
    return {"skill": skill, "p": dm_test(d, np.zeros_like(d), lag)["p_better"], "n_dates": len(d)}


def evaluate_split(preds: pd.DataFrame, task: Task, split: str, step: int, alpha: float = 0.05) -> pd.DataFrame:
    """One row per (model, horizon): skill/p on the SELECTION part (S) and on the VALIDATION part (V)."""
    split = pd.Timestamp(split)
    bm = preds[preds["model"] == task.benchmark][KEYS + ["pred"]].rename(columns={"pred": "pb"})
    df = preds[preds["model"] != task.benchmark].merge(bm, on=KEYS).dropna(subset=["pred", "pb", "actual"])
    sel = df[df["target_date"] < split]                       # outcomes realised before the split
    val = df[(df["origin_date"] >= split) & (df["target_date"] < DEV_END)]
    rows = []
    for (m, h), g in df.groupby(["model", "horizon"]):
        lag = overlap_lag(int(h), step)
        s, v = _cell_stats(sel[(sel.model == m) & (sel.horizon == h)], task, lag), _cell_stats(val[(val.model == m) & (val.horizon == h)], task, lag)
        rows.append({"model": m, "horizon": int(h), "skill_S": s["skill"], "p_S": s["p"], "n_S": s["n_dates"],
                     "skill_V": v["skill"], "p_V": v["p"], "n_V": v["n_dates"]})
    t = pd.DataFrame(rows)
    t["padj_S"] = holm(t["p_S"].to_numpy())
    t["padj_V"] = holm(t["p_V"].to_numpy())
    t["holds"] = t["skill_V"] > 0
    t["holds_sig"] = t["holds"] & (t["padj_V"] < alpha)
    return t


def apply_rules(t: pd.DataFrame, alpha: float = 0.05) -> dict[str, pd.Series]:
    best = t.loc[t.groupby("horizon")["skill_S"].idxmax()]
    best_idx = best[best["skill_S"] > 0].index
    return {
        "gate": (t["skill_S"] > 0) & (t["padj_S"] < alpha),
        "raw_dm": (t["skill_S"] > 0) & (t["p_S"] < alpha),
        "simple": t["skill_S"] > 0,
        "best_only": pd.Series(t.index.isin(best_idx), index=t.index),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def rule_summary(t: pd.DataFrame, rules: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, mask in rules.items():
        sel = t[mask]
        n, fail = len(sel), int((~sel["holds"]).sum())
        lo, hi = wilson(fail, n)
        rows.append({"rule": name, "n_cells": len(t), "n_promoted": n, "n_fail": fail,
                     "fail_rate": fail / n if n else float("nan"), "fail_lo": lo, "fail_hi": hi,
                     "mean_skill_V": float(sel["skill_V"].mean()) if n else float("nan"),
                     "n_hold_sig": int(sel["holds_sig"].sum()), "n_good_available": int(t["holds_sig"].sum())})
    return pd.DataFrame(rows)


def null_world(rng: np.random.Generator, n_stocks: int = 8, n_dates: int = 120, n_cand: int = 30) -> pd.DataFrame:
    """Synthetic return-like world in which every candidate has (essentially) zero skill: noise forecasts of a martingale."""
    dates = pd.bdate_range("2019-02-01", periods=n_dates * 10)[::10]
    rows = []
    for s in range(n_stocks):
        act = rng.normal(0, 1, n_dates)
        for name, pred in [("bm", np.zeros(n_dates))] + [(f"c{j}", rng.normal(0, 0.01, n_dates)) for j in range(n_cand)]:
            rows.append(pd.DataFrame({"symbol": f"S{s}", "model": name, "origin_idx": np.arange(n_dates) * 10, "horizon": 5,
                                      "origin_date": dates, "target_date": dates + pd.offsets.BDay(5), "pred": pred, "actual": act}))
    return pd.concat(rows, ignore_index=True)


def null_check(reps: int = 100, seed: int = 0, split: str = "2021-06-01") -> dict:
    """Run the full select-then-validate pipeline in the zero-skill world; the truth is known (no candidate is better)."""
    task = Task("return", ("bm",), "bm", (5,))
    rng = np.random.default_rng(seed)
    any_gate = gate_cells = simple_cells = simple_fail = total = 0
    for _ in range(reps):
        t = evaluate_split(null_world(rng), task, split, step=10)
        r = apply_rules(t)
        any_gate += int(r["gate"].any())
        gate_cells += int(r["gate"].sum())
        simple_cells += int(r["simple"].sum())
        simple_fail += int((r["simple"] & ~t["holds"]).sum())
        total += len(t)
    return {"reps": reps, "gate_any_rate": any_gate / reps, "gate_cell_rate": gate_cells / total,
            "simple_cell_rate": simple_cells / total, "simple_fail_share": simple_fail / max(simple_cells, 1)}
