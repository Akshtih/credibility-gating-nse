"""Replay serving policies over stored backtest predictions.

Every model's forecast at every origin is already stored, so we can ask: had we served the candidate,
the benchmark, or the candidate guarded by a monitor, what loss would we have incurred?

Information discipline: at origin k the monitor has seen only outcomes with ``target_date <= origin_date_k``.
The candidate keeps running in shadow while the benchmark is served, so the monitor can observe recovery.
(The models themselves are refit at every origin in the backtest, i.e. continuous retraining is the baseline;
what is tested here is switching between the candidate and the benchmark.)

Policies: benchmark | candidate | guarded_rolling (RollingRatioMonitor) | guarded_ph (Page-Hinkley)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credibility.evaluation.metrics import KEYS
from credibility.evaluation.tasks import Task
from credibility.monitoring.detectors import PageHinkley, RollingRatioMonitor, relative_differential

POLICIES = ("benchmark", "candidate", "guarded_rolling", "guarded_ph")
STRESS_WINDOWS = {"covid_2020": ("2020-02-15", "2020-06-30"), "drawdown_2022": ("2022-01-01", "2022-06-30")}


def _pair(preds: pd.DataFrame, task: Task, candidate: str, horizon: int) -> pd.DataFrame:
    c = preds[(preds["model"] == candidate) & (preds["horizon"] == horizon)]
    b = preds[(preds["model"] == task.benchmark) & (preds["horizon"] == horizon)]
    m = c.merge(b[KEYS + ["pred"]], on=KEYS, suffixes=("", "_bm")).dropna(subset=["pred", "pred_bm", "actual"])
    m["loss"] = task.loss(m["pred"].to_numpy(), m["actual"].to_numpy())
    m["loss_bm"] = task.loss(m["pred_bm"].to_numpy(), m["actual"].to_numpy())
    return m.sort_values(["symbol", "origin_idx"]).reset_index(drop=True)


def _replay_series(g: pd.DataFrame, k: int, theta: float, ph_delta: float, ph_thr: float) -> pd.DataFrame:
    o = g["origin_date"].to_numpy()
    t = g["target_date"].to_numpy()
    loss, bm = g["loss"].to_numpy(), g["loss_bm"].to_numpy()
    roll = RollingRatioMonitor(k, theta)
    ph = PageHinkley(ph_delta, ph_thr)
    ph_alarm, alarm_start = False, 0
    recent: list[float] = []
    seen = 0
    out = {p: np.empty(len(g)) for p in POLICIES}
    on = {"guarded_rolling": np.zeros(len(g), bool), "guarded_ph": np.zeros(len(g), bool)}
    for i in range(len(g)):
        m = min(int(np.searchsorted(t, o[i], side="right")), i)       # outcomes realised by this origin
        for j in range(seen, m):
            roll.update(loss[j], bm[j])
            x = relative_differential(loss[j], bm[j])
            recent.append(x)
            if not ph_alarm and ph.update(x):
                ph_alarm, alarm_start = True, len(recent)
            elif ph_alarm and len(recent) - alarm_start >= k and np.mean(recent[alarm_start:][-k:]) < 0:
                ph_alarm = False          # recovered: candidate beat the benchmark over k post-alarm outcomes
                ph.reset()
        seen = max(seen, m)
        out["benchmark"][i] = bm[i]
        out["candidate"][i] = loss[i]
        on["guarded_rolling"][i] = not roll.alarm
        out["guarded_rolling"][i] = loss[i] if not roll.alarm else bm[i]
        on["guarded_ph"][i] = not ph_alarm
        out["guarded_ph"][i] = loss[i] if not ph_alarm else bm[i]
    res = g[["symbol", "origin_date", "target_date", "horizon"]].copy()
    for p in POLICIES:
        res[f"loss_{p}"] = out[p]
    res["cand_on_rolling"], res["cand_on_ph"] = on["guarded_rolling"], on["guarded_ph"]
    return res


def replay(preds: pd.DataFrame, task: Task, candidate: str, horizon: int, k: int = 12, theta: float = 0.1,
           ph_delta: float = 0.1, ph_threshold: float = 4.0) -> pd.DataFrame:
    """Per-(symbol, origin) losses under each policy for one candidate and horizon."""
    pair = _pair(preds, task, candidate, horizon)
    parts = [_replay_series(g, k, theta, ph_delta, ph_threshold) for _, g in pair.groupby("symbol")]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def policy_table(rep: pd.DataFrame, extra_windows: dict | None = None) -> pd.DataFrame:
    """Skill vs the benchmark policy, overall and inside stress windows, plus switching behaviour."""
    rows = []
    for name in POLICIES:
        row = {"policy": name}
        for label, sel in (("all", slice(None)),) + tuple(
                (n, (rep["origin_date"] >= pd.Timestamp(a)) & (rep["origin_date"] <= pd.Timestamp(b)))
                for n, (a, b) in {**STRESS_WINDOWS, **(extra_windows or {})}.items()):
            d = rep if label == "all" else rep[sel]
            den = d["loss_benchmark"].sum()
            row[f"skill_{label}"] = float(1 - d[f"loss_{name}"].sum() / den) if den > 0 and len(d) else float("nan")
            row[f"n_{label}"] = int(len(d))
        key = {"guarded_rolling": "cand_on_rolling", "guarded_ph": "cand_on_ph"}.get(name)
        if key:
            on = rep[key]
            row["candidate_share"] = float(on.mean())
            row["switches_per_series"] = float(rep.groupby("symbol")[key].apply(lambda s: (s.astype(int).diff().abs() > 0).sum()).mean())
        else:
            row["candidate_share"] = 1.0 if name == "candidate" else 0.0
            row["switches_per_series"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def first_alarm_dates(rep: pd.DataFrame, window: tuple[str, str], col: str = "cand_on_rolling") -> pd.Series:
    """First origin date inside ``window`` where the guard switched the candidate off, per symbol."""
    a, b = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    sub = rep[(rep["origin_date"] >= a) & (rep["origin_date"] <= b) & (~rep[col])]
    return sub.groupby("symbol")["origin_date"].min()
