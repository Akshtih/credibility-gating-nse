"""Robustness checks on stored predictions (exploratory; they never change a gate decision or any original table).

skill_variants   skill of a model vs the benchmark under alternative views of the same forecasts:
                   full          all rows, the task's own loss (as in the main results)
                   ex_covid      origins outside 2020-02-15..2020-06-30
                   ex_top5       drop the 5% of rows with the highest REALISED value (selection on the outcome treats
                                 model and benchmark symmetrically; trimming on the benchmark's own loss would not)
                   alt_loss      volatility: squared error on the standard-deviation scale; return: absolute error
                   dates_won     share of origin dates on which the stock-averaged loss differential favours the model
dm_size          false-positive rate of our Diebold-Mariano implementation under a TRUE null with overlap dependence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credibility.evaluation.stats import dm_test
from credibility.evaluation.tasks import Task

COVID = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"))
KEYS = ["symbol", "origin_idx", "horizon"]


def cell_frame(preds: pd.DataFrame, task: Task, model: str, horizon: int) -> pd.DataFrame:
    """Model rows merged with the benchmark's prediction for the same stock/origin/horizon."""
    g = preds[(preds["model"] == model) & (preds["horizon"] == horizon)]
    b = preds[(preds["model"] == task.benchmark) & (preds["horizon"] == horizon)][KEYS + ["pred"]].rename(columns={"pred": "pb"})
    return g.merge(b, on=KEYS).dropna(subset=["pred", "pb", "actual"]).reset_index(drop=True)


def _skill(task: Task, g: pd.DataFrame) -> float:
    lm = task.loss(g["pred"].to_numpy(), g["actual"].to_numpy()).mean()
    lb = task.loss(g["pb"].to_numpy(), g["actual"].to_numpy()).mean()
    return float(1 - lm / lb) if lb else float("nan")


def skill_variants(preds: pd.DataFrame, task: Task, model: str, horizon: int) -> dict:
    g = cell_frame(preds, task, model, horizon)
    out = {"n": len(g), "full": _skill(task, g)}
    out["ex_covid"] = _skill(task, g[~g["origin_date"].between(*COVID)]) if len(g) else float("nan")
    keep = g["actual"] <= g["actual"].quantile(0.95)
    out["ex_top5"] = _skill(task, g[keep])
    if task.name == "volatility":
        sd = lambda p, a: (np.sqrt(np.maximum(a, 0)) - np.sqrt(np.maximum(p, 0))) ** 2
    else:
        sd = lambda p, a: np.abs(a - p)
    lm, lb = sd(g["pred"].to_numpy(), g["actual"].to_numpy()).mean(), sd(g["pb"].to_numpy(), g["actual"].to_numpy()).mean()
    out["alt_loss"] = float(1 - lm / lb) if lb else float("nan")
    d = task.loss(g["pred"].to_numpy(), g["actual"].to_numpy()) - task.loss(g["pb"].to_numpy(), g["actual"].to_numpy())
    out["dates_won"] = float((g.assign(d=d).groupby("origin_date")["d"].mean() < 0).mean())
    return out


def dm_size(n: int, lag: int, n_sims: int = 4000, seed: int = 1, level: float = 0.05) -> float:
    """Rejection rate of the one-sided DM test at ``level`` when the true mean loss differential is zero.

    lag=0: iid differentials. lag=L: equally-weighted MA(L) differentials, i.e. strongly overlapping outcomes
    (a deliberately demanding case; the real overlap of a 21-day horizon sampled every 10 days is milder).
    """
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(n_sims):
        e = rng.normal(0, 1, n + lag)
        d = np.convolve(e, np.ones(lag + 1) / (lag + 1), mode="valid")[:n] if lag else e[:n]
        rej += dm_test(d, np.zeros(n), lag)["p_better"] < level
    return rej / n_sims


def dm_size_overlap(n: int, step: int, h: int, lag: int, n_sims: int = 20000, seed: int = 1, level: float = 0.05) -> tuple[float, float]:
    """DM rejection rate under a TRUE null with the REAL overlap structure of rolling-origin forecasts.

    Daily innovations are iid; the loss differential at origin t is the sum of ``h`` consecutive innovations starting
    at t*step, so neighbouring origins share max(h - step, 0) innovations (for h=21, step=10: autocorrelations 0.52 and 0.05).
    Returns (rejection rate, Monte-Carlo standard error).
    """
    rng = np.random.default_rng(seed)
    starts = np.arange(n) * step
    rej = 0
    for _ in range(n_sims):
        c = np.concatenate([[0.0], np.cumsum(rng.normal(0, 1, n * step + h))])
        d = c[starts + h] - c[starts]
        rej += dm_test(d, np.zeros(n), lag)["p_better"] < level
    p = rej / n_sims
    return p, float(np.sqrt(p * (1 - p) / n_sims))
