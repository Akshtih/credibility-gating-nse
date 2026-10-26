"""Point-forecast metrics, generic over tasks.

Headline measure: skill versus the benchmark under the task's own loss,
    skill = 1 - mean(loss_model) / mean(loss_benchmark)
For the return task (squared error) this is the Campbell-Thompson out-of-sample R^2.
skill > 0 means the model beats the benchmark; <= 0 means it does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credibility.evaluation.tasks import Task, get_task  # noqa: F401  (get_task re-exported for callers)

KEYS = ["symbol", "origin_idx", "horizon"]


def _default_task() -> Task:
    return Task("return", ("naive",), "naive", (1,))


def align_with_benchmark(preds: pd.DataFrame, benchmark: str = "naive", task: Task | None = None) -> pd.DataFrame:
    """Add benchmark prediction/loss columns for the same ticker/origin/horizon; drop incomplete rows."""
    task = task or _default_task()
    bm = preds.loc[preds["model"] == benchmark, KEYS + ["pred"]].rename(columns={"pred": "pred_bm"})
    df = preds[preds["model"] != benchmark].merge(bm, on=KEYS, how="inner")
    df = df.dropna(subset=["pred", "pred_bm", "actual"]).copy()
    df["loss"] = task.loss(df["pred"].to_numpy(), df["actual"].to_numpy())
    df["loss_bm"] = task.loss(df["pred_bm"].to_numpy(), df["actual"].to_numpy())
    if task.name == "return":
        df["loss_abs"] = (df["actual"] - df["pred"]).abs()
        df["loss_abs_bm"] = (df["actual"] - df["pred_bm"]).abs()
    return df


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def summarize_group(g: pd.DataFrame) -> dict:
    n = len(g)
    lm, lb = float(g["loss"].mean()), float(g["loss_bm"].mean())
    skill = 1 - lm / lb if lb else float("nan")
    out = {"n": n, "loss": lm, "loss_bm": lb, "skill": skill, "r2_os": skill}
    if "loss_abs" in g.columns:
        out["rmse"] = float(np.sqrt(g["loss"].mean()))
        out["rmse_bm"] = float(np.sqrt(g["loss_bm"].mean()))
        mae, mae_bm = g["loss_abs"].mean(), g["loss_abs_bm"].mean()
        out["mae_ratio"] = float(mae / mae_bm) if mae_bm else float("nan")
        dirmask = (g["pred"] != 0) & (g["actual"] != 0)
        hits = int((np.sign(g.loc[dirmask, "pred"]) == np.sign(g.loc[dirmask, "actual"])).sum())
        nd = int(dirmask.sum())
        lo, hi = wilson_interval(hits, nd)
        out.update(dir_acc=hits / nd if nd else float("nan"), dir_ci_lo=lo, dir_ci_hi=hi,
                   always_up_rate=float((g.loc[dirmask, "actual"] > 0).mean()) if nd else float("nan"))
    else:
        for k in ("rmse", "rmse_bm", "mae_ratio", "dir_acc", "dir_ci_lo", "dir_ci_hi", "always_up_rate"):
            out[k] = float("nan")
    return out
