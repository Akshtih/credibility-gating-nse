"""Rolling split-conformal and Adaptive Conformal Inference (ACI) intervals for stored forecasts.

For every series (symbol, model, horizon), ordered by forecast origin, the interval at origin k is built
from the absolute link-scale residuals of PAST forecasts whose outcome was already realised at k
(``target_date <= origin_date_k``). Nothing realised after the origin is ever used, which is what makes the
intervals deployable and the coverage honest even with overlapping horizons.

Methods (nominal coverage 1 - alpha):
    split   quantile of the last ``cal_window`` realised |residuals|
    aci     same, with the miscoverage level adapted online (Gibbs & Candes, 2021):
            alpha_t <- alpha_t + gamma * (alpha - err_j) for each outcome j newly realised at the origin
    gauss   (return task only) pred +- z * sigma_d * sqrt(h): the standard random-walk Gaussian interval
    native  model-supplied interval if present (Prophet's 80% interval)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sst

from credibility.evaluation.tasks import Task


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = 0.2        # nominal coverage 80% (Prophet's default interval width)
    cal_window: int = 60
    min_cal: int = 20
    gamma: float = 0.02       # ACI step size


def conformal_quantile(abs_res: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)(1-alpha))-th smallest value (capped at the max)."""
    n = len(abs_res)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1 - alpha))
    s = np.sort(abs_res)
    return float(s[min(max(k, 1), n) - 1])


def series_intervals(origin_date: np.ndarray, target_date: np.ndarray, abs_res: np.ndarray,
                     cfg: ConformalConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q_split, q_aci, alpha_t) for one series sorted by origin. NaN where calibration is too short."""
    n = len(abs_res)
    q_split = np.full(n, np.nan)
    q_aci = np.full(n, np.nan)
    alpha_t_arr = np.full(n, np.nan)
    alpha_t = cfg.alpha
    processed = 0
    for k in range(n):
        # outcomes realised by the origin of forecast k (targets are non-decreasing for a fixed horizon)
        m = int(np.searchsorted(target_date, origin_date[k], side="right"))
        m = min(m, k)  # a forecast can never use itself or later ones
        for j in range(processed, m):
            if np.isfinite(q_aci[j]):
                err = 1.0 if abs_res[j] > q_aci[j] else 0.0
                alpha_t += cfg.gamma * (cfg.alpha - err)
        processed = max(processed, m)
        cal = abs_res[max(0, m - cfg.cal_window): m]
        if len(cal) < cfg.min_cal:
            continue
        q_split[k] = conformal_quantile(cal, cfg.alpha)
        a = min(max(alpha_t, 1e-3), 0.999)
        q_aci[k] = conformal_quantile(cal, a)
        alpha_t_arr[k] = alpha_t
    return q_split, q_aci, alpha_t_arr


def build_intervals(preds: pd.DataFrame, task: Task, cfg: ConformalConfig = ConformalConfig()) -> pd.DataFrame:
    """Add interval columns (original target scale) to every prediction row.

    Output columns added: split_lo/hi, aci_lo/hi, alpha_t, gauss_lo/hi (return task), native_lo/hi (if lo/hi given).
    """
    df = preds.sort_values(["symbol", "model", "horizon", "origin_idx"]).reset_index(drop=True).copy()
    z_pred = task.link(df["pred"].to_numpy())
    z_act = task.link(df["actual"].to_numpy())
    abs_res = np.abs(z_act - z_pred)
    qs = np.full(len(df), np.nan)
    qa = np.full(len(df), np.nan)
    at = np.full(len(df), np.nan)
    for _, idx in df.groupby(["symbol", "model", "horizon"]).indices.items():
        o = df["origin_date"].to_numpy()[idx]
        t = df["target_date"].to_numpy()[idx]
        s, a, alpha_arr = series_intervals(o, t, abs_res[idx], cfg)
        qs[idx], qa[idx], at[idx] = s, a, alpha_arr
    df["split_lo"] = task.inverse_link(z_pred - qs)
    df["split_hi"] = task.inverse_link(z_pred + qs)
    df["aci_lo"] = task.inverse_link(z_pred - qa)
    df["aci_hi"] = task.inverse_link(z_pred + qa)
    df["alpha_t"] = at
    if task.name == "return":
        z = sst.norm.ppf(1 - cfg.alpha / 2)
        half = z * df["sigma_d"].to_numpy() * np.sqrt(df["horizon"].to_numpy())
        df["gauss_lo"] = df["pred"] - half
        df["gauss_hi"] = df["pred"] + half
    if "lo" in df.columns:
        df["native_lo"], df["native_hi"] = df["lo"], df["hi"]
    return df
