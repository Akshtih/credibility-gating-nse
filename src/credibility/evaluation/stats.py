"""Significance tests for forecast comparison.

* ``dm_test``  Diebold-Mariano with a Newey-West (Bartlett) long-run variance and the
  Harvey-Leybourne-Newbold small-sample correction, compared with a Student-t.
  ``lag`` is the number of overlapping neighbouring observations: with origins every
  ``step`` days and a horizon ``h``, lag = max(0, ceil(h/step) - 1).
* ``holm``     Holm step-down adjustment for multiple comparisons.
* ``block_bootstrap_ci``  moving-block bootstrap CI for a mean loss differential.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sst


def overlap_lag(horizon: int, step: int) -> int:
    return max(0, math.ceil(horizon / step) - 1)


def long_run_variance(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=float)
    n = len(d)
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * float(dc[k:] @ dc[:-k]) / n
    return lrv


def dm_test(loss_model, loss_benchmark, lag: int = 0) -> dict:
    """Test H1: the model has LOWER expected loss than the benchmark (one-sided).

    d_t = loss_model - loss_benchmark;  mean(d) < 0 favours the model.
    Returns mean_diff, stat (HLN-corrected), p_better (one-sided), p_two_sided, n.
    """
    d = np.asarray(loss_model, dtype=float) - np.asarray(loss_benchmark, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()) if n else float("nan"),
           "stat": float("nan"), "p_better": float("nan"), "p_two_sided": float("nan")}
    if n < 10:
        return out
    lrv = long_run_variance(d, lag)
    if not lrv > 0:
        return out
    stat = d.mean() / math.sqrt(lrv / n)
    k = lag + 1  # effective "horizon" in observation units for the HLN correction
    corr = (n + 1 - 2 * k + k * (k - 1) / n) / n
    stat *= math.sqrt(max(corr, 1e-12))
    out["stat"] = float(stat)
    out["p_better"] = float(sst.t.cdf(stat, df=n - 1))
    out["p_two_sided"] = float(2 * sst.t.sf(abs(stat), df=n - 1))
    return out


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaNs are left as NaN and not counted)."""
    p = np.asarray(pvals, dtype=float)
    adj = np.full_like(p, np.nan)
    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adj
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def block_bootstrap_ci(d, block: int, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Moving-block bootstrap percentile CI for the mean of ``d``."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return (float("nan"), float("nan"))
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts_max = n - block + 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, n_blocks)
        sample = np.concatenate([d[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))
