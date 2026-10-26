"""Coverage, width and interval score for interval methods, plus causal market-regime labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credibility.evaluation.tasks import Task

METHODS = ("split", "aci", "gauss", "native")


def interval_score(y, lo, hi, alpha: float):
    """Gneiting-Raftery interval score (lower is better): width + penalties for misses."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    return (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)


def coverage_table(iv: pd.DataFrame, task: Task, alpha: float = 0.2, by: tuple[str, ...] = ("model", "horizon"),
                   methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    """Empirical coverage / mean width (link scale) / mean interval score per method and group.

    Methods are compared on the SAME rows: those where split and ACI intervals exist (calibration warm-up
    excluded), so differences are not a sampling artefact.
    """
    valid = iv[iv["split_lo"].notna() & iv["aci_lo"].notna()]
    rows = []
    for key, g in valid.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        for m in methods:
            lo_c, hi_c = f"{m}_lo", f"{m}_hi"
            if lo_c not in g.columns:
                continue
            gm = g[g[lo_c].notna() & g[hi_c].notna()]
            if gm.empty:
                continue
            inside = ((gm["actual"] >= gm[lo_c]) & (gm["actual"] <= gm[hi_c])).to_numpy()
            width = (task.link(gm[hi_c].to_numpy()) - task.link(gm[lo_c].to_numpy()))
            rows.append({**dict(zip(by, key)), "method": m, "n": len(gm), "coverage": float(inside.mean()),
                         "mean_width": float(np.mean(width)),
                         "interval_score": float(np.mean(interval_score(task.link(gm["actual"].to_numpy()),
                                                                        task.link(gm[lo_c].to_numpy()),
                                                                        task.link(gm[hi_c].to_numpy()), alpha)))})
    return pd.DataFrame(rows)


def nifty_regimes(index_prices: pd.DataFrame, dates: pd.Series, vol_window: int = 21, min_obs: int = 252) -> pd.Series:
    """'high' / 'low' volatility regime at each date, using only data up to that date.

    Regime = trailing ``vol_window``-day realised volatility of the index above/below its EXPANDING median.
    """
    s = index_prices.sort_values("date").set_index("date")["close"]
    r = np.log(s).diff()
    vol = r.rolling(vol_window).std()
    med = vol.expanding(min_periods=min_obs).median()
    label = pd.Series(np.where(vol > med, "high", "low"), index=s.index).where(med.notna())
    return label.reindex(pd.DatetimeIndex(dates), method="ffill").reset_index(drop=True)
