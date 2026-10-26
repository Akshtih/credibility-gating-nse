"""Forecasting tasks. A task fixes: what is predicted, how it is realised, the loss, and the models.

return      target = cumulative log-return over h days; loss = squared error;      benchmark naive
volatility  target = mean daily squared return over next h days (variance proxy);
            loss = normalised QLIKE  log(p/a) + a/p - 1  (>= 0, zero iff p == a); benchmark hist_vol

``link`` maps the target to the scale where conformal residuals are formed
(identity for returns, log for variance so intervals stay positive).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credibility.config import EvalConfig

EPS = 1e-10


@dataclass(frozen=True)
class Task:
    name: str
    models: tuple[str, ...]
    benchmark: str
    horizons: tuple[int, ...]

    def realised(self, logc: np.ndarray, i: int, h: int) -> float:
        if self.name == "return":
            return float(logc[i + h] - logc[i])
        r = np.diff(logc[i: i + h + 1])
        return float(np.mean(r * r))

    def loss(self, pred, actual) -> np.ndarray:
        pred = np.asarray(pred, dtype=float)
        actual = np.asarray(actual, dtype=float)
        if self.name == "return":
            return (actual - pred) ** 2
        p = np.maximum(pred, EPS)
        a = np.maximum(actual, EPS)
        return np.log(p / a) + a / p - 1.0

    def link(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x if self.name == "return" else np.log(np.maximum(x, EPS))

    def inverse_link(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return z if self.name == "return" else np.exp(z)


TASK_NAMES = ("return", "volatility")


def get_task(cfg: EvalConfig, name: str = "return") -> Task:
    if name == "return":
        return Task("return", tuple(cfg.models), cfg.benchmark_model, tuple(cfg.horizons))
    if name == "volatility":
        return Task("volatility", tuple(cfg.vol_models), cfg.vol_benchmark, tuple(cfg.vol_horizons))
    raise ValueError(f"unknown task '{name}'; choose from {TASK_NAMES}")
