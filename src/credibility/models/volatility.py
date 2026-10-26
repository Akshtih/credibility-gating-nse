"""Volatility forecasters. Target: average daily variance over the next ``h`` trading days,

    RV_h = (1/h) * sum_{j=1..h} r_{T+j}^2     (r = daily log-return, squared return as variance proxy)

Every model returns ``predict(h)`` = forecast of RV_h in variance units (> 0).
Benchmark is ``hist_vol``: trailing 21-day mean squared return, held flat (the volatility analogue of naive).
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

from credibility.models.base import DEFAULT_HORIZONS, Forecaster

EPS = 1e-10
MAX_HORIZON = 63


def _returns(logp: pd.Series) -> np.ndarray:
    return logp.diff().dropna().to_numpy(dtype=float)


class HistVolForecaster(Forecaster):
    """Benchmark: mean r^2 over the last ``lookback`` days, flat across horizons."""
    name = "hist_vol"

    def __init__(self, lookback: int = 21) -> None:
        super().__init__()
        self.lookback = lookback
        self._v = EPS

    def get_params(self) -> dict:
        return {"lookback": self.lookback}

    def _fit(self, logp: pd.Series) -> None:
        r = _returns(logp)
        self._v = max(float(np.mean(r[-self.lookback:] ** 2)), EPS)

    def _predict(self, horizon: int) -> float:
        return self._v


class EWMAForecaster(Forecaster):
    """RiskMetrics EWMA variance (lambda=0.94), held flat."""
    name = "ewma"

    def __init__(self, lam: float = 0.94, window: int = 756) -> None:
        super().__init__()
        self.lam, self.window = lam, window
        self._v = EPS

    def get_params(self) -> dict:
        return {"lam": self.lam, "window": self.window}

    def _fit(self, logp: pd.Series) -> None:
        r = _returns(logp)[-self.window:]
        v = float(np.mean(r[:30] ** 2)) if len(r) >= 30 else float(np.mean(r ** 2))
        for x in r:
            v = self.lam * v + (1 - self.lam) * x * x
        self._v = max(v, EPS)

    def _predict(self, horizon: int) -> float:
        return self._v


class GARCHForecaster(Forecaster):
    """Zero-mean GARCH(1,1), normal quasi-ML; average of the analytic multi-step variance forecasts."""
    name = "garch"

    def __init__(self, window: int = 1000) -> None:
        super().__init__()
        self.window = window
        self._path: np.ndarray | None = None

    def get_params(self) -> dict:
        return {"window": self.window}

    def _fit(self, logp: pd.Series) -> None:
        from arch import arch_model

        r = _returns(logp)[-self.window:] * 100.0  # percent scale for optimiser stability
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
            var = res.forecast(horizon=MAX_HORIZON, reindex=False).variance.to_numpy()[-1]
        self._path = np.maximum(var / 1e4, EPS)

    def _predict(self, horizon: int) -> float:
        if horizon > MAX_HORIZON:
            raise ValueError(f"horizon {horizon} exceeds MAX_HORIZON={MAX_HORIZON}")
        return float(np.mean(self._path[:horizon]))


def har_features(r2: pd.Series) -> pd.DataFrame:
    """Causal HAR regressors at row t: daily, weekly (5d) and monthly (22d) mean squared return."""
    return pd.DataFrame({"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()}, index=r2.index)


class HARForecaster(Forecaster):
    """HAR-RV (Corsi 2009) on squared returns, one direct OLS per horizon.

    The last ``h`` rows have no realised target and are dropped. Forecasts are floored at 10% of the
    window's mean variance so OLS cannot return non-positive variance.
    """
    name = "har"

    def __init__(self, horizons: Sequence[int] = DEFAULT_HORIZONS, window: int = 1000) -> None:
        super().__init__()
        self.horizons, self.window = tuple(horizons), window
        self._coef: dict[int, np.ndarray] = {}
        self._x_last: np.ndarray | None = None
        self._floor = EPS

    def get_params(self) -> dict:
        return {"horizons": self.horizons, "window": self.window}

    def _fit(self, logp: pd.Series) -> None:
        r2 = (logp.diff() ** 2).dropna().iloc[-self.window:]
        if len(r2) < 150:
            raise ValueError(f"need at least 150 returns, got {len(r2)}")
        X = har_features(r2)
        self._floor = max(0.1 * float(r2.mean()), EPS)
        self._x_last = np.r_[1.0, X.iloc[-1].to_numpy()]
        self._coef = {}
        for h in self.horizons:
            y = r2[::-1].rolling(h).mean()[::-1].shift(-1)   # mean of r2 over t+1..t+h
            mask = X.notna().all(axis=1) & y.notna()
            A = np.c_[np.ones(int(mask.sum())), X[mask].to_numpy()]
            self._coef[h] = np.linalg.lstsq(A, y[mask].to_numpy(), rcond=None)[0]

    def _predict(self, horizon: int) -> float:
        if horizon not in self._coef:
            raise ValueError(f"horizon {horizon} not trained; trained horizons: {self.horizons}")
        return max(float(self._x_last @ self._coef[horizon]), self._floor)
