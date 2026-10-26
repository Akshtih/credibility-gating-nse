"""ARIMA on daily log-returns with a fixed small order (no order search: keeps it fast and honest).

Cumulative log-return forecast = sum of the first ``h`` step-ahead return forecasts.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from credibility.models.base import Forecaster

MAX_HORIZON = 63


class ARIMAForecaster(Forecaster):
    name = "arima"

    def __init__(self, order: tuple[int, int, int] = (1, 0, 1), window: int = 756, trend: str = "c") -> None:
        super().__init__()
        self.order, self.window, self.trend = order, window, trend
        self._cum: np.ndarray | None = None

    def get_params(self) -> dict:
        return {"order": self.order, "window": self.window, "trend": self.trend}

    def _fit(self, logp: pd.Series) -> None:
        from statsmodels.tsa.arima.model import ARIMA

        returns = logp.diff().dropna().to_numpy()[-self.window:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(returns, order=self.order, trend=self.trend).fit()
            path = np.asarray(res.forecast(MAX_HORIZON), dtype=float)
        self._cum = np.cumsum(path)

    def _predict(self, horizon: int) -> float:
        if horizon > MAX_HORIZON:
            raise ValueError(f"horizon {horizon} exceeds MAX_HORIZON={MAX_HORIZON}")
        return float(self._cum[horizon - 1])
