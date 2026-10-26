"""Common forecaster contract.

Every model predicts the **cumulative log-return** from the last observed close to
``horizon`` trading days ahead:

    predict(h) = E[ log(P_{T+h}) - log(P_T) ]

where T is the last row of the history passed to ``fit``. Using one target for all
models makes them directly comparable and makes the naive baseline exactly 0.

Leakage rule: a model may only use the rows it is given in ``fit``. The backtester
(Phase 3) is responsible for truncating history at the forecast origin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 21)


class NotFittedError(RuntimeError):
    pass


def log_prices(history: pd.DataFrame) -> pd.Series:
    """Validated log-close series indexed by date from a canonical price frame."""
    if not {"date", "close"} <= set(history.columns):
        raise ValueError("history must have 'date' and 'close' columns")
    df = history[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any() or not df["date"].is_monotonic_increasing:
        raise ValueError("history dates must be unique and ascending")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("history contains missing or non-positive closes")
    return pd.Series(np.log(df["close"].to_numpy(dtype=float)), index=pd.DatetimeIndex(df["date"]), name="log_close")


class Forecaster(ABC):
    """Stateful forecaster: ``fit(history)`` then ``predict(h)``."""

    name: str = "base"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def _fit(self, logp: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> float: ...

    def fit(self, history: pd.DataFrame) -> "Forecaster":
        """Fit on ``history`` (never mutated). Returns self."""
        self._fit(log_prices(history))
        self._fitted = True
        return self

    def predict(self, horizon: int) -> float:
        """Expected cumulative log-return over ``horizon`` trading days."""
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fitted before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return float(self._predict(int(horizon)))

    def predict_many(self, horizons: Sequence[int]) -> dict[int, float]:
        return {h: self.predict(h) for h in horizons}

    def interval(self, horizon: int) -> tuple[float, float] | None:
        """Model-native prediction interval on the same scale as ``predict`` (None if unsupported)."""
        return None

    def get_params(self) -> dict:
        return {}
