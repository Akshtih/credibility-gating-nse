"""XGBoost on causal lag features, one direct model per horizon.

Features at row t use only data up to and including t. The training target for row t at
horizon h is log P_{t+h} - log P_t, so the last h rows of the history have no target and
are dropped (otherwise future information would leak into training).
Hyper-parameters are fixed and deliberately conservative; no tuning in Phase 2.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from credibility.models.base import DEFAULT_HORIZONS, Forecaster

LAGS = (1, 2, 3, 5, 10, 21)
MIN_ROWS = 126


def build_features(logp: pd.Series) -> pd.DataFrame:
    """Causal feature frame aligned to ``logp.index`` (NaN where history is too short)."""
    r = logp.diff()
    f = {f"ret_lag{k}": r.shift(k - 1) for k in LAGS}
    f["vol_5"] = r.rolling(5).std()
    f["vol_21"] = r.rolling(21).std()
    f["vol_63"] = r.rolling(63).std()
    f["mom_21"] = logp - logp.shift(21)
    f["mom_63"] = logp - logp.shift(63)
    f["dist_ma_50"] = logp - logp.rolling(50).mean()
    return pd.DataFrame(f, index=logp.index)


def build_targets(logp: pd.Series, horizon: int) -> pd.Series:
    """Forward cumulative log-return; NaN for the last ``horizon`` rows."""
    return logp.shift(-horizon) - logp


class XGBoostForecaster(Forecaster):
    name = "xgboost"

    def __init__(self, horizons: Sequence[int] = DEFAULT_HORIZONS, window: int = 1000, seed: int = 0,
                 n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.03,
                 subsample: float = 0.8, min_child_weight: float = 20.0) -> None:
        super().__init__()
        self.horizons = tuple(horizons)
        self.window, self.seed = window, seed
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                           subsample=subsample, min_child_weight=min_child_weight)
        self._models: dict[int, object] = {}
        self._x_last: pd.DataFrame | None = None

    def get_params(self) -> dict:
        return {"horizons": self.horizons, "window": self.window, "seed": self.seed, **self.params}

    def _fit(self, logp: pd.Series) -> None:
        from xgboost import XGBRegressor

        if len(logp) < MIN_ROWS:
            raise ValueError(f"need at least {MIN_ROWS} rows, got {len(logp)}")
        logp = logp.iloc[-self.window:]
        X = build_features(logp)
        self._x_last = X.iloc[[-1]]
        self._models = {}
        for h in self.horizons:
            y = build_targets(logp, h)
            mask = X.notna().all(axis=1) & y.notna()
            if mask.sum() < 50:
                raise ValueError(f"too few training rows for horizon {h}")
            model = XGBRegressor(random_state=self.seed, n_jobs=1, tree_method="hist", **self.params)
            model.fit(X[mask], y[mask])
            self._models[h] = model

    def _predict(self, horizon: int) -> float:
        if horizon not in self._models:
            raise ValueError(f"horizon {horizon} not trained; trained horizons: {self.horizons}")
        if self._x_last.isna().any(axis=None):
            raise ValueError("insufficient history to build features at the forecast origin")
        return float(self._models[horizon].predict(self._x_last)[0])
