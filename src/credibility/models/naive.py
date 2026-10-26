"""Random-walk baseline: tomorrow's price equals today's, so every cumulative log-return is 0."""

from __future__ import annotations

import pandas as pd

from credibility.models.base import Forecaster


class NaiveForecaster(Forecaster):
    name = "naive"

    def _fit(self, logp: pd.Series) -> None:
        pass

    def _predict(self, horizon: int) -> float:
        return 0.0
