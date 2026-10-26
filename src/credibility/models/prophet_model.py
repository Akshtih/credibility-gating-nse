"""Prophet on log-close. Future steps are business days after the last observed date.

Two variants share one fit:
    prophet           raw: cumulative log-return = yhat(future) - log(last observed close)
    prophet_anchored  anchored: yhat(future) - yhat(last date), i.e. only the *change* in Prophet's
                      fitted level. Removes the level error between the fitted curve and the last
                      close, which is the fairer way to use Prophet as a return forecaster.

The raw variant also exposes Prophet's native 80% interval so we can test its calibration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from credibility.models.base import Forecaster

MAX_HORIZON = 63
NATIVE_INTERVAL_WIDTH = 0.8


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(self, changepoint_prior_scale: float = 0.05, seasonality_prior_scale: float = 10.0,
                 window: int = 756, anchored: bool = False, uncertainty_samples: int = 200) -> None:
        super().__init__()
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.window = window
        self.anchored = anchored
        self.uncertainty_samples = 0 if anchored else uncertainty_samples
        self.name = "prophet_anchored" if anchored else "prophet"
        self._cum: np.ndarray | None = None
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None

    def get_params(self) -> dict:
        return {"changepoint_prior_scale": self.changepoint_prior_scale,
                "seasonality_prior_scale": self.seasonality_prior_scale, "window": self.window,
                "anchored": self.anchored, "uncertainty_samples": self.uncertainty_samples}

    def _fit(self, logp: pd.Series) -> None:
        from prophet import Prophet

        for noisy in ("cmdstanpy", "prophet"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        s = logp.iloc[-self.window:]
        df = pd.DataFrame({"ds": s.index, "y": s.to_numpy()})
        m = Prophet(
            growth="linear", seasonality_mode="additive",
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
            uncertainty_samples=self.uncertainty_samples, interval_width=NATIVE_INTERVAL_WIDTH,
        )
        m.fit(df)
        last = s.index[-1]
        dates = [last, *pd.bdate_range(last + pd.Timedelta(days=1), periods=MAX_HORIZON)]
        if self.uncertainty_samples:
            np.random.seed(0)  # Prophet samples with the global RNG; fix it for reproducibility
        out = m.predict(pd.DataFrame({"ds": dates}))
        yhat = out["yhat"].to_numpy()
        base = yhat[0] if self.anchored else float(s.iloc[-1])
        self._cum = yhat[1:] - base
        if self.uncertainty_samples:
            self._lo = out["yhat_lower"].to_numpy()[1:] - base
            self._hi = out["yhat_upper"].to_numpy()[1:] - base

    def _check(self, horizon: int) -> None:
        if horizon > MAX_HORIZON:
            raise ValueError(f"horizon {horizon} exceeds MAX_HORIZON={MAX_HORIZON}")

    def _predict(self, horizon: int) -> float:
        self._check(horizon)
        return float(self._cum[horizon - 1])

    def interval(self, horizon: int):
        if self._lo is None:
            return None
        self._check(horizon)
        return float(self._lo[horizon - 1]), float(self._hi[horizon - 1])
