"""Phase 2: forecasters behind the ``Forecaster`` contract, for two tasks.

Return task  (``predict(h)`` = expected cumulative log-return):
    naive, arima, prophet, prophet_anchored, xgboost
Volatility task (``predict(h)`` = expected average daily variance over the next h days):
    hist_vol (benchmark), ewma, garch, har

Use ``make_model(name)`` so the backtester stays model-agnostic.
"""

from __future__ import annotations

from credibility.models.base import DEFAULT_HORIZONS, Forecaster, NotFittedError, log_prices
from credibility.models.naive import NaiveForecaster


def _arima(**kw):
    from credibility.models.arima import ARIMAForecaster
    return ARIMAForecaster(**kw)


def _prophet(**kw):
    from credibility.models.prophet_model import ProphetForecaster
    return ProphetForecaster(**kw)


def _prophet_anchored(**kw):
    from credibility.models.prophet_model import ProphetForecaster
    return ProphetForecaster(anchored=True, **kw)


def _xgboost(**kw):
    from credibility.models.xgboost_model import XGBoostForecaster
    return XGBoostForecaster(**kw)


def _hist_vol(**kw):
    from credibility.models.volatility import HistVolForecaster
    return HistVolForecaster(**kw)


def _ewma(**kw):
    from credibility.models.volatility import EWMAForecaster
    return EWMAForecaster(**kw)


def _garch(**kw):
    from credibility.models.volatility import GARCHForecaster
    return GARCHForecaster(**kw)


def _har(**kw):
    from credibility.models.volatility import HARForecaster
    return HARForecaster(**kw)


RETURN_FACTORIES = {"naive": NaiveForecaster, "arima": _arima, "prophet": _prophet,
                    "prophet_anchored": _prophet_anchored, "xgboost": _xgboost}
VOL_FACTORIES = {"hist_vol": _hist_vol, "ewma": _ewma, "garch": _garch, "har": _har}
_FACTORIES = {**RETURN_FACTORIES, **VOL_FACTORIES}
MODEL_NAMES = tuple(RETURN_FACTORIES)
VOL_MODEL_NAMES = tuple(VOL_FACTORIES)
HORIZON_AWARE = {"xgboost", "har"}      # models that train one head per horizon


def make_model(name: str, **params) -> Forecaster:
    try:
        return _FACTORIES[name](**params)
    except KeyError:
        raise ValueError(f"unknown model '{name}'; choose from {tuple(_FACTORIES)}") from None


__all__ = ["DEFAULT_HORIZONS", "Forecaster", "NotFittedError", "log_prices", "make_model",
           "MODEL_NAMES", "VOL_MODEL_NAMES", "HORIZON_AWARE"]
