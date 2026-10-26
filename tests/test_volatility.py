import numpy as np
import pandas as pd
import pytest

from credibility.config import EvalConfig, StatsConfig
from credibility.evaluation import get_task
from credibility.evaluation.backtest import forecast_at_origin
from credibility.models import VOL_MODEL_NAMES, NotFittedError, make_model
from credibility.models.volatility import har_features


def vol_series(n=900, calm=0.008, storm=0.03, storm_len=40, seed=0, storm_at_end=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sig = np.full(n, calm)
    if storm_at_end:
        sig[-storm_len:] = storm
    close = 100 * np.exp(np.cumsum(rng.normal(0, sig)))
    return pd.DataFrame({"date": pd.bdate_range(end="2024-12-31", periods=n), "close": close,
                         "open": close, "high": close, "low": close, "volume": 1})


def cfg(tmp_path=None) -> EvalConfig:
    return EvalConfig(snapshot="x", models=("naive",), benchmark_model="naive", horizons=(1, 5),
                      min_train_rows=300, origin_step=20, holdout_start="2024-06-01", results_dir="r",
                      stats=StatsConfig())


@pytest.mark.parametrize("name", VOL_MODEL_NAMES)
def test_contract(name):
    df = vol_series()
    before = df.copy(deep=True)
    with pytest.raises(NotFittedError):
        make_model(name).predict(5)
    m = make_model(name, **({"horizons": (5, 21)} if name == "har" else {})).fit(df)
    pd.testing.assert_frame_equal(df, before)
    for h in (5, 21):
        v = m.predict(h)
        assert np.isfinite(v) and 0 < v < 0.05          # daily variance, sane scale
    again = make_model(name, **({"horizons": (5, 21)} if name == "har" else {})).fit(df)
    assert m.predict(5) == again.predict(5)


@pytest.mark.parametrize("name", ["hist_vol", "ewma", "garch", "har"])
def test_variance_forecast_rises_after_a_volatility_spike(name):
    kw = {"horizons": (5, 21)} if name == "har" else {}
    calm = make_model(name, **kw).fit(vol_series(storm_at_end=False)).predict(5)
    spike = make_model(name, **kw).fit(vol_series(storm_at_end=True)).predict(5)
    assert spike > 1.5 * calm


def test_hist_vol_is_flat_and_matches_definition():
    df = vol_series(storm_at_end=False)
    r = np.diff(np.log(df["close"]))[-21:]
    m = make_model("hist_vol").fit(df)
    assert m.predict(5) == m.predict(21) == pytest.approx(float(np.mean(r ** 2)))


def test_har_untrained_horizon_and_short_history():
    m = make_model("har", horizons=(5,)).fit(vol_series())
    with pytest.raises(ValueError):
        m.predict(21)
    with pytest.raises(ValueError):
        make_model("har").fit(vol_series(n=100))


def test_har_features_are_causal():
    r2 = pd.Series(np.random.default_rng(0).uniform(0, 1, 300))
    full = har_features(r2)
    pd.testing.assert_frame_equal(har_features(r2.iloc[:200]), full.iloc[:200])


def test_vol_model_prediction_ignores_future_rows():
    df = vol_series(n=700)
    i = 450
    for name in ("ewma", "garch", "har"):
        kw = {}
        clean = forecast_at_origin(name, df, i, (5, 21))
        bad = df.copy()
        bad.loc[i + 1:, "close"] = np.random.default_rng(1).uniform(1, 1e5, len(df) - i - 1)
        assert forecast_at_origin(name, bad, i, (5, 21)) == clean


# ---------------- task definitions ----------------

def test_volatility_task_realised_and_loss():
    t = get_task(cfg(), "volatility")
    logc = np.log(np.array([100.0, 101.0, 100.0, 102.0, 101.0, 103.0]))
    r2 = np.diff(logc) ** 2
    assert t.realised(logc, 0, 3) == pytest.approx(float(r2[:3].mean()))
    assert t.realised(logc, 1, 4) == pytest.approx(float(r2[1:5].mean()))
    p = np.array([1.0, 2.0, 0.5]); a = np.array([1.0, 1.0, 1.0])
    loss = t.loss(p, a)
    assert loss[0] == pytest.approx(0.0, abs=1e-12) and (loss >= 0).all()
    assert loss[1] > 0 and loss[2] > 0


def test_return_task_realised_and_loss():
    t = get_task(cfg(), "return")
    logc = np.log(np.array([100.0, 110.0, 99.0]))
    assert t.realised(logc, 0, 2) == pytest.approx(np.log(0.99))
    assert t.loss([0.1], [0.3])[0] == pytest.approx(0.04)


def test_link_roundtrip():
    for name in ("return", "volatility"):
        t = get_task(cfg(), name)
        x = np.array([0.001, 0.02])
        assert np.allclose(t.inverse_link(t.link(x)), x)


def test_unknown_task():
    with pytest.raises(ValueError):
        get_task(cfg(), "sentiment")
