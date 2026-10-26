import numpy as np
import pandas as pd
import pytest

from credibility.config import SNAPSHOT_DIR
from credibility.data import Snapshot
from credibility.models import MODEL_NAMES, NotFittedError, make_model
from credibility.models.xgboost_model import build_features, build_targets

HORIZONS = (1, 5, 21)


def synth(n=600, drift=0.0, vol=0.01, seed=0, end="2024-12-31") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n)
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, n)))
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close,
                         "close": close, "volume": 1000})


# ---------- contract shared by every model ----------

@pytest.mark.parametrize("name", MODEL_NAMES)
def test_predict_before_fit_raises(name):
    with pytest.raises(NotFittedError):
        make_model(name).predict(5)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_predict_finite_and_does_not_mutate_input(name):
    df = synth()
    before = df.copy(deep=True)
    m = make_model(name).fit(df)
    pd.testing.assert_frame_equal(df, before)
    for h in HORIZONS:
        v = m.predict(h)
        assert isinstance(v, float) and np.isfinite(v) and abs(v) < 1.0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_deterministic(name):
    df = synth()
    a = make_model(name).fit(df).predict_many(HORIZONS)
    b = make_model(name).fit(df).predict_many(HORIZONS)
    assert a == b


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_invalid_horizon(name):
    m = make_model(name).fit(synth())
    with pytest.raises(ValueError):
        m.predict(0)


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("lstm")


# ---------- input validation ----------

@pytest.mark.parametrize("bad", ["unsorted", "nan", "nonpositive", "dupe"])
def test_rejects_bad_history(bad):
    df = synth()
    if bad == "unsorted":
        df = df.iloc[::-1]
    elif bad == "nan":
        df.loc[5, "close"] = np.nan
    elif bad == "nonpositive":
        df.loc[5, "close"] = 0.0
    else:
        df = pd.concat([df, df.iloc[[3]]], ignore_index=True).sort_values("date")
    with pytest.raises(ValueError):
        make_model("naive").fit(df)


# ---------- model behaviour ----------

def test_naive_is_zero():
    m = make_model("naive").fit(synth())
    assert all(m.predict(h) == 0.0 for h in HORIZONS)


@pytest.mark.parametrize("name", ["arima", "prophet", "prophet_anchored", "xgboost"])
def test_picks_up_a_strong_trend(name):
    up = make_model(name).fit(synth(drift=0.002, vol=0.003)).predict(21)
    down = make_model(name).fit(synth(drift=-0.002, vol=0.003)).predict(21)
    assert up > 0.0 > down


def test_arima_and_prophet_horizon_cap():
    for name in ("arima", "prophet", "prophet_anchored"):
        m = make_model(name).fit(synth())
        with pytest.raises(ValueError):
            m.predict(500)


def test_xgboost_untrained_horizon_and_short_history():
    m = make_model("xgboost").fit(synth())
    with pytest.raises(ValueError):
        m.predict(7)
    with pytest.raises(ValueError):
        make_model("xgboost").fit(synth(n=60))


# ---------- leakage guards for the feature pipeline ----------

def test_features_are_causal():
    logp = np.log(synth(n=400)["close"])
    logp.index = pd.bdate_range(end="2024-12-31", periods=400)
    full = build_features(logp)
    for n in (150, 250, 399):
        pd.testing.assert_frame_equal(build_features(logp.iloc[:n]), full.iloc[:n])


def test_targets_align_and_last_rows_are_nan():
    logp = pd.Series(np.arange(100, dtype=float))
    for h in (1, 5, 21):
        y = build_targets(logp, h)
        assert y.iloc[: 100 - h].eq(float(h)).all()
        assert y.iloc[100 - h:].isna().all()


def test_xgboost_ignores_rows_without_a_realised_target(monkeypatch):
    """Training must only use targets fully inside the history (no future data)."""
    import credibility.models.xgboost_model as xm

    seen = {}
    real = xm.build_targets

    def spy(logp, h):
        y = real(logp, h)
        seen[h] = (len(logp), int(y.notna().sum()))
        return y

    monkeypatch.setattr(xm, "build_targets", spy)
    make_model("xgboost").fit(synth(n=500))
    for h in HORIZONS:
        n, valid = seen[h]
        assert valid == n - h


# ---------- real data smoke test (skipped if snapshot absent) ----------

@pytest.mark.skipif(not (SNAPSHOT_DIR / "nse_v1" / "manifest.json").exists(), reason="snapshot not built")
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_real_data_smoke(name):
    df = Snapshot.open("nse_v1").load("RELIANCE.NS")
    hist = df[df["date"] <= "2024-12-31"]
    m = make_model(name).fit(hist)
    for h in HORIZONS:
        assert abs(m.predict(h)) < 0.5
