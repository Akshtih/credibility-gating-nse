import math

import numpy as np
import pandas as pd
import pytest

from credibility.evaluation import get_task
from credibility.uncertainty import (ConformalConfig, build_intervals, conformal_quantile, coverage_table,
                                     interval_score, nifty_regimes, series_intervals)

from test_volatility import cfg as eval_cfg

TASK = get_task(eval_cfg(), "return")


def make_preds(n=400, horizon=5, step=5, scale_fn=lambda k: 1.0, seed=0, symbol="A", model="m") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2018-01-01", periods=n * step + horizon + 5)
    idx = np.arange(0, n * step, step)
    res = np.array([rng.normal(0, scale_fn(k)) for k in range(n)])
    return pd.DataFrame({
        "symbol": symbol, "model": model, "origin_idx": idx, "horizon": horizon,
        "origin_date": cal[idx], "target_date": cal[idx + horizon],
        "pred": 0.0, "actual": res, "sigma_d": 1.0 / math.sqrt(horizon),
    })


def test_conformal_quantile_known():
    x = np.arange(1, 11, dtype=float)            # n=10
    assert conformal_quantile(x, 0.2) == 9.0     # ceil(11*0.8)=9th smallest
    assert conformal_quantile(x, 0.01) == 10.0   # rank > n is capped at the max
    assert math.isnan(conformal_quantile(np.array([]), 0.2))


def test_no_interval_during_calibration_warmup():
    p = make_preds(n=60)
    iv = build_intervals(p, TASK, ConformalConfig(min_cal=20))
    assert iv["split_lo"].isna().sum() >= 20
    assert iv["split_lo"].notna().sum() > 0
    first_valid = iv["split_lo"].notna().to_numpy().argmax()
    assert first_valid >= 20


def test_iid_residuals_give_nominal_coverage():
    iv = build_intervals(make_preds(n=1500, seed=1), TASK, ConformalConfig(cal_window=100))
    t = coverage_table(iv, TASK, by=("model",))
    for m in ("split", "aci"):
        cov = float(t[t["method"] == m]["coverage"].iloc[0])
        assert 0.76 < cov < 0.84, (m, cov)


def test_aci_recovers_after_variance_shift_better_than_split():
    # residual scale triples at k=300: split lags behind, ACI widens faster
    p = make_preds(n=700, scale_fn=lambda k: 1.0 if k < 300 else 3.0, seed=2)
    iv = build_intervals(p, TASK, ConformalConfig(cal_window=100, gamma=0.05))
    post = iv.iloc[300:500].dropna(subset=["split_lo"])
    cov = lambda m: float(((post["actual"] >= post[f"{m}_lo"]) & (post["actual"] <= post[f"{m}_hi"])).mean())
    assert cov("aci") > cov("split")


def test_intervals_never_use_outcomes_realised_after_the_origin():
    p = make_preds(n=300, horizon=21, step=10, seed=3)
    base = build_intervals(p, TASK)
    k = 150
    cut_date = p.iloc[k]["origin_date"]
    corrupted = p.copy()
    future = corrupted["target_date"] > cut_date                      # outcomes unknown at origin k
    corrupted.loc[future, "actual"] = 1e6
    again = build_intervals(corrupted, TASK)
    for col in ("split_lo", "split_hi", "aci_lo", "aci_hi"):
        a = base.loc[base["origin_date"] <= cut_date, col].to_numpy()
        b = again.loc[again["origin_date"] <= cut_date, col].to_numpy()
        # the forecast at an origin may only depend on outcomes with target_date <= that origin
        assert len(a) == k + 1
        assert np.allclose(a, b, equal_nan=True)                      # exact equality for every origin <= cut


def test_series_intervals_use_only_realised_outcomes_overlapping_horizon():
    n, h, step = 60, 21, 10
    cal = pd.bdate_range("2018-01-01", periods=n * step + h + 5)
    idx = np.arange(0, n * step, step)
    o, t = cal[idx].to_numpy(), cal[idx + h].to_numpy()
    res = np.arange(1, n + 1, dtype=float)
    qs, qa, _ = series_intervals(o, t, res, ConformalConfig(min_cal=5, cal_window=1000))
    k = 40
    m = int(np.searchsorted(t, o[k], side="right"))
    assert m <= k - 2                                                 # last 2 origins not realised yet
    assert qs[k] == conformal_quantile(res[:m], 0.2)


def test_gauss_and_native_columns():
    p = make_preds(n=100)
    p["lo"], p["hi"] = -1.0, 1.0
    iv = build_intervals(p, TASK)
    assert {"gauss_lo", "gauss_hi", "native_lo", "native_hi"} <= set(iv.columns)
    assert np.allclose(iv["gauss_hi"] - iv["gauss_lo"], 2 * 1.2815515655446004 * 1.0, atol=1e-6)


def test_volatility_intervals_are_positive_and_ordered():
    vt = get_task(eval_cfg(), "volatility")
    p = make_preds(n=120)
    p["pred"] = 1e-4
    p["actual"] = np.exp(np.random.default_rng(0).normal(np.log(1e-4), 0.3, len(p)))
    iv = build_intervals(p, vt).dropna(subset=["split_lo"])
    assert (iv["split_lo"] > 0).all() and (iv["split_lo"] < iv["pred"]).all() and (iv["pred"] < iv["split_hi"]).all()


def test_interval_score_known_values():
    assert interval_score(0.5, 0.0, 1.0, 0.2) == pytest.approx(1.0)
    assert interval_score(2.0, 0.0, 1.0, 0.2) == pytest.approx(1.0 + 10 * 1.0)
    assert interval_score(-1.0, 0.0, 1.0, 0.2) == pytest.approx(1.0 + 10 * 1.0)


def test_nifty_regimes_are_causal():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=900)
    vol = np.where(np.arange(900) < 600, 0.005, 0.03)
    close = 100 * np.exp(np.cumsum(rng.normal(0, vol)))
    px = pd.DataFrame({"date": dates, "close": close})
    full = nifty_regimes(px, pd.Series(dates[300:850]))
    part = nifty_regimes(px.iloc[:700], pd.Series(dates[300:650]))
    assert (full.iloc[:350].to_numpy() == part.to_numpy()).all() or \
        (pd.Series(full.iloc[:350]).fillna("x").to_numpy() == pd.Series(part).fillna("x").to_numpy()).all()
    assert (full.iloc[-100:] == "high").mean() > 0.8
