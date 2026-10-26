import math

import numpy as np
import pandas as pd
import pytest

from credibility.config import EvalConfig, StatsConfig
from credibility.data import Snapshot
from credibility.evaluation import backtest as bt
from credibility.evaluation import (HoldoutLockedError, block_bootstrap_ci, dm_test, forecast_at_origin, holm,
                                    overlap_lag, run_backtest, summarize)
from credibility.evaluation.metrics import align_with_benchmark, summarize_group, wilson_interval

from test_models import synth


def make_cfg(tmp_path, **kw) -> EvalConfig:
    base = dict(snapshot="syn", models=("naive", "arima"), benchmark_model="naive", horizons=(1, 5),
                min_train_rows=300, origin_step=20, holdout_start="2024-06-01",
                results_dir=str(tmp_path / "results"), stats=StatsConfig(bootstrap_resamples=200))
    base.update(kw)
    cfg = EvalConfig(**base)
    return cfg


@pytest.fixture
def snap(tmp_path, monkeypatch):
    # snapshot lives under tmp; point the module's default base at it
    monkeypatch.setattr("credibility.config.SNAPSHOT_DIR", tmp_path / "snaps")
    s = Snapshot.create("syn", {})
    for k, sym in enumerate(["A.NS", "B.NS"]):
        s.add(sym, synth(n=900, seed=k, end="2024-12-31"), {"passed": True, "issues": []})
    return s


@pytest.fixture
def cfg_results(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    monkeypatch.setattr(EvalConfig, "results_path", property(lambda self: tmp_path / "results"))
    return cfg


# ---------------- leakage guards ----------------

def test_prediction_unchanged_when_future_rows_are_corrupted():
    df = synth(n=700, seed=3)
    i = 450
    clean = forecast_at_origin("arima", df, i, (1, 5))
    corrupted = df.copy()
    rng = np.random.default_rng(99)
    corrupted.loc[i + 1:, "close"] = rng.uniform(1, 1e6, len(df) - i - 1)  # garbage future
    assert forecast_at_origin("arima", corrupted, i, (1, 5)) == clean
    xg_clean = forecast_at_origin("xgboost", df, i, (1, 5))
    assert forecast_at_origin("xgboost", corrupted, i, (1, 5)) == xg_clean


def test_models_only_see_history_up_to_origin(monkeypatch):
    from credibility.models.naive import NaiveForecaster

    seen = []
    orig = NaiveForecaster._fit

    def spy(self, logp):
        seen.append(logp.index[-1])
        return orig(self, logp)

    monkeypatch.setattr(NaiveForecaster, "_fit", spy)
    df = synth(n=500)
    forecast_at_origin("naive", df, 300, (1,))
    assert seen == [df["date"].iloc[300]]


# ---------------- period logic (purging) ----------------

def test_is_valid_rules():
    hold = pd.Timestamp("2024-01-01")
    d = pd.Timestamp
    assert bt.is_valid("dev", d("2023-12-01"), d("2023-12-20"), hold)
    assert not bt.is_valid("dev", d("2023-12-20"), d("2024-01-05"), hold)   # purged: outcome in holdout
    assert not bt.is_valid("holdout", d("2023-12-29"), d("2024-01-05"), hold)
    assert bt.is_valid("holdout", d("2024-01-02"), d("2024-01-09"), hold)


def test_dev_and_holdout_are_disjoint_and_dev_is_purged(snap, cfg_results):
    cfg = cfg_results
    d = run_backtest(cfg, "dev", run_name="d", models=["naive"], workers=1, log=lambda m: None)
    dev = pd.read_parquet(d / "predictions.parquet")
    h = run_backtest(cfg, "holdout", run_name="h", models=["naive"], workers=1, confirm_holdout=True,
                     log=lambda m: None)
    hold = pd.read_parquet(h / "predictions.parquet")
    hs = pd.Timestamp(cfg.holdout_start)
    assert len(dev) and len(hold)
    assert (dev["target_date"] < hs).all()
    assert (hold["origin_date"] >= hs).all()
    key = ["symbol", "origin_idx", "horizon"]
    assert dev.merge(hold, on=key).empty


def test_holdout_runs_once_and_needs_confirmation(snap, cfg_results):
    cfg = cfg_results
    with pytest.raises(HoldoutLockedError):
        run_backtest(cfg, "holdout", run_name="x", models=["naive"], workers=1)
    run_backtest(cfg, "holdout", run_name="h1", models=["naive"], workers=1, confirm_holdout=True, log=lambda m: None)
    with pytest.raises(HoldoutLockedError):
        run_backtest(cfg, "holdout", run_name="h2", models=["naive"], workers=1, confirm_holdout=True,
                     log=lambda m: None)


def test_realised_values_and_grid(snap, cfg_results):
    cfg = cfg_results
    out = run_backtest(cfg, "dev", run_name="g", models=["naive"], workers=1, log=lambda m: None)
    p = pd.read_parquet(out / "predictions.parquet")
    df = snap.load("A.NS")
    r = p[(p.symbol == "A.NS") & (p.horizon == 5)].iloc[0]
    i = int(r.origin_idx)
    assert i == cfg.min_train_rows - 1
    assert math.isclose(r.actual, math.log(df.close.iloc[i + 5] / df.close.iloc[i]), rel_tol=1e-9)
    assert (p["pred"] == 0).all()
    assert set(p.origin_idx.unique()) <= set(range(cfg.min_train_rows - 1, 900, cfg.origin_step))


def test_backtest_is_deterministic(snap, cfg_results):
    cfg = cfg_results
    a = pd.read_parquet(run_backtest(cfg, "dev", run_name="a", models=["naive", "arima"], workers=1,
                                     max_origins=4, log=lambda m: None) / "predictions.parquet")
    b = pd.read_parquet(run_backtest(cfg, "dev", run_name="b", models=["naive", "arima"], workers=1,
                                     max_origins=4, log=lambda m: None) / "predictions.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_failures_are_recorded_not_fatal(snap, cfg_results, monkeypatch):
    def boom(name, df, i, horizons):
        raise RuntimeError("fit exploded")

    monkeypatch.setattr(bt, "forecast_full", boom)
    out = run_backtest(cfg_results, "dev", run_name="f", models=["naive"], workers=1, max_origins=2,
                       log=lambda m: None)
    f = pd.read_csv(out / "failures.csv")
    assert len(f) > 0 and f["error"].str.contains("exploded").all()


# ---------------- metrics ----------------

def test_metrics_known_values():
    preds = pd.DataFrame({
        "symbol": "A", "origin_idx": [0, 1, 2, 3], "horizon": 1,
        "model": ["m"] * 4, "pred": [0.1, -0.1, 0.1, 0.1], "actual": [0.1, -0.1, -0.1, 0.1],
    })
    bm = preds.assign(model="naive", pred=0.0)
    df = align_with_benchmark(pd.concat([preds, bm]), "naive")
    m = summarize_group(df)
    assert m["n"] == 4
    assert m["dir_acc"] == 0.75
    # model squared errors: [0,0,0.04,0]; benchmark: [0.01,0.01,0.01,0.01]
    assert math.isclose(m["r2_os"], 1 - 0.04 / 0.04)  # equal total squared error -> 0
    assert math.isclose(m["mae_ratio"], (0.2 / 4) / 0.1)


def test_wilson_interval_contains_p_and_is_bounded():
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


# ---------------- statistics ----------------

def test_overlap_lag():
    assert overlap_lag(1, 10) == 0 and overlap_lag(5, 10) == 0
    assert overlap_lag(21, 10) == 2 and overlap_lag(21, 5) == 4


def test_dm_better_model_gets_small_p_and_worse_gets_large():
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.2, 400)
    better = base - 0.3 + rng.normal(0, 0.05, 400)
    r = dm_test(better, base, lag=0)
    assert r["mean_diff"] < 0 and r["p_better"] < 0.001
    r2 = dm_test(base, better, lag=0)
    assert r2["p_better"] > 0.99


def test_dm_equal_losses_are_not_significant_and_type1_rate_is_sane():
    rng = np.random.default_rng(1)
    ps = []
    for _ in range(300):
        a = rng.normal(1, 0.3, 120)
        b = rng.normal(1, 0.3, 120)
        ps.append(dm_test(a, b, 0)["p_better"])
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.10   # nominal 5%


def test_dm_hac_widens_for_autocorrelated_differentials():
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 500)
    d = np.convolve(e, np.ones(5) / 5, mode="same")        # strongly autocorrelated, zero mean
    s0 = abs(dm_test(d, 0 * d, 0)["stat"])
    s4 = abs(dm_test(d, 0 * d, 4)["stat"])
    assert s4 < s0


def test_dm_degenerate_inputs_are_nan():
    assert math.isnan(dm_test([1, 1, 1], [1, 1, 1])["p_better"])          # too short
    assert math.isnan(dm_test(np.ones(50), np.ones(50))["p_better"])      # zero variance


def test_holm_known_example():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([0.01, np.nan])[1]) and holm([0.01, np.nan])[0] == 0.01


def test_block_bootstrap_ci_covers_mean_and_is_seeded():
    rng = np.random.default_rng(3)
    d = rng.normal(0.5, 1, 300)
    lo, hi = block_bootstrap_ci(d, block=5, n_boot=500, seed=1)
    assert lo < d.mean() < hi
    assert (lo, hi) == block_bootstrap_ci(d, block=5, n_boot=500, seed=1)


# ---------------- end to end report ----------------

def test_summarize_pipeline_runs(snap, cfg_results):
    cfg = cfg_results
    out = run_backtest(cfg, "dev", run_name="s", workers=1, log=lambda m: None)
    tbl = summarize(pd.read_parquet(out / "predictions.parquet"), cfg)
    assert set(tbl["model"]) == {"arima"} and set(tbl["horizon"]) == {1, 5}
    assert tbl["dm_p_adj"].between(0, 1).all()
    assert tbl["n_tickers"].eq(2).all()
