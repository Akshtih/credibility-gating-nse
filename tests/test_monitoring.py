import numpy as np
import pandas as pd

from credibility.evaluation import get_task
from credibility.monitoring import (PageHinkley, RollingRatioMonitor, first_alarm_dates, ks_data_drift,
                                    policy_table, relative_differential, replay)

from test_volatility import cfg as vol_cfg

TASK = get_task(vol_cfg(), "return")


def test_page_hinkley_quiet_on_stationary_noise_and_fires_after_a_shift():
    rng = np.random.default_rng(0)
    quiet = PageHinkley()
    assert not any(quiet.update(x) for x in rng.normal(0, 0.2, 400))
    ph = PageHinkley()
    for x in rng.normal(0, 0.2, 100):
        assert not ph.update(x)
    fired = [i for i, x in enumerate(rng.normal(0.6, 0.2, 100)) if ph.update(x)]
    assert fired and fired[0] < 15                      # detects within ~15 observations


def test_rolling_ratio_monitor_hysteresis():
    m = RollingRatioMonitor(k=5, theta=0.1)
    for _ in range(5):
        assert not m.update(1.0, 1.0)                   # equal: no alarm
    for _ in range(5):
        m.update(2.0, 1.0)                              # candidate twice as bad
    assert m.alarm
    for _ in range(5):
        m.update(0.5, 1.0)                              # candidate recovers
    assert not m.alarm


def test_relative_differential_bounds_and_sign():
    assert relative_differential(2.0, 1.0) > 0 > relative_differential(1.0, 2.0)
    assert relative_differential(0.0, 0.0) == 0.0
    assert -1.0 <= relative_differential(1e9, 1e-9) <= 1.0


def test_ks_data_drift_detects_variance_change():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 0.01, 252)
    assert not ks_data_drift(ref, rng.normal(0, 0.01, 63)).drift
    assert ks_data_drift(ref, rng.normal(0, 0.04, 63)).drift
    assert not ks_data_drift(ref[:10], ref[:5]).drift


def make_series(n=300, switch=None, symbol="S", horizon=5, step=5):
    cal = pd.bdate_range("2019-01-01", periods=n * step + 30)
    idx = np.arange(0, n * step, step)
    rng = np.random.default_rng(0)
    actual = rng.normal(0, 1, n)
    good = actual + rng.normal(0, 0.2, n)                # accurate candidate
    bad = actual + rng.normal(0, 3.0, n)                 # useless candidate
    cand = good if switch is None else np.where(np.arange(n) < switch, good, bad)
    base = dict(symbol=symbol, origin_idx=idx, horizon=horizon, origin_date=cal[idx], target_date=cal[idx + horizon],
                actual=actual)
    return pd.concat([pd.DataFrame({**base, "model": "naive", "pred": 0.0}),
                      pd.DataFrame({**base, "model": "cand", "pred": cand})], ignore_index=True)


def test_guard_keeps_a_good_candidate_and_drops_a_failing_one():
    good = policy_table(replay(make_series(), TASK, "cand", 5)).set_index("policy")
    assert good.loc["candidate", "skill_all"] > 0.5
    assert good.loc["guarded_rolling", "skill_all"] > 0.9 * good.loc["candidate", "skill_all"]
    assert good.loc["guarded_rolling", "candidate_share"] > 0.9

    rep = replay(make_series(switch=150), TASK, "cand", 5)
    t = policy_table(rep).set_index("policy")
    assert t.loc["candidate", "skill_all"] < t.loc["guarded_rolling", "skill_all"]
    assert t.loc["guarded_ph", "skill_all"] > t.loc["candidate", "skill_all"]
    assert t.loc["benchmark", "skill_all"] == 0.0
    assert 0.2 < t.loc["guarded_rolling", "candidate_share"] < 0.8


def test_guard_decision_does_not_use_unrealised_outcomes():
    p = make_series(switch=None, n=200)
    base = replay(p, TASK, "cand", 5)
    k = 120
    cut = base.sort_values("origin_date").iloc[k]["origin_date"]
    bad = p.copy()
    bad.loc[(bad["model"] == "cand") & (bad["target_date"] > cut), "pred"] = 1e6   # unrealised at the cut origin
    again = replay(bad, TASK, "cand", 5)
    a = base[base["origin_date"] <= cut].sort_values("origin_date")
    b = again[again["origin_date"] <= cut].sort_values("origin_date")
    assert (a["cand_on_rolling"].to_numpy() == b["cand_on_rolling"].to_numpy()).all()
    assert (a["cand_on_ph"].to_numpy() == b["cand_on_ph"].to_numpy()).all()


def test_first_alarm_dates():
    rep = replay(make_series(switch=150), TASK, "cand", 5)
    d = first_alarm_dates(rep, ("2019-01-01", "2030-01-01"))
    assert len(d) == 1 and d.iloc[0] > rep["origin_date"].sort_values().iloc[150]
