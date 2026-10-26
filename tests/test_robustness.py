import numpy as np
import pandas as pd
import pytest

from credibility.evaluation import get_task
from credibility.robustness import COVID, cell_frame, dm_size, skill_variants

from test_volatility import cfg as eval_cfg


def frame(n=400, model_noise=0.5, seed=0, task="return"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-06-03", periods=n)
    act = rng.normal(0, 1, n) if task == "return" else np.exp(rng.normal(0, 0.5, n)) * 1e-4
    base = dict(symbol="A", origin_idx=np.arange(n), horizon=5, origin_date=dates, target_date=dates + pd.offsets.BDay(5), actual=act)
    if task == "return":
        good = act + rng.normal(0, model_noise, n)
        return pd.concat([pd.DataFrame({**base, "model": "bm", "pred": 0.0}), pd.DataFrame({**base, "model": "m", "pred": good})])
    good = act * np.exp(rng.normal(0, model_noise, n))
    return pd.concat([pd.DataFrame({**base, "model": "bm", "pred": 1e-4}), pd.DataFrame({**base, "model": "m", "pred": good})])


def task_for(name):
    t = get_task(eval_cfg(), name)
    return type(t)(name, ("m",), "bm", (5,))


def test_identical_to_benchmark_has_zero_skill_everywhere():
    p = frame()
    p.loc[p.model == "m", "pred"] = 0.0
    v = skill_variants(p, task_for("return"), "m", 5)
    assert v["full"] == v["ex_covid"] == v["ex_top5"] == v["alt_loss"] == 0.0 and v["dates_won"] == 0.0


def test_informative_model_has_positive_skill_in_every_view_and_wins_most_dates():
    for t in ("return", "volatility"):
        v = skill_variants(frame(task=t, model_noise=0.5 if t == "return" else 0.1), task_for(t), "m", 5)
        assert v["full"] > 0 and v["ex_covid"] > 0 and v["ex_top5"] > 0 and v["alt_loss"] > 0 and v["dates_won"] > 0.6


def test_ex_covid_drops_exactly_the_window_and_ex_top5_drops_the_top_outcomes():
    p = frame(n=900)                                   # covers 2020-02..2020-06
    g = cell_frame(p, task_for("return"), "m", 5)
    in_window = g["origin_date"].between(*COVID)
    assert in_window.sum() > 50
    # corrupt only in-window rows: the ex-COVID skill must not move, the full skill must
    base = skill_variants(p, task_for("return"), "m", 5)
    p2 = p.copy()
    mask = (p2.model == "m") & p2["origin_date"].between(*COVID)
    p2.loc[mask, "pred"] = 50.0
    again = skill_variants(p2, task_for("return"), "m", 5)
    assert again["ex_covid"] == pytest.approx(base["ex_covid"]) and again["full"] < base["full"]
    # ex_top5 keeps 95% of rows: corrupting only the single largest outcome leaves ex_top5 unchanged
    p3 = p.copy()
    top = p3[p3.model == "m"]["actual"].idxmax()
    big = p3.loc[(p3.model == "m")].sort_values("actual").index[-1]
    p3.loc[big, "pred"] = -500.0
    assert skill_variants(p3, task_for("return"), "m", 5)["ex_top5"] == pytest.approx(base["ex_top5"])


def test_dm_size_is_nominal_for_iid_and_liberal_with_strong_overlap():
    iid = dm_size(100, 0, n_sims=1500, seed=3)
    assert 0.03 < iid < 0.075
    overlap = dm_size(100, 2, n_sims=1500, seed=3)
    assert overlap > iid                               # documented over-rejection under strong overlap


def test_realistic_overlap_size_depends_on_lag():
    from credibility.robustness import dm_size_overlap
    p0, se0 = dm_size_overlap(122, 10, 21, lag=0, n_sims=3000, seed=2)
    p2, se2 = dm_size_overlap(122, 10, 21, lag=2, n_sims=3000, seed=2)
    assert p0 > p2 + 0.02                                  # ignoring the overlap is clearly more liberal
    assert 0.03 < p2 < 0.09 and se2 < 0.01                 # the chosen lag is close to nominal
    p5, _ = dm_size_overlap(122, 10, 5, lag=0, n_sims=3000, seed=2)
    assert 0.03 < p5 < 0.075                               # no overlap at h=5: lag 0 is right
