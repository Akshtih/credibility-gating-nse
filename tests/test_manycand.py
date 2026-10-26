import numpy as np
import pandas as pd
import pytest

from credibility import manycand
from credibility.evaluation.tasks import Task
from credibility.manycand import (apply_rules, evaluate_split, load_protocol, return_candidates, rule_summary,
                                  volatility_candidates, wilson)

from test_volatility import vol_series


class FakeSnapshot:
    """Serves two synthetic series; prices after ``cut`` can be corrupted to prove candidates never look ahead."""
    corrupt_after = None

    @classmethod
    def open(cls, name):
        return cls()

    def load(self, sym):
        df = vol_series(n=900, seed=1 if sym == "A" else 2, storm_at_end=False)
        if FakeSnapshot.corrupt_after is not None:
            df = df.copy()
            df.loc[FakeSnapshot.corrupt_after + 1:, "close"] = np.random.default_rng(9).uniform(1, 1e6, len(df) - FakeSnapshot.corrupt_after - 1)
        return df


def key_frame(benchmark, horizons, origins=(780, 800, 820, 840)):
    rows = []
    for sym in ("A", "B"):
        for i in origins:
            for h in horizons:
                rows.append({"symbol": sym, "origin_idx": i, "origin_date": pd.Timestamp("2022-01-03") + pd.offsets.BDay(i - 780),
                             "target_date": pd.Timestamp("2022-01-03") + pd.offsets.BDay(i - 780 + h), "horizon": h, "actual": 0.001,
                             "model": benchmark, "pred": 0.0 if benchmark == "naive" else 1e-4})
    return pd.DataFrame(rows)


def with_stored(base, names, rng=None):
    rng = rng or np.random.default_rng(0)
    out = [base]
    for n in names:
        f = base.copy()
        f["model"], f["pred"] = n, rng.normal(1e-4, 2e-5, len(f)) if base["model"].iloc[0] == "hist_vol" else rng.normal(0, 0.01, len(f))
        out.append(f)
    return pd.concat(out, ignore_index=True)


@pytest.fixture(autouse=True)
def fake_snapshot(monkeypatch):
    FakeSnapshot.corrupt_after = None
    monkeypatch.setattr(manycand, "Snapshot", FakeSnapshot)
    yield
    FakeSnapshot.corrupt_after = None


def test_protocol_is_the_preregistered_one():
    p = load_protocol()
    assert p["alpha"] == 0.05 and p["splits"] == ["2021-07-01", "2022-01-01", "2022-07-01"]
    assert set(p["rules"]) == {"gate", "raw_dm", "simple", "best_only"}


def test_return_candidates_count_and_no_lookahead():
    spec = load_protocol()["candidates"]["return"]
    base = with_stored(key_frame("naive", (1, 5, 21)), ["arima", "xgboost", "prophet_anchored"])
    clean = return_candidates(base, "x", spec)
    n_expected = 5 + 4 * 2 + 2 * 1 + 3 * 4
    assert clean["model"].nunique() == n_expected
    assert clean["pred"].notna().all()
    FakeSnapshot.corrupt_after = 800                       # corrupt every price after row 800
    dirty = return_candidates(base, "x", spec)
    k = ["model", "symbol", "origin_idx", "horizon"]
    a = clean[clean.origin_idx <= 800].sort_values(k)["pred"].to_numpy()
    b = dirty[dirty.origin_idx <= 800].sort_values(k)["pred"].to_numpy()
    assert np.array_equal(a, b)
    assert not np.array_equal(clean[clean.origin_idx > 800].sort_values(k)["pred"].to_numpy(),
                              dirty[dirty.origin_idx > 800].sort_values(k)["pred"].to_numpy())   # the corruption is real


def test_shrunk_candidates_are_exact_multiples_of_stored_predictions():
    spec = load_protocol()["candidates"]["return"]
    base = with_stored(key_frame("naive", (5,)), ["arima", "xgboost", "prophet_anchored"])
    c = return_candidates(base, "x", spec)
    a = base[base.model == "arima"].set_index(["symbol", "origin_idx"])["pred"]
    s = c[c.model == "arima_x0.5"].set_index(["symbol", "origin_idx"])["pred"]
    assert np.allclose(s.reindex(a.index), 0.5 * a)


def test_volatility_candidates_count_no_lookahead_and_positive():
    spec = load_protocol()["candidates"]["volatility"]
    base = with_stored(key_frame("hist_vol", (5, 21)), ["ewma", "garch", "har"])
    clean = volatility_candidates(base, "x", spec, (5, 21))
    assert clean["model"].nunique() == 6 + 4 + 3 + 1 + 1 + 4
    assert clean["pred"].notna().all() and (clean["pred"] > 0).all()
    FakeSnapshot.corrupt_after = 800
    dirty = volatility_candidates(base, "x", spec, (5, 21))
    k = ["model", "symbol", "origin_idx", "horizon"]
    fitted = clean["model"].str.startswith(("ewma_0", "hist_"))
    a = clean[(clean.origin_idx <= 800) & fitted].sort_values(k)["pred"].to_numpy()
    b = dirty[(dirty.origin_idx <= 800) & dirty["model"].str.startswith(("ewma_0", "hist_"))].sort_values(k)["pred"].to_numpy()
    assert np.array_equal(a, b)


def test_rules_on_a_known_table():
    t = pd.DataFrame({"model": list("abcd"), "horizon": 5, "skill_S": [0.05, 0.02, -0.1, 0.03],
                      "p_S": [0.001, 0.04, 0.9, 0.2], "padj_S": [0.01, 0.2, 1.0, 0.6],
                      "skill_V": [0.04, -0.01, -0.2, 0.02], "padj_V": [0.02, 0.9, 1.0, 0.5]})
    t["holds"] = t["skill_V"] > 0
    t["holds_sig"] = t["holds"] & (t["padj_V"] < 0.05)
    r = apply_rules(t)
    assert list(r["gate"]) == [True, False, False, False]
    assert list(r["raw_dm"]) == [True, True, False, False]
    assert list(r["simple"]) == [True, True, False, True]
    assert list(r["best_only"]) == [True, False, False, False]
    s = rule_summary(t, r).set_index("rule")
    assert s.loc["gate", "n_fail"] == 0 and s.loc["simple", "n_fail"] == 1 and s.loc["raw_dm", "n_fail"] == 1
    assert s.loc["simple", "n_promoted"] == 3 and s.loc["gate", "n_hold_sig"] == 1
    assert wilson(0, 0)[0] != wilson(0, 0)[0]                        # NaN for empty


def test_null_world_gate_controls_false_promotion_while_simple_rule_does_not():
    res = manycand.null_check(reps=40, seed=0)
    assert res["gate_any_rate"] <= 0.15                   # family-wise error ~<= alpha (Monte-Carlo slack)
    assert 0.35 < res["simple_cell_rate"] < 0.65          # skill_S>0 happens by chance about half the time
    assert 0.35 < res["simple_fail_share"] < 0.65         # and those promotions fail on validation about half the time
    assert res["gate_cell_rate"] < 0.01
