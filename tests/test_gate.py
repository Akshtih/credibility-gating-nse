import numpy as np
import pandas as pd
import pytest

from credibility.evaluation import get_task, run_backtest
from credibility.gate import GateConfig, Registry, ServeError, build_gate, champions, decide, serve_forecast, ticker_verdicts
from credibility.uncertainty import ConformalConfig

from test_evaluation import cfg_results, make_cfg, snap  # noqa: F401  (fixtures)
from test_volatility import cfg as vol_cfg

TASK = get_task(vol_cfg(), "return")


def summary_rows(rows):
    return pd.DataFrame([{"model": m, "horizon": h, "skill": s, "dm_p_adj": p, "n_origins": n}
                         for m, h, s, p, n in rows])


def cov_rows(rows):
    return pd.DataFrame([{"model": m, "horizon": h, "method": "aci", "coverage": c} for m, h, c in rows])


def test_decide_checks_each_condition_separately():
    s = summary_rows([("good", 5, 0.05, 0.001, 100), ("noskill", 5, -0.02, 1.0, 100),
                      ("miscal", 5, 0.05, 0.001, 100), ("short", 5, 0.05, 0.001, 20),
                      ("notsig", 5, 0.01, 0.4, 100)])
    c = cov_rows([("good", 5, 0.81), ("noskill", 5, 0.8), ("miscal", 5, 0.5), ("short", 5, 0.8), ("notsig", 5, 0.8)])
    d = decide(s, c, TASK, GateConfig()).set_index("model")
    assert d.loc["good", "promote"]
    assert not d.loc["noskill", "promote"] and not d.loc["noskill", "beats_benchmark"]
    assert d.loc["miscal", "beats_benchmark"] and not d.loc["miscal", "calibrated"] and not d.loc["miscal", "promote"]
    assert not d.loc["short", "promote"] and not d.loc["short", "enough_evidence"]
    assert not d.loc["notsig", "promote"]
    assert "miscalibrated" in d.loc["miscal", "reason"]


def test_champion_is_highest_skill_promoted_else_fallback():
    s = summary_rows([("a", 5, 0.03, 0.001, 100), ("b", 5, 0.07, 0.001, 100), ("a", 21, -0.1, 1.0, 100)])
    c = cov_rows([("a", 5, 0.8), ("b", 5, 0.8), ("a", 21, 0.8)])
    task = type(TASK)("return", ("naive", "a", "b"), "naive", (5, 21))
    ch = champions(decide(s, c, task), task)
    assert ch[5]["model"] == "b" and ch[5]["role"] == "champion"
    assert ch[21] == {"model": "naive", "role": "fallback", "skill": 0.0}


def test_ticker_verdicts_distinguish_good_neutral_and_bad_tickers():
    rng = np.random.default_rng(0)
    n = 150
    rows = []
    for sym, noise_model in (("GOOD", 0.2), ("BAD", 2.0)):
        actual = rng.normal(0, 1, n)
        for model, pred in (("naive", np.zeros(n)), ("m", actual + rng.normal(0, noise_model, n))):
            rows.append(pd.DataFrame({"symbol": sym, "model": model, "origin_idx": np.arange(n), "horizon": 1,
                                      "origin_date": pd.bdate_range("2020-01-01", periods=n),
                                      "target_date": pd.bdate_range("2020-01-02", periods=n),
                                      "pred": pred, "actual": actual}))
    cfg = make_cfg(__import__("pathlib").Path("."))
    v = ticker_verdicts(pd.concat(rows), cfg, TASK).set_index("symbol")
    assert v.loc["GOOD", "verdict"] == "TRUSTED"
    assert v.loc["BAD", "verdict"] == "NOT_BETTER"


def test_registry_is_content_addressed_and_serves_latest(tmp_path):
    reg = Registry(tmp_path / "reg")
    e = {"task": "return", "horizon": 5, "model": "naive", "role": "fallback", "params": {}, "metrics": {}, "reason": "x"}
    a = reg.register(e, serve=True)
    b = reg.register(e, serve=False)
    assert a == b
    assert reg.served("return", 5)["model"] == "naive"
    assert reg.served("return", 21) is None
    e2 = {**e, "model": "arima", "role": "champion"}
    c = reg.register(e2, serve=True)
    assert c != a and reg.served("return", 5)["model"] == "arima"
    assert len(reg.entries()) == 2
    assert reg.get(a)["git_commit"]


def test_build_gate_and_serve_end_to_end(snap, cfg_results, tmp_path):  # noqa: F811
    cfg = cfg_results
    run = run_backtest(cfg, "dev", run_name="dev_e2e", workers=1, log=lambda m: None)
    reg_root = tmp_path / "registry"
    decisions = build_gate(cfg, "return", run, reg_root, GateConfig(), ConformalConfig(min_cal=10), log=lambda m: None)
    assert set(decisions["model"]) == {"arima"} and not decisions["promote"].any()    # random walk: nothing promoted
    reg = Registry(reg_root)
    for h in (1, 5):
        e = reg.served("return", h)
        assert e["model"] == "naive" and e["role"] == "fallback"
        assert e["snapshot_manifest_sha256"] and e["config_sha256"]
    assert (reg_root / "evidence" / "return" / "decisions.csv").exists()
    out = serve_forecast("A.NS", "return", 5, cfg, reg, conf=ConformalConfig(min_cal=10))
    assert out["verdict"] == "BENCHMARK_ONLY" and out["served_model"] == "naive"
    assert out["forecast"] == 0.0 and out["forecast_price"] == pytest.approx(out["last_close"])
    lo, hi = out["interval_80"]
    assert lo is not None and lo < 0.0 < hi
    assert out["interval_price_80"][0] < out["last_close"] < out["interval_price_80"][1]
    with pytest.raises(ServeError):
        serve_forecast("ZZZ.NS", "return", 5, cfg, reg)
    with pytest.raises(ServeError):
        serve_forecast("A.NS", "return", 99, cfg, reg)


def test_serve_respects_as_of_and_does_not_peek_at_future(snap, cfg_results, tmp_path):  # noqa: F811
    cfg = cfg_results
    run = run_backtest(cfg, "dev", run_name="dev_asof", workers=1, log=lambda m: None)
    build_gate(cfg, "return", run, tmp_path / "r", GateConfig(), ConformalConfig(min_cal=10), log=lambda m: None)
    reg = Registry(tmp_path / "r")
    early = serve_forecast("A.NS", "return", 5, cfg, reg, as_of="2023-06-30", conf=ConformalConfig(min_cal=10))
    assert early["as_of"] <= "2023-06-30"
    late = serve_forecast("A.NS", "return", 5, cfg, reg, conf=ConformalConfig(min_cal=10))
    assert late["as_of"] > early["as_of"]
