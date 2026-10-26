import pytest

from credibility.app import _clean, create_app
from credibility.evaluation import run_backtest
from credibility.gate import GateConfig, build_gate
from credibility.uncertainty import ConformalConfig

from test_evaluation import cfg_results, snap  # noqa: F401  (fixtures)


@pytest.fixture
def client(snap, cfg_results, tmp_path, monkeypatch):  # noqa: F811
    run = run_backtest(cfg_results, "dev", run_name="dev_app", workers=1, log=lambda m: None)
    reg_root = tmp_path / "registry"
    build_gate(cfg_results, "return", run, reg_root, GateConfig(), ConformalConfig(min_cal=10), log=lambda m: None)
    import credibility.app as appmod
    monkeypatch.setattr(appmod, "load_eval_config", lambda path=None: cfg_results)
    app = create_app(registry_root=reg_root)
    return app.test_client()


def test_clean_replaces_nan_and_inf():
    assert _clean({"a": float("nan"), "b": [float("inf"), 1.0]}) == {"a": None, "b": [None, 1.0]}


def test_index_and_health(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Forecast Credibility" in r.data
    h = client.get("/health").get_json()
    assert h["status"] == "ok" and h["registry_entries"] > 0 and "return" in h["served"]


def test_universe_and_history(client):
    u = client.get("/api/universe").get_json()
    assert {s["symbol"] for s in u["symbols"]} == {"A.NS", "B.NS"} and u["tasks"]["return"] == [1, 5]
    h = client.get("/api/history?symbol=A.NS&n=50").get_json()
    assert len(h["close"]) == 50 and len(h["dates"]) == 50
    assert client.get("/api/history?symbol=NOPE").status_code == 404


def test_forecast_endpoint_serves_fallback_with_verdict(client):
    r = client.get("/api/forecast?symbol=A.NS&task=return&horizon=5")
    assert r.status_code == 200
    j = r.get_json()
    assert j["verdict"] == "BENCHMARK_ONLY" and j["served_model"] == "naive"
    assert j["forecast_price"] == pytest.approx(j["last_close"])
    assert j["evidence"]["reason"]


def test_forecast_endpoint_errors(client):
    assert client.get("/api/forecast?symbol=ZZ.NS&task=return&horizon=5").status_code == 404
    assert client.get("/api/forecast?symbol=A.NS&task=sentiment&horizon=5").status_code == 400
    assert client.get("/api/forecast?symbol=A.NS&task=return&horizon=abc").status_code == 400
    assert client.get("/api/forecast?symbol=A.NS&task=volatility&horizon=5").status_code == 404   # nothing registered


def test_evidence_endpoint(client):
    j = client.get("/api/evidence?task=return").get_json()
    assert j["decisions"] and {"model", "horizon", "promote", "reason"} <= set(j["decisions"][0])
    assert client.get("/api/evidence?task=volatility").status_code == 404
