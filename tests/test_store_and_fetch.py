import pandas as pd
import pytest

from credibility.config import DataConfig, Instrument, QualityConfig
from credibility.data import fetch as fetch_mod
from credibility.data import pipeline
from credibility.data.fetch import FetchError, fetch_prices, normalise
from credibility.data.store import Snapshot, SnapshotError, content_hash

from test_quality import make_prices


def test_content_hash_is_stable_and_sensitive():
    df = make_prices()
    assert content_hash(df) == content_hash(df.copy())
    df2 = df.copy()
    df2.loc[3, "close"] += 0.01
    assert content_hash(df) != content_hash(df2)


def test_snapshot_roundtrip_and_tamper_detection(tmp_path):
    snap = Snapshot.create("t", {"a": 1}, base=tmp_path)
    df = make_prices()
    snap.add("X.NS", df, {"passed": True, "issues": []})
    snap2 = Snapshot.open("t", base=tmp_path)
    out = snap2.load("X.NS")
    pd.testing.assert_frame_equal(out, df, check_dtype=False)
    assert snap2.symbols() == ["X.NS"]

    # tamper with the stored file
    bad = df.copy()
    bad.loc[0, "close"] = 1.0
    bad.to_parquet(snap2._file("X.NS"), index=False)
    with pytest.raises(SnapshotError):
        snap2.load("X.NS")
    assert snap2.verify_all() == {"X.NS": False}


def test_snapshots_are_immutable(tmp_path):
    Snapshot.create("t", {}, base=tmp_path)
    with pytest.raises(SnapshotError):
        Snapshot.create("t", {}, base=tmp_path)
    Snapshot.create("t", {}, base=tmp_path, overwrite=True)


def test_symbols_filters_failed_quality(tmp_path):
    snap = Snapshot.create("t", {}, base=tmp_path)
    snap.add("GOOD.NS", make_prices(), {"passed": True, "issues": []})
    snap.add("BAD.NS", make_prices(), {"passed": False, "issues": []})
    assert snap.symbols() == ["GOOD.NS"]
    assert set(snap.symbols(only_passed=False)) == {"GOOD.NS", "BAD.NS"}


def _raw_yahoo(n=5):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata", name="Date")
    return pd.DataFrame(
        {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100}, index=idx
    )


def test_normalise_schema_and_tz():
    df = normalise(_raw_yahoo())
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["date"].dt.tz is None
    assert normalise(pd.DataFrame()).empty


def test_fetch_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake(symbol, start, end, adj):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return _raw_yahoo()

    monkeypatch.setattr(fetch_mod, "_history", fake)
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    assert len(fetch_prices("X.NS", "2024-01-01", "2024-01-31")) == 5
    assert calls["n"] == 3


def test_fetch_raises_after_retries(monkeypatch):
    monkeypatch.setattr(fetch_mod, "_history", lambda *a: pd.DataFrame())
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    with pytest.raises(FetchError):
        fetch_prices("X.NS", "2024-01-01", "2024-01-31", retries=2)


def test_build_snapshot_end_to_end(tmp_path, monkeypatch):
    prices = make_prices(n=300, end="2024-12-30")

    def fake_fetch(symbol, start, as_of, auto_adjust=True, **kw):
        if symbol == "MISSING.NS":
            raise FetchError("nope")
        return prices

    monkeypatch.setattr(pipeline, "fetch_prices", fake_fetch)
    cfg = DataConfig(
        start_date="2023-01-01", as_of_date="2024-12-31", auto_adjust=True,
        snapshot_name="e2e", benchmark="^NSEI",
        universe=(Instrument("A.NS", "x"), Instrument("MISSING.NS", "x")),
        quality=QualityConfig(min_rows=100),
    )
    snap = pipeline.build_snapshot(cfg, base=tmp_path, log=lambda m: None)
    assert set(snap.symbols()) == {"A.NS", "^NSEI"}
    assert "MISSING.NS" in snap.manifest["failed"]
    assert all(Snapshot.open("e2e", base=tmp_path).verify_all().values())
