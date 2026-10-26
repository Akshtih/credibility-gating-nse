import numpy as np
import pandas as pd
import pytest

from credibility.config import QualityConfig
from credibility.data.quality import validate

CFG = QualityConfig(min_rows=100)
AS_OF = "2024-12-31"


def make_prices(n=300, end="2024-12-30", seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.integers(1_000, 10_000, n),
    })


def checks(report):
    return {(i.check, i.severity) for i in report.issues}


def test_clean_data_passes():
    r = validate(make_prices(), "X.NS", CFG, AS_OF)
    assert r.passed and not r.issues


def test_too_short_fails():
    r = validate(make_prices(n=50), "X.NS", CFG, AS_OF)
    assert not r.passed and ("length", "error") in checks(r)


def test_missing_values_fail_when_over_threshold():
    df = make_prices()
    df.loc[df.sample(frac=0.2, random_state=1).index, "close"] = np.nan
    r = validate(df, "X.NS", CFG, AS_OF)
    assert not r.passed and ("missing", "error") in checks(r)


def test_duplicate_dates_fail():
    df = make_prices()
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    r = validate(df, "X.NS", CFG, AS_OF)
    assert ("duplicate_dates", "error") in checks(r)


def test_negative_price_fails():
    df = make_prices()
    df.loc[10, "close"] = -1.0
    assert ("non_positive", "error") in checks(validate(df, "X.NS", CFG, AS_OF))


def test_systematic_ohlc_violation_fails_rare_one_warns():
    df = make_prices()
    df.loc[10, "high"] = df.loc[10, "low"] - 1
    assert ("ohlc_integrity", "warning") in checks(validate(df, "X.NS", CFG, AS_OF))
    df2 = make_prices()
    df2.loc[:20, "high"] = df2.loc[:20, "low"] - 1
    assert ("ohlc_integrity", "error") in checks(validate(df2, "X.NS", CFG, AS_OF))


def test_stale_series_fails():
    r = validate(make_prices(end="2024-06-01"), "X.NS", CFG, AS_OF)
    assert ("staleness", "error") in checks(r)


def test_extreme_return_warns_not_fails():
    df = make_prices()
    df.loc[150:, ["open", "high", "low", "close"]] *= 3  # un-adjusted split-like jump
    r = validate(df, "X.NS", CFG, AS_OF)
    assert r.passed and ("extreme_returns", "warning") in checks(r)


def test_calendar_gap_warns():
    df = make_prices()
    df = df.drop(index=range(100, 115)).reset_index(drop=True)
    assert ("calendar_gaps", "warning") in checks(validate(df, "X.NS", CFG, AS_OF))


def test_missing_column_is_schema_error():
    r = validate(make_prices().drop(columns=["volume"]), "X.NS", CFG, AS_OF)
    assert not r.passed and ("schema", "error") in checks(r)
