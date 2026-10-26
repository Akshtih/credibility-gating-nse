"""Yahoo Finance fetcher. Returns a clean, timezone-naive, lowercase OHLCV frame."""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume"]


class FetchError(RuntimeError):
    pass


def _history(symbol: str, start: str, end_exclusive: str, auto_adjust: bool) -> pd.DataFrame:
    """Thin wrapper over yfinance so tests can patch it without importing yfinance."""
    import yfinance as yf

    return yf.Ticker(symbol).history(
        start=start, end=end_exclusive, auto_adjust=auto_adjust, actions=False
    )


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a yfinance history frame to the canonical schema."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = raw.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"datetime": "date"})
    df = df[[c for c in COLUMNS if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return df


def fetch_prices(
    symbol: str,
    start: str,
    as_of: str,
    auto_adjust: bool = True,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Fetch daily prices for ``symbol`` on [start, as_of] inclusive, with retries."""
    end_exclusive = (date.fromisoformat(as_of) + timedelta(days=1)).isoformat()
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = normalise(_history(symbol, start, end_exclusive, auto_adjust))
            if not df.empty:
                return df
            last_err = FetchError(f"no data returned for {symbol}")
        except Exception as exc:  # network / vendor errors
            last_err = exc
        time.sleep(backoff * (attempt + 1))
    raise FetchError(f"failed to fetch {symbol}: {last_err}")
