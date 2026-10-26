"""Data quality gate.

Unlike a single score, the gate returns explicit issues with severities and a hard
pass/fail verdict. Any ERROR makes the series unusable for evaluation; WARNINGs are
recorded in the report but do not block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import pandas as pd

from credibility.config import QualityConfig

Severity = Literal["error", "warning"]
OHLCV = ["open", "high", "low", "close", "volume"]


@dataclass
class Issue:
    check: str
    severity: Severity
    detail: str


@dataclass
class QualityReport:
    symbol: str
    n_rows: int
    start: str | None
    end: str | None
    passed: bool
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        return asdict(self)


def validate(df: pd.DataFrame, symbol: str, cfg: QualityConfig, as_of: str) -> QualityReport:
    """Validate a price frame with columns date + open/high/low/close/volume."""
    report = QualityReport(symbol=symbol, n_rows=len(df), start=None, end=None, passed=False)

    def add(check: str, severity: Severity, detail: str) -> None:
        report.issues.append(Issue(check, severity, detail))

    missing_cols = [c for c in ["date", *OHLCV] if c not in df.columns]
    if missing_cols:
        add("schema", "error", f"missing columns: {missing_cols}")
        return report
    if df.empty:
        add("length", "error", "no rows")
        return report

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    report.start = df["date"].min().strftime("%Y-%m-%d")
    report.end = df["date"].max().strftime("%Y-%m-%d")

    if len(df) < cfg.min_rows:
        add("length", "error", f"{len(df)} rows < min_rows={cfg.min_rows}")

    if df["date"].duplicated().any():
        add("duplicate_dates", "error", f"{int(df['date'].duplicated().sum())} duplicate dates")
    if not df["date"].is_monotonic_increasing:
        add("date_order", "error", "dates are not sorted ascending")

    cells = df[OHLCV]
    missing_frac = float(cells.isna().sum().sum()) / max(cells.size, 1)
    if missing_frac > cfg.max_missing_frac:
        add("missing", "error", f"{missing_frac:.2%} missing cells > {cfg.max_missing_frac:.2%}")
    elif missing_frac > 0:
        add("missing", "warning", f"{missing_frac:.2%} missing cells")

    prices = df[["open", "high", "low", "close"]]
    n_nonpos = int((prices <= 0).sum().sum())
    if n_nonpos or (df["volume"] < 0).any():
        add("non_positive", "error", f"{n_nonpos} non-positive prices / negative volumes")

    ohlc_bad = int(
        (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ).sum()
    )
    if ohlc_bad:
        # Tolerate rare vendor rounding artefacts; fail if systematic (>0.5% of rows).
        sev: Severity = "error" if ohlc_bad / len(df) > 0.005 else "warning"
        add("ohlc_integrity", sev, f"{ohlc_bad} rows violate OHLC ordering")

    zero_vol = float((df["volume"] == 0).mean())
    if zero_vol > cfg.max_zero_volume_frac:
        add("zero_volume", "warning", f"{zero_vol:.2%} zero-volume days")

    ret = df["close"].pct_change(fill_method=None).abs()
    n_jumps = int((ret > cfg.max_abs_daily_return).sum())
    if n_jumps:
        add("extreme_returns", "warning", f"{n_jumps} days with |return| > {cfg.max_abs_daily_return:.0%} (possible unadjusted split)")

    gaps = df["date"].diff().dt.days.dropna()
    n_gaps = int((gaps > cfg.max_gap_days).sum())
    if n_gaps:
        add("calendar_gaps", "warning", f"{n_gaps} gaps longer than {cfg.max_gap_days} days (max {int(gaps.max())})")

    stale = (pd.Timestamp(as_of) - df["date"].max()).days
    if stale > cfg.max_staleness_days:
        add("staleness", "error", f"last row is {stale} days before as_of {as_of}")

    report.passed = not report.errors
    return report
