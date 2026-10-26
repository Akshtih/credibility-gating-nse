"""Phase-1 pipeline: fetch -> validate -> snapshot."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from credibility.config import SNAPSHOT_DIR, DataConfig
from credibility.data.fetch import FetchError, fetch_prices
from credibility.data.quality import validate
from credibility.data.store import Snapshot


def build_snapshot(cfg: DataConfig, base: Path = SNAPSHOT_DIR, overwrite: bool = False, log=print) -> Snapshot:
    """Fetch every instrument (+ benchmark), run the quality gate, write an immutable snapshot."""
    snap = Snapshot.create(
        cfg.snapshot_name,
        config={
            "start_date": cfg.start_date,
            "as_of_date": cfg.as_of_date,
            "auto_adjust": cfg.auto_adjust,
            "benchmark": cfg.benchmark,
            "universe": [asdict(i) for i in cfg.universe],
            "quality": asdict(cfg.quality),
        },
        base=base,
        overwrite=overwrite,
    )
    for symbol in [*cfg.symbols, cfg.benchmark]:
        try:
            df = fetch_prices(symbol, cfg.start_date, cfg.as_of_date, cfg.auto_adjust)
        except FetchError as exc:
            snap.add_failure(symbol, str(exc))
            log(f"[FAIL] {symbol}: {exc}")
            continue
        report = validate(df, symbol, cfg.quality, cfg.as_of_date)
        snap.add(symbol, df, report.to_dict())
        status = "PASS" if report.passed else "FAIL-QUALITY"
        log(f"[{status}] {symbol}: {len(df)} rows {report.start}..{report.end} "
            f"({len(report.errors)} errors, {len(report.warnings)} warnings)")
    return snap
