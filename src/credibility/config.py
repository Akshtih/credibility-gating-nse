"""Typed loading of configs/data.yaml and project path helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"


@dataclass(frozen=True)
class QualityConfig:
    min_rows: int = 1000
    max_missing_frac: float = 0.01
    max_zero_volume_frac: float = 0.02
    max_abs_daily_return: float = 0.25
    max_gap_days: int = 10
    max_staleness_days: int = 7


@dataclass(frozen=True)
class Instrument:
    symbol: str
    sector: str


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    as_of_date: str
    auto_adjust: bool
    snapshot_name: str
    benchmark: str
    universe: tuple[Instrument, ...]
    quality: QualityConfig = field(default_factory=QualityConfig)

    @property
    def symbols(self) -> list[str]:
        return [i.symbol for i in self.universe]


@dataclass(frozen=True)
class StatsConfig:
    alpha: float = 0.05
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    snapshot: str
    models: tuple[str, ...]
    benchmark_model: str
    horizons: tuple[int, ...]
    min_train_rows: int
    origin_step: int
    holdout_start: str
    results_dir: str
    stats: StatsConfig = field(default_factory=StatsConfig)
    vol_models: tuple[str, ...] = ("hist_vol", "ewma", "garch", "har")
    vol_benchmark: str = "hist_vol"
    vol_horizons: tuple[int, ...] = (5, 21)

    @property
    def results_path(self) -> Path:
        return PROJECT_ROOT / self.results_dir


def load_eval_config(path: Path | str | None = None) -> EvalConfig:
    path = Path(path) if path else CONFIG_DIR / "eval.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalConfig(
        snapshot=str(raw["snapshot"]),
        models=tuple(raw["models"]),
        benchmark_model=str(raw["benchmark_model"]),
        horizons=tuple(int(h) for h in raw["horizons"]),
        min_train_rows=int(raw["min_train_rows"]),
        origin_step=int(raw["origin_step"]),
        holdout_start=str(raw["holdout_start"]),
        results_dir=str(raw["results_dir"]),
        stats=StatsConfig(**raw.get("stats", {})),
        vol_models=tuple(raw.get("volatility", {}).get("models", ("hist_vol", "ewma", "garch", "har"))),
        vol_benchmark=str(raw.get("volatility", {}).get("benchmark_model", "hist_vol")),
        vol_horizons=tuple(int(h) for h in raw.get("volatility", {}).get("horizons", (5, 21))),
    )


def load_data_config(path: Path | str | None = None) -> DataConfig:
    path = Path(path) if path else CONFIG_DIR / "data.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return DataConfig(
        start_date=str(raw["start_date"]),
        as_of_date=str(raw["as_of_date"]),
        auto_adjust=bool(raw["auto_adjust"]),
        snapshot_name=str(raw["snapshot_name"]),
        benchmark=str(raw["benchmark"]),
        universe=tuple(Instrument(**i) for i in raw["universe"]),
        quality=QualityConfig(**raw.get("quality", {})),
    )
