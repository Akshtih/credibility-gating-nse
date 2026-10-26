"""Phase 4: conformal prediction intervals (split + adaptive), baselines, coverage metrics, regimes."""

from credibility.uncertainty.conformal import ConformalConfig, build_intervals, conformal_quantile, series_intervals
from credibility.uncertainty.coverage import coverage_table, interval_score, nifty_regimes

__all__ = ["ConformalConfig", "build_intervals", "conformal_quantile", "series_intervals",
           "coverage_table", "interval_score", "nifty_regimes"]
