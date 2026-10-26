"""Phase 3: rolling-origin backtester, metrics, Diebold-Mariano tests, dev/holdout protocol (2 tasks)."""

from credibility.evaluation.backtest import HoldoutLockedError, forecast_at_origin, forecast_full, run_backtest
from credibility.evaluation.report import format_table, summarize
from credibility.evaluation.stats import block_bootstrap_ci, dm_test, holm, overlap_lag
from credibility.evaluation.tasks import TASK_NAMES, Task, get_task

__all__ = ["HoldoutLockedError", "forecast_at_origin", "forecast_full", "run_backtest", "format_table", "summarize",
           "block_bootstrap_ci", "dm_test", "holm", "overlap_lag", "TASK_NAMES", "Task", "get_task"]
