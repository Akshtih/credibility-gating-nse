"""stock-credibility: credibility gating for stock forecasts on Indian markets.

Subpackages (build order, see docs/HANDOFF.md):
    data         Phase 1  fetch, quality gate, versioned snapshots   [DONE]
    models       Phase 2  forecaster interface + naive/ARIMA/Prophet/XGBoost
    evaluation   Phase 3  rolling-origin backtester, metrics, DM tests
    uncertainty  Phase 4  conformal prediction intervals
    gate         Phase 5  promotion gate + model registry + fallback
    monitoring   Phase 6  drift detection + regime replay
"""

__version__ = "0.1.0"
