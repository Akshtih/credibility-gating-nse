"""Phase 5: credibility gate, model registry, serving with benchmark fallback."""

from credibility.gate.gate import GateConfig, champions, decide, ticker_verdicts
from credibility.gate.pipeline import build_gate
from credibility.gate.registry import Registry
from credibility.gate.serve import ServeError, serve_forecast

__all__ = ["GateConfig", "champions", "decide", "ticker_verdicts", "build_gate", "Registry", "ServeError",
           "serve_forecast"]
