"""Credibility dashboard (Flask). Shows each forecast together with its verdict and evidence.

    GET /                    dashboard page
    GET /health              liveness + registry status
    GET /api/universe        tickers + sectors
    GET /api/forecast        ?symbol=TCS.NS&task=return|volatility&horizon=5
    GET /api/evidence        ?task=return   gate decisions + interval coverage tables
    GET /api/history         ?symbol=TCS.NS  recent closes

Run:  python -m credibility.app      (dev server)   |   gunicorn "credibility.app:create_app()"   (container)
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from credibility.config import PROJECT_ROOT, load_data_config, load_eval_config
from credibility.data import Snapshot
from credibility.evaluation import TASK_NAMES
from credibility.gate import Registry, ServeError, serve_forecast

WEB_DIR = Path(__file__).parent / "web"


def _clean(obj):
    """JSON-safe: NaN/inf -> None."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def create_app(registry_root: str | Path | None = None, eval_config: str | None = None) -> Flask:
    app = Flask(__name__)
    cfg = load_eval_config(eval_config)
    data_cfg = load_data_config()
    reg_root = Path(registry_root or os.environ.get("REGISTRY_DIR") or PROJECT_ROOT / "registry")
    registry = Registry(reg_root)
    sectors = {i.symbol: i.sector for i in data_cfg.universe}

    def verdicts(task: str) -> pd.DataFrame | None:
        p = reg_root / "evidence" / task / "ticker_verdicts.csv"
        return pd.read_csv(p) if p.exists() else None

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "registry_entries": len(registry.entries()),
                        "served": registry.index(), "snapshot": cfg.snapshot})

    @app.get("/api/universe")
    def universe():
        snap = Snapshot.open(cfg.snapshot)
        syms = [s for s in snap.symbols() if not s.startswith("^")]
        return jsonify({"symbols": [{"symbol": s, "sector": sectors.get(s, "")} for s in syms],
                        "tasks": {"return": list(cfg.horizons), "volatility": list(cfg.vol_horizons)}})

    @app.get("/api/forecast")
    def forecast():
        symbol = request.args.get("symbol", "").strip()
        task = request.args.get("task", "return")
        if task not in TASK_NAMES:
            return jsonify({"error": f"task must be one of {TASK_NAMES}"}), 400
        try:
            horizon = int(request.args.get("horizon", "5"))
        except ValueError:
            return jsonify({"error": "horizon must be an integer"}), 400
        try:
            out = serve_forecast(symbol, task, horizon, cfg, registry, as_of=request.args.get("as_of"),
                                 verdicts=verdicts(task))
        except ServeError as exc:
            return jsonify({"error": str(exc)}), 404
        return jsonify(_clean(out))

    @app.get("/api/evidence")
    def evidence():
        task = request.args.get("task", "return")
        if task not in TASK_NAMES:
            return jsonify({"error": f"task must be one of {TASK_NAMES}"}), 400
        ev = reg_root / "evidence" / task
        if not (ev / "decisions.csv").exists():
            return jsonify({"error": "no evidence yet; run `gate build`"}), 404
        out = {"decisions": pd.read_csv(ev / "decisions.csv").to_dict("records")}
        if (ev / "coverage.csv").exists():
            out["coverage"] = pd.read_csv(ev / "coverage.csv").to_dict("records")
        return jsonify(_clean(out))

    @app.get("/api/history")
    def history():
        symbol = request.args.get("symbol", "").strip()
        snap = Snapshot.open(cfg.snapshot)
        if symbol not in snap.symbols():
            return jsonify({"error": f"unknown symbol {symbol}"}), 404
        df = snap.load(symbol).tail(int(request.args.get("n", 250)))
        return jsonify({"symbol": symbol, "dates": df["date"].dt.strftime("%Y-%m-%d").tolist(),
                        "close": [round(float(v), 2) for v in df["close"]]})

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)
