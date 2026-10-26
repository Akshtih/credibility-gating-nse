"""The credibility gate: decide which models may serve forecasts.

A candidate model is PROMOTED for a (task, horizon) only if, on the development period,
    1. it beats the benchmark: skill > min_skill AND Holm-adjusted pooled Diebold-Mariano p < alpha,
    2. its (adaptive conformal) intervals are calibrated: |coverage - nominal| <= coverage_tol,
    3. it has enough evidence: n_origins >= min_origins.
Among promoted candidates the champion is the one with the highest skill; if none is promoted the
BENCHMARK (naive / hist_vol) is served as the fallback. The gate never promotes on in-sample error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credibility.config import EvalConfig
from credibility.evaluation.metrics import align_with_benchmark
from credibility.evaluation.stats import dm_test, holm, overlap_lag
from credibility.evaluation.tasks import Task


@dataclass(frozen=True)
class GateConfig:
    alpha: float = 0.05
    min_skill: float = 0.0
    coverage_nominal: float = 0.8
    coverage_tol: float = 0.05
    min_origins: int = 60
    interval_method: str = "aci"


def decide(summary: pd.DataFrame, coverage: pd.DataFrame, task: Task, cfg: GateConfig = GateConfig()) -> pd.DataFrame:
    """One row per (model, horizon) with the individual checks, the verdict and a human-readable reason."""
    cov = coverage[coverage["method"] == cfg.interval_method].set_index(["model", "horizon"])["coverage"]
    rows = []
    for _, r in summary.iterrows():
        c = float(cov.get((r["model"], r["horizon"]), np.nan))
        beats = bool(r["skill"] > cfg.min_skill and r["dm_p_adj"] < cfg.alpha)
        calibrated = bool(np.isfinite(c) and abs(c - cfg.coverage_nominal) <= cfg.coverage_tol)
        enough = bool(r["n_origins"] >= cfg.min_origins)
        reasons = []
        if not beats:
            reasons.append(f"does not beat {task.benchmark} (skill={r['skill']:+.4f}, adj p={r['dm_p_adj']:.3f})")
        if not calibrated:
            reasons.append(f"intervals miscalibrated (coverage={c:.3f} vs {cfg.coverage_nominal})")
        if not enough:
            reasons.append(f"too few origins ({int(r['n_origins'])})")
        rows.append({
            "task": task.name, "model": r["model"], "horizon": int(r["horizon"]), "skill": float(r["skill"]),
            "dm_p_adj": float(r["dm_p_adj"]), "coverage": c, "n_origins": int(r["n_origins"]),
            "beats_benchmark": beats, "calibrated": calibrated, "enough_evidence": enough,
            "promote": beats and calibrated and enough, "reason": "; ".join(reasons) or "passes all checks",
        })
    return pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)


def champions(decisions: pd.DataFrame, task: Task) -> dict[int, dict]:
    """Per horizon: the promoted model with the highest skill, else the benchmark fallback."""
    out = {}
    for h in task.horizons:
        d = decisions[(decisions["horizon"] == h) & decisions["promote"]]
        if len(d):
            best = d.sort_values("skill", ascending=False).iloc[0]
            out[h] = {"model": best["model"], "role": "champion", "skill": float(best["skill"])}
        else:
            out[h] = {"model": task.benchmark, "role": "fallback", "skill": 0.0}
    return out


def ticker_verdicts(preds: pd.DataFrame, cfg: EvalConfig, task: Task, alpha: float = 0.05) -> pd.DataFrame:
    """Per (ticker, model, horizon): TRUSTED (skill>0 and significant after Holm over tickers),
    CAUTION (skill>0, not significant), NOT_BETTER (skill<=0)."""
    df = align_with_benchmark(preds, task.benchmark, task)
    rows = []
    for (model, h), g in df.groupby(["model", "horizon"]):
        lag = overlap_lag(int(h), cfg.origin_step)
        recs = []
        for sym, gs in g.groupby("symbol"):
            gs = gs.sort_values("origin_idx")
            lm, lb = gs["loss"].mean(), gs["loss_bm"].mean()
            recs.append((sym, 1 - lm / lb if lb else np.nan,
                         dm_test(gs["loss"].to_numpy(), gs["loss_bm"].to_numpy(), lag)["p_better"], len(gs)))
        adj = holm([x[2] for x in recs])
        for (sym, skill, p, n), pa in zip(recs, adj):
            verdict = ("TRUSTED" if (skill > 0 and pa < alpha) else "CAUTION" if skill > 0 else "NOT_BETTER")
            rows.append({"task": task.name, "symbol": sym, "model": model, "horizon": int(h), "skill": skill,
                         "p": p, "p_adj": pa, "n": n, "verdict": verdict})
    return pd.DataFrame(rows)
