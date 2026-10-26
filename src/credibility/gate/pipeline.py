"""Evidence -> decisions -> registry. One call per task from stored backtest predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credibility import config as _config
from credibility.config import CONFIG_DIR, EvalConfig
from credibility.evaluation import get_task, summarize
from credibility.gate.gate import GateConfig, champions, decide, ticker_verdicts
from credibility.gate.registry import Registry, file_hash
from credibility.models import HORIZON_AWARE, make_model
from credibility.uncertainty import ConformalConfig, build_intervals, coverage_table


def _spec(model: str, horizons) -> dict:
    kw = {"horizons": tuple(horizons)} if model in HORIZON_AWARE else {}
    return make_model(model, **kw).get_params()


def build_gate(cfg: EvalConfig, task_name: str, run_dir: Path, registry_root: Path, gate_cfg: GateConfig = GateConfig(),
               conf_cfg: ConformalConfig = ConformalConfig(), log=print) -> pd.DataFrame:
    """Evaluate a dev run, decide, write evidence tables and registry entries. Returns the decisions table."""
    run_dir = Path(run_dir)
    task = get_task(cfg, task_name)
    preds = pd.read_parquet(run_dir / "predictions.parquet")
    summary = summarize(preds, cfg, task)
    iv = build_intervals(preds, task, conf_cfg)
    cov = coverage_table(iv, task, alpha=conf_cfg.alpha)
    decisions = decide(summary, cov, task, gate_cfg)
    champs = champions(decisions, task)
    verdicts = ticker_verdicts(preds, cfg, task, gate_cfg.alpha)

    ev = Path(registry_root) / "evidence" / task.name
    ev.mkdir(parents=True, exist_ok=True)
    summary.to_csv(ev / "summary.csv", index=False)
    cov.to_csv(ev / "coverage.csv", index=False)
    decisions.to_csv(ev / "decisions.csv", index=False)
    verdicts.to_csv(ev / "ticker_verdicts.csv", index=False)

    reg = Registry(registry_root)
    manifest_hash = file_hash(_config.SNAPSHOT_DIR / cfg.snapshot / "manifest.json")
    cfg_hash = {p.name: file_hash(p) for p in (CONFIG_DIR / "data.yaml", CONFIG_DIR / "eval.yaml")}
    base = {"task": task.name, "snapshot": cfg.snapshot, "snapshot_manifest_sha256": manifest_hash,
            "config_sha256": cfg_hash, "dev_run": run_dir.name, "calibration_runs": [run_dir.name],
            "gate_config": gate_cfg.__dict__, "conformal_config": conf_cfg.__dict__}
    for h in task.horizons:
        for _, d in decisions[decisions["horizon"] == h].iterrows():
            reg.register({**base, "horizon": h, "model": d["model"], "params": _spec(d["model"], task.horizons),
                          "role": "candidate_promoted" if d["promote"] else "candidate_rejected",
                          "metrics": {k: d[k] for k in ("skill", "dm_p_adj", "coverage", "n_origins")},
                          "reason": d["reason"]})
        c = champs[h]
        row = decisions[(decisions["horizon"] == h) & (decisions["model"] == c["model"])]
        metrics = ({k: row.iloc[0][k] for k in ("skill", "dm_p_adj", "coverage", "n_origins")} if len(row)
                   else {"skill": 0.0})
        reg.register({**base, "horizon": h, "model": c["model"], "params": _spec(c["model"], task.horizons),
                      "role": c["role"], "metrics": metrics,
                      "reason": ("highest-skill promoted model" if c["role"] == "champion"
                                 else f"no candidate passed the gate; serving benchmark {task.benchmark}")},
                     serve=True)
        log(f"[{task.name} h={h}] serving {c['model']} ({c['role']})")
    return decisions
