"""Summary table: one row per (model, horizon) versus the task benchmark.

Columns of note
    skill         1 - loss/loss_benchmark under the task loss (>0 beats the benchmark; = R2_OS for returns)
    dm_p_adj      pooled Diebold-Mariano one-sided p-value, Holm-adjusted across all model x horizon tests
                  (loss differential averaged across tickers per origin date first, which
                  accounts for cross-sectional correlation)
    ci_lo/ci_hi   block-bootstrap CI of the pooled mean loss differential (model - benchmark)
    n_sig_tickers number of tickers where the model is significantly better (per-ticker DM, Holm over tickers)
"""

from __future__ import annotations

import pandas as pd

from credibility.config import EvalConfig
from credibility.evaluation.metrics import align_with_benchmark, summarize_group
from credibility.evaluation.stats import block_bootstrap_ci, dm_test, holm, overlap_lag
from credibility.evaluation.tasks import Task, get_task


def summarize(preds: pd.DataFrame, cfg: EvalConfig, task: Task | str = "return") -> pd.DataFrame:
    task = get_task(cfg, task) if isinstance(task, str) else task
    df = align_with_benchmark(preds, task.benchmark, task)
    rows = []
    for (model, h), g in df.groupby(["model", "horizon"]):
        lag = overlap_lag(int(h), cfg.origin_step)
        row = {"model": model, "horizon": int(h), **summarize_group(g)}

        pooled = (g.assign(d=g["loss"] - g["loss_bm"]).groupby("origin_date")["d"].mean().sort_index())
        dm = dm_test(pooled.to_numpy(), 0.0 * pooled.to_numpy(), lag)
        row.update(dm_stat=dm["stat"], dm_p=dm["p_better"], n_origins=dm["n"])
        lo, hi = block_bootstrap_ci(pooled.to_numpy(), block=max(lag + 1, 5),
                                    n_boot=cfg.stats.bootstrap_resamples, seed=cfg.stats.bootstrap_seed,
                                    alpha=cfg.stats.alpha)
        row.update(mean_diff=dm["mean_diff"], ci_lo=lo, ci_hi=hi)

        pvals, syms = [], []
        for sym, gs in g.groupby("symbol"):
            gs = gs.sort_values("origin_idx")
            pvals.append(dm_test(gs["loss"].to_numpy(), gs["loss_bm"].to_numpy(), lag)["p_better"])
            syms.append(sym)
        adj = holm(pvals)
        row["n_sig_tickers"] = int((adj < cfg.stats.alpha).sum())
        row["n_tickers"] = len(syms)
        row["n_tickers_better"] = int(sum(1 for p in pvals if p == p and p < 0.5))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.insert(0, "task", task.name)
    out["dm_p_adj"] = holm(out["dm_p"].to_numpy())
    out["beats_benchmark"] = (out["dm_p_adj"] < cfg.stats.alpha) & (out["skill"] > 0)
    cols = ["task", "model", "horizon", "n", "n_origins", "loss", "loss_bm", "skill", "rmse", "rmse_bm", "mae_ratio",
            "dir_acc", "dir_ci_lo", "dir_ci_hi", "always_up_rate", "mean_diff", "ci_lo", "ci_hi", "dm_stat", "dm_p",
            "dm_p_adj", "n_sig_tickers", "n_tickers_better", "n_tickers", "beats_benchmark"]
    return out[cols].sort_values(["horizon", "model"]).reset_index(drop=True)


def format_table(tbl: pd.DataFrame) -> str:
    show = tbl[["task", "model", "horizon", "n", "skill", "dm_p_adj", "n_sig_tickers", "n_tickers_better",
                "n_tickers", "beats_benchmark"]].copy()
    for c in ["skill", "dm_p_adj"]:
        show[c] = show[c].map(lambda v: f"{v:.4f}" if v == v else "nan")
    if tbl["task"].iloc[0] == "return":
        extra = tbl[["dir_acc", "always_up_rate"]].map(lambda v: f"{v:.3f}" if v == v else "nan")
        show = pd.concat([show, extra], axis=1)
    return show.to_string(index=False)
