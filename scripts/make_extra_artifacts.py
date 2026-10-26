"""Extra paper artifacts (robustness checks; many-candidate experiment). Writes NEW files only:

    paper/generated/table_robust.tex, table_many.tex, numbers_extra.tex

The original generated files (numbers.tex, table_*.tex, fig_*.pdf) are never touched; scripts/protect_check.py verifies it.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credibility.config import load_eval_config  # noqa: E402
from credibility.evaluation import get_task  # noqa: E402
from credibility.robustness import dm_size, skill_variants  # noqa: E402

OUT = ROOT / "paper" / "generated"
NUMS: dict[str, str] = {}
LABEL = {"arima": "ARIMA", "prophet": "Prophet", "prophet_anchored": "Prophet-A", "xgboost": "XGBoost",
         "ewma": "EWMA", "garch": "GARCH", "har": "HAR"}
WORD = {1: "one", 5: "five", 21: "twentyone", 68: "sixtyeight", 122: "onetwentytwo", 0: "zero", 2: "two"}


def num(name: str, value) -> None:
    NUMS["".join(ch for ch in name if ch.isalpha())] = str(value)


def f(x, d=3, signed=True):
    return "--" if x is None or not np.isfinite(x) else (f"{x:+.{d}f}" if signed else f"{x:.{d}f}")


def robustness(cfg) -> None:
    preds = {(t, p): pd.read_parquet(cfg.results_path / r / "predictions.parquet")
             for t, p, r in (("return", "dev", "ret_dev"), ("return", "hold", "ret_hold"),
                             ("volatility", "dev", "vol_dev"), ("volatility", "hold", "vol_hold"))}
    # ---- volatility table
    vt = get_task(cfg, "volatility")
    rows, won = [], []
    for h in cfg.vol_horizons:
        for m in ("ewma", "garch", "har"):
            d = skill_variants(preds[("volatility", "dev")], vt, m, h)
            o = skill_variants(preds[("volatility", "hold")], vt, m, h)
            rows.append(f"{h} & {LABEL[m]} & {f(d['full'])} & {f(d['ex_covid'])} & {f(d['ex_top5'])} & {f(d['alt_loss'])} & "
                        f"{f(o['full'])} & {f(o['alt_loss'])} \\\\")
            won += [d["dates_won"], o["dates_won"]]
            for k in ("full", "ex_covid", "ex_top5", "alt_loss"):
                num(f"Rob{WORD[h]}{m}Dev{k}", f(d[k]))
            num(f"Rob{WORD[h]}{m}Holdfull", f(o["full"]))
    (OUT / "table_robust.tex").write_text(
        "\\begin{tabular}{clrrrrrr}\n\\toprule\n & & \\multicolumn{4}{c}{Development} & \\multicolumn{2}{c}{Holdout} \\\\\n"
        "\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\n$h$ & Model & Full & Ex-COVID & Ex-top-5\\% & SD loss & Full & SD loss \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    num("RobDatesWonMin", f"{100 * min(won):.0f}")
    num("RobDatesWonMax", f"{100 * max(won):.0f}")
    # ---- return task: best skill over every candidate and view
    rt = get_task(cfg, "return")
    best = {}
    for period in ("dev", "hold"):
        vals_full, vals_ex = [], []
        for m in ("arima", "prophet", "prophet_anchored", "xgboost"):
            for h in cfg.horizons:
                v = skill_variants(preds[("return", period)], rt, m, h)
                vals_full.append(v["full"])
                vals_ex.append(v["ex_covid"])
                if period == "dev":
                    num(f"RobRetDevAlt{m}{WORD[h]}", f(v["alt_loss"]))
        best[period] = (max(vals_full), max(vals_ex))
    num("RobRetMaxDevFull", f(best["dev"][0]))
    num("RobRetMaxDevExCovid", f(best["dev"][1]))
    num("RobRetMaxHoldFull", f(best["hold"][0]))
    # ---- size of the DM test
    for n in (68, 122):
        for lag in (0, 2):
            num(f"Size{WORD[n]}Lag{WORD[lag]}", f"{100 * dm_size(n, lag, n_sims=20000, seed=1):.1f}")


def size_table() -> None:
    """DM test size under the real overlap structure of 21-day forecasts sampled every 10 days, by lag, with MC error."""
    from credibility.robustness import dm_size_overlap

    rows, se_max = [], 0.0
    for lag in range(5):
        cells = []
        for n in (68, 122):
            p, se = dm_size_overlap(n, 10, 21, lag, n_sims=20000, seed=1)
            cells.append(f"{100 * p:.1f} $\\pm$ {100 * se:.1f}")
            se_max = max(se_max, se)
            num(f"SizeReal{WORD[n]}Lag{WORD[lag] if lag in WORD else ['zero','one','two','three','four'][lag]}", f"{100 * p:.1f}")
        rows.append(f"{lag} & {cells[0]} & {cells[1]} \\\\")
    (OUT / "table_dmsize.tex").write_text(
        "\\begin{tabular}{lcc}\n\\toprule\nLag & $n=68$ origins & $n=122$ origins \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    num("SizeMCSEMax", f"{100 * se_max:.2f}")


def lag_sensitivity(cfg) -> None:
    """Would any gate decision change if the 21-day Bartlett lag were 2 (used), 3 or 4?"""
    import credibility.evaluation.report as rep
    from credibility.evaluation import summarize

    orig = rep.overlap_lag
    flips = 0
    try:
        for task_name, runs in (("return", ("ret_dev", "ret_hold")), ("volatility", ("vol_dev", "vol_hold"))):
            task = get_task(cfg, task_name)
            for run in runs:
                p = pd.read_parquet(cfg.results_path / run / "predictions.parquet")
                base = None
                for L in (2, 3, 4):
                    rep.overlap_lag = lambda h, step, L=L: (L if h == 21 else orig(h, step))
                    t = summarize(p, cfg, task)[["model", "horizon", "beats_benchmark"]].set_index(["model", "horizon"])["beats_benchmark"]
                    if base is None:
                        base = t
                    else:
                        flips += int((t != base).sum())
    finally:
        rep.overlap_lag = orig
    num("LagFlips", flips)


def calibration_tables(cfg) -> None:
    """ACI coverage by horizon and across individual stocks (pooled over models incl. the benchmark), dev and holdout."""
    from credibility.uncertainty import ConformalConfig, build_intervals, coverage_table

    conf = ConformalConfig()
    rows = []
    for tname, dev_run, hold_run in (("return", "ret_dev", "ret_hold"), ("volatility", "vol_dev", "vol_hold")):
        task = get_task(cfg, tname)
        dev = pd.read_parquet(cfg.results_path / dev_run / "predictions.parquet")
        hold = pd.read_parquet(cfg.results_path / hold_run / "predictions.parquet")
        views = {"Dev.": build_intervals(dev, task, conf),
                 "Holdout": build_intervals(pd.concat([dev, hold], ignore_index=True), task, conf)}
        views["Holdout"] = views["Holdout"][views["Holdout"]["origin_date"] >= pd.Timestamp(cfg.holdout_start)]
        for label, iv in views.items():
            byh = coverage_table(iv, task, alpha=conf.alpha, by=("horizon",), methods=("aci",)).set_index("horizon")["coverage"]
            bys = coverage_table(iv, task, alpha=conf.alpha, by=("symbol",), methods=("aci",))["coverage"]
            h_cells = " & ".join(f"{100 * byh[h]:.1f}" if h in byh.index else "--" for h in (1, 5, 21))
            rows.append(f"{'Return' if tname == 'return' else 'Volatility'} & {label} & {h_cells} & "
                        f"{100 * bys.min():.1f} & {100 * bys.median():.1f} & {100 * bys.max():.1f} \\\\")
            key = f"Calib{'Ret' if tname == 'return' else 'Vol'}{'Dev' if label == 'Dev.' else 'Hold'}"
            num(key + "StockMin", f"{100 * bys.min():.1f}")
            num(key + "StockMax", f"{100 * bys.max():.1f}")
    (OUT / "table_calib.tex").write_text(
        "\\begin{tabular}{llcccccc}\n\\toprule\n & & \\multicolumn{3}{c}{By horizon (\\%)} & \\multicolumn{3}{c}{Across stocks (\\%)} \\\\\n"
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\nTask & Period & $h{=}1$ & $h{=}5$ & $h{=}21$ & Min & Median & Max \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")


def main() -> None:
    cfg = load_eval_config()
    robustness(cfg)
    size_table()
    lag_sensitivity(cfg)
    calibration_tables(cfg)
    if (cfg.results_path / "many_candidates" / "rules_summary.csv").exists():
        from make_many_artifacts import many_artifacts  # noqa: PLC0415  (added in the many-candidates phase)
        many_artifacts(cfg, OUT, num, f)
    (OUT / "numbers_extra.tex").write_text("\n".join(f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in sorted(NUMS.items())) + "\n",
                                           encoding="utf-8")
    print("wrote", [p.name for p in OUT.glob("*extra*")], "and", [p.name for p in OUT.glob("table_robust.tex")],
          "|", len(NUMS), "macros")


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    main()
