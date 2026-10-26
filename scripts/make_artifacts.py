"""Regenerate every table, figure and number used in the paper from stored results.

    python scripts/make_artifacts.py            (after: ret_dev, vol_dev, ret_hold, vol_hold runs + gate build)

Writes paper/generated/*.tex (tables, numbers.tex macros) and paper/generated/*.pdf (figures).
Nothing in the paper's results sections is typed by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credibility.config import load_data_config, load_eval_config  # noqa: E402
from credibility.data import Snapshot  # noqa: E402
from credibility.evaluation import get_task, summarize  # noqa: E402
from credibility.experiments import confusion, gate_validation, holdout_coverage  # noqa: E402
from credibility.gate import GateConfig  # noqa: E402
from credibility.gate.gate import decide  # noqa: E402
from credibility.monitoring import STRESS_WINDOWS, first_alarm_dates, policy_table, replay  # noqa: E402
from credibility.uncertainty import ConformalConfig, build_intervals, coverage_table, nifty_regimes  # noqa: E402

RUNS = {"return": ("ret_dev", "ret_hold"), "volatility": ("vol_dev", "vol_hold")}
OUT = ROOT / "paper" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6, "figure.dpi": 150,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType (IEEE PDF eXpress rejects Type 3 fonts)
NUMS: dict[str, str] = {}
LABEL = {"naive": "naive", "arima": "ARIMA", "prophet": "Prophet", "prophet_anchored": "Prophet-A", "xgboost": "XGBoost",
         "hist_vol": "HistVol", "ewma": "EWMA", "garch": "GARCH", "har": "HAR"}


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def f(x, d=3, signed=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{d}f}" if signed else f"{x:.{d}f}"


def pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def num(name: str, value) -> None:
    NUMS["".join(ch for ch in name if ch.isalpha())] = str(value)   # LaTeX macro names: letters only


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body, encoding="utf-8")


def main() -> None:
    cfg = load_eval_config()
    gate_cfg, conf = GateConfig(), ConformalConfig()
    snap = Snapshot.open(cfg.snapshot)
    nifty = snap.load(load_data_config().benchmark)
    data = {}
    for t, (dev_run, hold_run) in RUNS.items():
        task = get_task(cfg, t)
        dev = pd.read_parquet(cfg.results_path / dev_run / "predictions.parquet")
        hold = pd.read_parquet(cfg.results_path / hold_run / "predictions.parquet")
        dsum, hsum = summarize(dev, cfg, task), summarize(hold, cfg, task)
        cov_dev = coverage_table(build_intervals(dev, task, conf), task, alpha=conf.alpha)
        cov_hold = holdout_coverage(dev, hold, task, cfg.holdout_start, conf)
        dec = decide(dsum, cov_dev, task, gate_cfg)
        val = gate_validation(dec, hsum, gate_cfg.alpha)
        data[t] = dict(task=task, dev=dev, hold=hold, dsum=dsum, hsum=hsum, cov_dev=cov_dev, cov_hold=cov_hold,
                       dec=dec, val=val)
        n_series = dev["symbol"].nunique()
        num(f"{t}DevPreds", f"{len(dev):,}")
        num(f"{t}HoldPreds", f"{len(hold):,}")
        num(f"{t}DevOrigins", int(dev.groupby("symbol")["origin_idx"].nunique().median()))
        num(f"{t}HoldOrigins", int(hold.groupby("symbol")["origin_idx"].nunique().median()))
        num("NTickers", n_series)
        num("NSectors", len({i.sector for i in load_data_config().universe}))
    num("DevStart", str(data["return"]["dev"]["origin_date"].min().date()))
    num("DevEnd", str(data["return"]["dev"]["target_date"].max().date()))
    num("HoldStart", cfg.holdout_start)
    num("HoldEnd", str(data["return"]["hold"]["target_date"].max().date()))
    num("OriginStep", cfg.origin_step)


    # ---------------- forecast counts (exact, from the stored prediction files) ----------------
    crow = []
    for t in RUNS:
        for label, key in (("Dev.", "dev"), ("Holdout", "hold")):
            p = data[t][key]
            nm = p["model"].nunique()
            for h, g in p.groupby("horizon"):
                per_model = g[g["model"] == g["model"].iloc[0]]
                crow.append(f"{'Return' if t == 'return' else 'Volatility'} & {label} & {h} & "
                            f"{int(per_model.groupby('symbol')['origin_idx'].nunique().median())} & "
                            f"{len(per_model):,} & {nm} & {len(g):,} \\\\")
    write("table_counts.tex", "\\begin{tabular}{llcrrrr}\n\\toprule\nTask & Period & $h$ & Origins/stock & Forecasts/model & Models & Total \\\\\n\\midrule\n"
          + "\n".join(crow) + "\n\\bottomrule\n\\end{tabular}\n")
    mf = snap.manifest
    num("SnapshotDate", mf["created_utc"][:10])
    num("NFetchFailures", len(mf["failed"]))
    num("NQualityWarnings", sum(1 for v in mf["symbols"].values() if v["quality"]["issues"]))
    # how often the coverage criterion could have changed a gate decision
    num("CalibratedAll", int(all(data[t]["dec"]["calibrated"].all() for t in RUNS)))
    num("CovMinCand", f"{100 * min(data[t]['dec']['coverage'].min() for t in RUNS):.1f}")
    num("CovMaxCand", f"{100 * max(data[t]['dec']['coverage'].max() for t in RUNS):.1f}")

    # ---------------- named numbers for prose ----------------
    hw = {1: "one", 5: "five", 21: "twentyone"}
    for t in RUNS:
        for label, tbl in (("Dev", data[t]["dsum"]), ("Hold", data[t]["hsum"])):
            for _, r in tbl.iterrows():
                key = f"{label}{t}{r['model']}{hw[int(r['horizon'])]}"
                num("Skill" + key, f(r["skill"], 3, True))
                num("P" + key, pval(r["dm_p_adj"]))
    proto = ROOT / "pilot" / "backtest_summary.csv"
    if proto.exists():
        pr = pd.read_csv(proto, index_col=0).loc["MEAN"]
        num("ProtoInSample", f"{pr['mape_insample']:.1f}")
        num("ProtoOutSample", f"{pr['mape_prophet']:.1f}")
        num("ProtoNaive", f"{pr['mape_naive']:.1f}")
        num("ProtoCover", f"{100 * pr['cover']:.0f}")
        num("ProtoDir", f"{100 * pr['dir_prophet']:.0f}")

    # ---------------- results tables ----------------
    for t in RUNS:
        d = data[t]
        rows = []
        for _, r in d["dsum"].iterrows():
            h = d["hsum"][(d["hsum"]["model"] == r["model"]) & (d["hsum"]["horizon"] == r["horizon"])]
            hs = h.iloc[0] if len(h) else None
            extra = (f" & {f(r['dir_acc'], 3)} & {f(r['always_up_rate'], 3)}" if t == "return" else "")
            rows.append(f"{r['horizon']} & {LABEL[r['model']]} & {f(r['skill'], 3, True)} & "
                        f"[{f(-r['ci_hi'] / r['loss_bm'], 3, True)}, {f(-r['ci_lo'] / r['loss_bm'], 3, True)}] & "
                        f"{pval(r['dm_p_adj'])}{extra} & {f(hs['skill'], 3, True) if hs is not None else '--'} & "
                        f"{pval(hs['dm_p_adj']) if hs is not None else '--'} \\\\")
        extra_h = " & Dir. & Up-rate" if t == "return" else ""
        extra_c = "cc" if t == "return" else ""
        bm = LABEL[d["task"].benchmark]
        write(f"table_{t}.tex", "\\begin{tabular}{cl" + "rc" + "r" + extra_c + "rr}\n\\toprule\n"
              f"$h$ & Model & Skill & 95\\% CI & Adj.\\ $p$" + extra_h + " & Skill & Adj.\\ $p$ \\\\\n"
              " & & \\multicolumn{3}{c}{Development}" + (" & \\multicolumn{2}{c}{}" if t == "return" else "") +
              " & \\multicolumn{2}{c}{Holdout} \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        num(f"{t}Benchmark", LABEL[d["task"].benchmark])
        _ = bm

    # ---------------- gate validation ----------------
    vrows, conf_all = [], {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    naive_all = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for t in RUNS:
        v = data[t]["val"]
        for k, c in confusion(v).items():
            conf_all[k] += c
        for k, c in confusion(v, "naive_outcome").items():
            naive_all[k] += c
        for _, r in v.iterrows():
            vrows.append(f"{t[:3].capitalize() if t == 'return' else 'Vol'} & {r['horizon']} & {LABEL[r['model']]} & "
                         f"{f(r['dev_skill'], 3, True)} & {pval(r['dev_p_adj'])} & {'yes' if r['gate_promoted'] else 'no'} & "
                         f"{f(r['hold_skill'], 3, True)} & {pval(r['hold_p_adj'])} & {r['outcome']} \\\\")
    write("table_gate.tex", "\\begin{tabular}{clcrrcrrc}\n\\toprule\nTask & $h$ & Model & Dev.\\ skill & Dev.\\ $p$ & Promoted & "
          "Hold.\\ skill & Hold.\\ $p$ & Outcome \\\\\n\\midrule\n" + "\n".join(vrows) + "\n\\bottomrule\n\\end{tabular}\n")
    for k, c in conf_all.items():
        num(f"Gate{k}", c)
    for k, c in naive_all.items():
        num(f"Naive{k}", c)
    num("GateN", sum(conf_all.values()))

    # ---------------- interval coverage ----------------
    crow = []
    for t in RUNS:
        for label, tbl in (("Dev.", data[t]["cov_dev"]), ("Holdout", data[t]["cov_hold"])):
            g = tbl.groupby("method").agg(cov=("coverage", "mean"), width=("mean_width", "mean"),
                                          score=("interval_score", "mean"), n=("n", "sum"))
            for m in ("gauss", "native", "split", "aci"):
                if m in g.index:
                    r = g.loc[m]
                    crow.append(f"{'Return' if t == 'return' else 'Volatility'} & {label} & {m.replace('gauss', 'Gaussian').replace('native', 'Prophet native').replace('split', 'Split conformal').replace('aci', 'Adaptive conformal')} & "
                                f"{100 * r['cov']:.1f} & {r['width']:.3f} & {r['score']:.3f} \\\\")
                    num(f"Cov_{t}_{label[:3]}_{m}", f"{100 * r['cov']:.1f}")
    write("table_coverage.tex", "\\begin{tabular}{llrrrr}\n\\toprule\nTask & Period & Interval & Coverage (\\%) & Width & Score \\\\\n\\midrule\n"
          + "\n".join(crow) + "\n\\bottomrule\n\\end{tabular}\n")

    # coverage by regime (ACI vs Gaussian) on dev, return task
    reg_rows = []
    for t in RUNS:
        iv = build_intervals(data[t]["dev"], data[t]["task"], conf)
        iv = iv[iv["aci_lo"].notna()].copy()
        iv["regime"] = nifty_regimes(nifty, iv["origin_date"]).to_numpy()
        iv = iv.dropna(subset=["regime"])
        for reg in ("low", "high"):
            g = iv[iv["regime"] == reg]
            for m in (("gauss", "aci") if t == "return" else ("aci",)):
                cov = float(((g["actual"] >= g[f"{m}_lo"]) & (g["actual"] <= g[f"{m}_hi"])).mean())
                reg_rows.append(f"{'Return' if t == 'return' else 'Volatility'} & {reg} & {'Gaussian' if m == 'gauss' else 'Adaptive conformal'} & "
                                f"{100 * cov:.1f} & {len(g):,} \\\\")
                num(f"CovReg_{t}_{reg}_{m}", f"{100 * cov:.1f}")
    write("table_regime.tex", "\\begin{tabular}{lllrr}\n\\toprule\nTask & Regime & Interval & Coverage (\\%) & $n$ \\\\\n\\midrule\n"
          + "\n".join(reg_rows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- monitoring replay ----------------
    mrows = []
    cum_data = None
    for t, cands in (("volatility", [("garch", 5), ("ewma", 21)]), ("return", [("arima", 5), ("xgboost", 5), ("xgboost", 21)])):
        task = data[t]["task"]
        allp = pd.concat([data[t]["dev"], data[t]["hold"]], ignore_index=True)
        for c, h in cands:
            rep = replay(allp, task, c, h)
            pt = policy_table(rep, {"holdout": (cfg.holdout_start, "2030-01-01")}).set_index("policy")
            for pol in ("candidate", "guarded_rolling", "guarded_ph"):
                r = pt.loc[pol]
                mrows.append(f"{'Vol' if t == 'volatility' else 'Ret'} & {LABEL[c]} & {h} & "
                             f"{pol.replace('guarded_rolling', 'guard (rolling)').replace('guarded_ph', 'guard (PH)')} & "
                             f"{f(r['skill_all'], 3, True)} & {f(r['skill_covid_2020'], 3, True)} & "
                             f"{f(r['skill_drawdown_2022'], 3, True)} & {f(r['skill_holdout'], 3, True)} & "
                             f"{100 * r['candidate_share']:.0f} & {r['switches_per_series']:.1f} \\\\")
            if (t, c, h) == ("volatility", "ewma", 21):
                cum_data = rep
    write("table_monitor.tex", "\\begin{tabular}{llclrrrrrr}\n\\toprule\nTask & Cand. & $h$ & Policy & All & COVID & 2022 & Holdout & Cand.\\ \\% & Sw. \\\\\n\\midrule\n"
          + "\n".join(mrows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, t in zip(axes, RUNS):
        d = data[t]["dsum"]
        models = list(dict.fromkeys(d["model"]))
        hs = sorted(d["horizon"].unique())
        w = 0.8 / len(hs)
        for i, h in enumerate(hs):
            sub = d[d["horizon"] == h].set_index("model").reindex(models)
            y = sub["skill"].to_numpy()
            lo = -sub["ci_hi"].to_numpy() / sub["loss_bm"].to_numpy()
            hi = -sub["ci_lo"].to_numpy() / sub["loss_bm"].to_numpy()
            x = np.arange(len(models)) + i * w - 0.4 + w / 2
            ax.bar(x, np.clip(y, -0.6, None), w, color=["#bbbbbb", "#666666", "#222222"][i % 3], edgecolor="k", lw=0.4,
                   label=f"h={h}")
            ok = y > -0.6
            ax.errorbar(x[ok], y[ok], yerr=[np.maximum(y - lo, 0)[ok], np.maximum(hi - y, 0)[ok]], fmt="none",
                        ecolor="k", elinewidth=0.5, capsize=1.2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABEL[m] for m in models], rotation=30, ha="right")
        ax.set_ylabel("skill vs benchmark" + (" (clipped at -0.6)" if t == "return" else ""))
        ax.set_title("Return task" if t == "return" else "Volatility task")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_skill.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.2))
    for ax, t in zip(axes, RUNS):
        for k, (label, tbl, col) in enumerate((("Dev.", data[t]["cov_dev"], "#999999"), ("Holdout", data[t]["cov_hold"], "#222222"))):
            g = tbl.groupby("method")["coverage"].mean()
            ms = [m for m in ("gauss", "native", "split", "aci") if m in g.index]
            ax.bar(np.arange(len(ms)) + k * 0.38 - 0.19, [100 * g[m] for m in ms], 0.36, color=col, edgecolor="k", lw=0.4, label=label)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([{"gauss": "Gaussian", "native": "Prophet", "split": "Split", "aci": "ACI"}[m] for m in ms])
        ax.axhline(80, color="r", ls="--", lw=0.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("empirical coverage (%)")
        ax.set_title(("Return" if t == "return" else "Volatility") + " (nominal 80%)")
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage.pdf")
    plt.close(fig)

    if cum_data is not None:
        c = cum_data.copy()
        c["bm_minus_cand"] = c["loss_benchmark"] - c["loss_candidate"]
        c["bm_minus_guard"] = c["loss_benchmark"] - c["loss_guarded_rolling"]
        by = c.groupby("origin_date")[["bm_minus_cand", "bm_minus_guard"]].mean().sort_index().cumsum()
        fig, ax = plt.subplots(figsize=(3.45, 2.2))
        ax.plot(by.index, by["bm_minus_cand"], color="#222222", lw=1.0, label="EWMA always on")
        ax.plot(by.index, by["bm_minus_guard"], color="#888888", lw=1.0, ls="--", label="EWMA + guard")
        for name, (a, b) in STRESS_WINDOWS.items():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#dddddd", alpha=0.7, lw=0)
        ax.axvline(pd.Timestamp(cfg.holdout_start), color="r", lw=0.7, ls=":")
        ax.set_ylabel("cumulative QLIKE gain\nover HistVol")
        ax.legend(frameon=False, fontsize=6)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUT / "fig_cumulative.pdf")
        plt.close(fig)
        fa = first_alarm_dates(cum_data, STRESS_WINDOWS["covid_2020"])
        num("CovidAlarmN", len(fa))
        num("CovidAlarmMedian", str(fa.median().date()) if len(fa) else "n/a")

    write("numbers.tex", "\n".join(f"\\newcommand{{\\num{k}}}{{{v}}}" for k, v in sorted(NUMS.items())) + "\n")
    print(f"wrote {len(list(OUT.glob('*')))} files to {OUT}")
    for t in RUNS:
        print(t, "gate confusion", confusion(data[t]["val"]), "naive-selection", confusion(data[t]["val"], "naive_outcome"))


if __name__ == "__main__":
    main()
