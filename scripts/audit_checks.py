"""Independent audit of the paper's numbers (exploratory; NOT used to produce any paper number).

A. Pipeline integrity: re-derive stored predictions/outcomes with separate, plain code.
B. Aggregation: recompute every skill value with plain pandas and compare with the stored summaries.
C. Robustness: do conclusions survive outliers, the COVID window, a different loss, and stock-level resampling?
"""
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credibility.config import load_eval_config  # noqa: E402
from credibility.data import Snapshot  # noqa: E402
from credibility.evaluation import get_task  # noqa: E402
from credibility.models import make_model  # noqa: E402

cfg = load_eval_config()
snap = Snapshot.open(cfg.snapshot)
runs = {("return", "dev"): "ret_dev", ("return", "hold"): "ret_hold", ("volatility", "dev"): "vol_dev", ("volatility", "hold"): "vol_hold"}
P = {k: pd.read_parquet(cfg.results_path / v / "predictions.parquet") for k, v in runs.items()}

print("=== A. pipeline integrity (independent re-derivation) ===")
sym = "RELIANCE.NS"
df = snap.load(sym)
close = df["close"].to_numpy(float)
lc = np.log(close)
r = np.diff(lc)                                     # r[j] = lc[j+1]-lc[j]
for (task, period), p in [(("return", "dev"), P[("return", "dev")]), (("volatility", "dev"), P[("volatility", "dev")])]:
    i = int(p[p.symbol == sym]["origin_idx"].iloc[40])
    for h in cfg.horizons if task == "return" else cfg.vol_horizons:
        stored = p[(p.symbol == sym) & (p.origin_idx == i) & (p.horizon == h)].set_index("model")
        if task == "return":
            act = lc[i + h] - lc[i]
            ok = np.isclose(stored["actual"].iloc[0], act)
            ar = make_model("arima").fit(df.iloc[: i + 1]).predict(h)
            print(f" return  h={h:2d} origin {i}: actual match={ok} naive_pred={stored.loc['naive','pred']} "
                  f"arima refit match={np.isclose(ar, stored.loc['arima','pred'])}")
        else:
            act = float(np.mean(r[i: i + h] ** 2))
            hv = float(np.mean(r[i - 21: i] ** 2))                    # last 21 returns up to and including row i
            v = float(np.mean(r[:30] ** 2)) if False else None
            ew = make_model("ewma").fit(df.iloc[: i + 1]).predict(h)
            print(f" vol     h={h:2d} origin {i}: actual match={np.isclose(stored['actual'].iloc[0], act)} "
                  f"hist_vol match={np.isclose(hv, stored.loc['hist_vol','pred'])} ewma refit match={np.isclose(ew, stored.loc['ewma','pred'])}")

print("\n=== B. aggregation: skill recomputed with plain pandas vs stored summary ===")
EPS = 1e-10


def loss(task, pred, act):
    if task == "return":
        return (act - pred) ** 2
    p_, a_ = np.maximum(pred, EPS), np.maximum(act, EPS)
    return np.log(p_ / a_) + a_ / p_ - 1


worst = 0.0
for (task, period), p in P.items():
    bm = cfg.benchmark_model if task == "return" else cfg.vol_benchmark
    b = p[p.model == bm].set_index(["symbol", "origin_idx", "horizon"])["pred"]
    summ = pd.read_csv(ROOT / "registry" / "evidence" / task / "summary.csv") if period == "dev" else None
    for (m, h), g in p[p.model != bm].groupby(["model", "horizon"]):
        pb = b.reindex(pd.MultiIndex.from_frame(g[["symbol", "origin_idx", "horizon"]])).to_numpy()
        ok = np.isfinite(g["pred"].to_numpy()) & np.isfinite(pb)
        sk = 1 - loss(task, g["pred"].to_numpy()[ok], g["actual"].to_numpy()[ok]).mean() / loss(task, pb[ok], g["actual"].to_numpy()[ok]).mean()
        if summ is not None:
            ref = float(summ[(summ.model == m) & (summ.horizon == h)]["skill"].iloc[0])
            worst = max(worst, abs(sk - ref))
print(f" max |skill_recomputed - skill_stored| over all dev cells: {worst:.2e}")

print("\n=== C. robustness of the surprising results ===")
covid = (pd.Timestamp("2020-02-15"), pd.Timestamp("2020-06-30"))


def skill_cell(task, p, bm, m, h, mask_fn=None, lossfn=None):
    lf = lossfn or (lambda pr, ac: loss(task, pr, ac))
    g = p[(p.model == m) & (p.horizon == h)]
    b = p[(p.model == bm) & (p.horizon == h)][["symbol", "origin_idx", "pred"]].rename(columns={"pred": "pb"})
    g = g.merge(b, on=["symbol", "origin_idx"]).dropna(subset=["pred", "pb", "actual"])
    if mask_fn is not None:
        g = g[mask_fn(g)]
    return 1 - lf(g["pred"].to_numpy(), g["actual"].to_numpy()).mean() / lf(g["pb"].to_numpy(), g["actual"].to_numpy()).mean(), g


rng = np.random.default_rng(0)
print("-- volatility (QLIKE; skill>0 = beats HistVol)")
print(f"{'period':5s}{'model':6s}{'h':>3s} {'skill':>7s} {'ex-COVID':>9s} {'trim5%':>7s} {'MSE(sd)':>8s} {'dates won':>9s} {'stock-boot 95% CI':>20s}")
for period in ("dev", "hold"):
    p = P[("volatility", period)]
    tiny = int((p["actual"] < 1e-8).sum())
    for m in ("ewma", "garch", "har"):
        for h in (5, 21):
            sk, g = skill_cell("volatility", p, "hist_vol", m, h)
            skx, _ = skill_cell("volatility", p, "hist_vol", m, h, lambda g: ~g["origin_date"].between(*covid))
            d = loss("volatility", g["pred"].to_numpy(), g["actual"].to_numpy()) - loss("volatility", g["pb"].to_numpy(), g["actual"].to_numpy())
            lb = loss("volatility", g["pb"].to_numpy(), g["actual"].to_numpy())
            keep = lb <= np.quantile(lb, 0.95)
            skt = 1 - (loss("volatility", g["pred"].to_numpy(), g["actual"].to_numpy())[keep].mean() / lb[keep].mean())
            sd = lambda pr, ac: (np.sqrt(np.maximum(ac, 0)) - np.sqrt(np.maximum(pr, 0))) ** 2
            skm, _ = skill_cell("volatility", p, "hist_vol", m, h, lossfn=sd)
            won = float((g.assign(d=d).groupby("origin_date")["d"].mean() < 0).mean())
            syms = g["symbol"].unique()
            boots = []
            for _ in range(1000):
                pick = rng.choice(syms, len(syms))
                gg = pd.concat([g[g.symbol == s] for s in pick])
                boots.append(1 - loss("volatility", gg["pred"].to_numpy(), gg["actual"].to_numpy()).mean() / loss("volatility", gg["pb"].to_numpy(), gg["actual"].to_numpy()).mean())
            lo, hi = np.quantile(boots, [0.025, 0.975])
            print(f"{period:5s}{m:6s}{h:3d} {sk:+7.3f} {skx:+9.3f} {skt:+7.3f} {skm:+8.3f} {won:9.2f} [{lo:+.3f}, {hi:+.3f}]")
    print(f"      (actual < 1e-8 in {period}: {tiny} rows)")

print("-- return (squared error; skill>0 = beats naive)")
print(f"{'period':5s}{'model':17s}{'h':>3s} {'skill':>8s} {'ex-COVID':>9s} {'MAE ratio':>10s} {'dates won':>9s}")
for period in ("dev", "hold"):
    p = P[("return", period)]
    for m in ("arima", "prophet_anchored", "xgboost"):
        for h in (1, 5, 21):
            sk, g = skill_cell("return", p, "naive", m, h)
            skx, _ = skill_cell("return", p, "naive", m, h, lambda g: ~g["origin_date"].between(*covid))
            mae = (g["actual"] - g["pred"]).abs().mean() / g["actual"].abs().mean()
            d = (g["actual"] - g["pred"]) ** 2 - g["actual"] ** 2
            won = float((g.assign(d=d).groupby("origin_date")["d"].mean() < 0).mean())
            print(f"{period:5s}{m:17s}{h:3d} {sk:+8.3f} {skx:+9.3f} {mae:10.3f} {won:9.2f}")
