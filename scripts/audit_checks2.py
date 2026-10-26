"""Second audit: (1) fairer outlier check for volatility skill, (2) size (false-positive rate) of the DM implementation."""
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credibility.config import load_eval_config  # noqa: E402
from credibility.evaluation.stats import dm_test  # noqa: E402

cfg = load_eval_config()
EPS = 1e-10
q = lambda p, a: np.log(np.maximum(p, EPS) / np.maximum(a, EPS)) + np.maximum(a, EPS) / np.maximum(p, EPS) - 1

print("=== 1. volatility skill after removing the 5% of rows with the HIGHEST REALISED variance (selection on the outcome,")
print("       which treats model and benchmark symmetrically, unlike trimming on the benchmark's own loss) ===")
for period, run in (("dev", "vol_dev"), ("hold", "vol_hold")):
    p = pd.read_parquet(cfg.results_path / run / "predictions.parquet")
    b = p[p.model == "hist_vol"][["symbol", "origin_idx", "horizon", "pred"]].rename(columns={"pred": "pb"})
    for m in ("ewma", "garch", "har"):
        row = []
        for h in (5, 21):
            g = p[(p.model == m) & (p.horizon == h)].merge(b, on=["symbol", "origin_idx", "horizon"])
            full = 1 - q(g.pred, g.actual).mean() / q(g.pb, g.actual).mean()
            k = g.actual <= g.actual.quantile(0.95)
            trim = 1 - q(g.pred[k], g.actual[k]).mean() / q(g.pb[k], g.actual[k]).mean()
            row.append(f"h={h}: full {full:+.3f} -> ex-top-5%-outcomes {trim:+.3f}")
        print(f" {period:4s} {m:6s} " + " | ".join(row))

print("\n=== 2. size of the DM implementation under a TRUE null with overlapping-horizon (MA) dependence ===")
rng = np.random.default_rng(1)
for n in (68, 122):
    for lag, label in ((0, "iid, lag 0"), (2, "MA(2) overlap, lag 2")):
        rej = {0.05: 0, 0.10: 0}
        N = 4000
        for _ in range(N):
            e = rng.normal(0, 1, n + lag)
            d = np.convolve(e, np.ones(lag + 1) / (lag + 1), mode="valid")[:n] if lag else e[:n]
            pv = dm_test(d, np.zeros(n), lag)["p_better"]
            rej[0.05] += pv < 0.05
            rej[0.10] += pv < 0.10
        print(f" n={n:3d} {label:22s} rejection rate at nominal 5%: {rej[0.05] / N:.3f}   at 10%: {rej[0.10] / N:.3f}")
print("   (a correct one-sided test rejects 5% / 10% of the time under the null)")
