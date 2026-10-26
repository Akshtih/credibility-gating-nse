"""Many-candidates experiment runner.   python scripts/run_many_candidates.py build | evaluate

build     create the extra candidates' forecasts for the stored DEVELOPMENT runs (holdout never read) and check them
evaluate  apply the pre-registered selection rules on each split and write rules_summary.csv / cells_<task>_<split>.csv
Outputs go to results/many_candidates/ (new directory); no original file is touched.
"""
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credibility.config import CONFIG_DIR, load_eval_config  # noqa: E402
from credibility.evaluation import get_task  # noqa: E402
from credibility.manycand import (apply_rules, evaluate_split, load_protocol, return_candidates, rule_summary,  # noqa: E402
                                  volatility_candidates)

cfg = load_eval_config()
OUT = cfg.results_path / "many_candidates"
RUNS = {"return": "ret_dev", "volatility": "vol_dev"}          # development only
proto = load_protocol()
proto_sha = hashlib.sha256((CONFIG_DIR / "many.yaml").read_bytes()).hexdigest()


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    info = {"protocol_sha256": proto_sha, "tasks": {}}
    for tname, run in RUNS.items():
        task = get_task(cfg, tname)
        base = pd.read_parquet(cfg.results_path / run / "predictions.parquet")
        assert base["target_date"].max() < pd.Timestamp(cfg.holdout_start), "holdout data present in a development run"
        spec = proto["candidates"][tname]
        new = (return_candidates(base, cfg.snapshot, spec) if tname == "return"
               else volatility_candidates(base, cfg.snapshot, spec, task.horizons))
        key_n = int((base["model"] == task.benchmark).sum())
        per = new.groupby("model").size()
        checks = {
            "all_pred_finite": bool(np.isfinite(new["pred"]).all()),
            "every_candidate_covers_every_key": bool((per == key_n).all()),
            "n_new_candidates": int(per.size),
            "n_original_candidates": len(spec["original"]),
            "max_target_date": str(new["target_date"].max().date()),
        }
        if tname == "volatility":
            checks["all_pred_positive"] = bool((new["pred"] > 0).all())
        ok = checks["all_pred_finite"] and checks["every_candidate_covers_every_key"]
        print(f"[{tname}] {checks}  -> {'OK' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)
        new.to_parquet(OUT / f"{tname}_candidates.parquet", index=False)
        info["tasks"][tname] = checks
    (OUT / "build.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def evaluate() -> None:
    rows = []
    for tname, run in RUNS.items():
        task = get_task(cfg, tname)
        base = pd.read_parquet(cfg.results_path / run / "predictions.parquet")
        new = pd.read_parquet(OUT / f"{tname}_candidates.parquet")
        allp = pd.concat([base, new], ignore_index=True)
        keep = set(proto["candidates"][tname]["original"]) | set(new["model"].unique()) | {task.benchmark}
        allp = allp[allp["model"].isin(keep)]
        for split in proto["splits"]:
            t = evaluate_split(allp, task, split, cfg.origin_step, proto["alpha"])
            t.to_csv(OUT / f"cells_{tname}_{split}.csv", index=False)
            s = rule_summary(t, apply_rules(t, proto["alpha"]))
            s.insert(0, "split", split)
            s.insert(0, "task", tname)
            rows.append(s)
            print(f"[{tname} split {split}] cells={len(t)} S-dates~{int(t['n_S'].median())} V-dates~{int(t['n_V'].median())}")
    out = pd.concat(rows, ignore_index=True)
    out["protocol_sha256"] = proto_sha
    out.to_csv(OUT / "rules_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["protocol_sha256", "fail_lo", "fail_hi"]).round(3).to_string(index=False))


if __name__ == "__main__":
    {"build": build, "evaluate": evaluate}[sys.argv[1]]()
