"""Many-candidates artifacts for the paper (called by make_extra_artifacts.py). Reads results/many_candidates only."""
import json

import pandas as pd

from credibility.manycand import null_check

RULES = ("gate", "raw_dm", "simple", "best_only")
RULE_LABEL = {"gate": "Gate (Holm DM)", "raw_dm": "Unadjusted DM", "simple": "Positive skill", "best_only": "Best single cell"}
TASK_LABEL = {"return": "Return", "volatility": "Volatility"}
TASK_WORD = {"return": "Ret", "volatility": "Vol"}
RULE_WORD = {"gate": "Gate", "raw_dm": "Raw", "simple": "Simple", "best_only": "Best"}


def _ratio_list(sub: pd.DataFrame) -> str:
    parts = [f"{int(r.n_fail)}/{int(r.n_promoted)}" for r in sub.itertuples()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def many_artifacts(cfg, OUT, num, f) -> None:
    d = cfg.results_path / "many_candidates"
    s = pd.read_csv(d / "rules_summary.csv")
    build = json.loads((d / "build.json").read_text(encoding="utf-8"))
    rows = []
    for task in ("return", "volatility"):
        t = s[s["task"] == task]
        num(f"Many{TASK_WORD[task]}Cells", int(t["n_cells"].max()))
        b = build["tasks"][task]
        num(f"Many{TASK_WORD[task]}New", b["n_new_candidates"])
        num(f"Many{TASK_WORD[task]}Cands", b["n_new_candidates"] + b["n_original_candidates"])
        for rule in RULES:
            r = t[t["rule"] == rule].sort_values("split")
            prom, fail = int(r["n_promoted"].sum()), int(r["n_fail"].sum())
            wskill = float((r["mean_skill_V"].fillna(0) * r["n_promoted"]).sum() / prom) if prom else float("nan")
            rows.append(f"{TASK_LABEL[task]} & {RULE_LABEL[rule]} & {prom} & {fail} & "
                        f"{('%.0f' % (100 * fail / prom)) if prom else '--'} & {f(wskill)} \\\\")
            num(f"Many{TASK_WORD[task]}{RULE_WORD[rule]}Ratio", _ratio_list(r))
            lo, hi = int(r["n_promoted"].min()), int(r["n_promoted"].max())
            num(f"Many{TASK_WORD[task]}{RULE_WORD[rule]}Range", f"{lo}" if lo == hi else f"{lo}--{hi}")
    (OUT / "table_many.tex").write_text(
        "\\begin{tabular}{llrrrr}\n\\toprule\nTask & Selection rule & Promoted & Failed & Failed (\\%) & Mean val.\\ skill \\\\\n\\midrule\n"
        + "\n".join(rows[:4]) + "\n\\midrule\n" + "\n".join(rows[4:]) + "\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
    nc = null_check(reps=200, seed=0)
    num("ManyNullAny", f"{100 * nc['gate_any_rate']:.1f}")
    num("ManyNullSimpleCells", f"{100 * nc['simple_cell_rate']:.0f}")
    num("ManyNullSimpleFail", f"{100 * nc['simple_fail_share']:.0f}")
    num("ManyNullReps", nc["reps"])
