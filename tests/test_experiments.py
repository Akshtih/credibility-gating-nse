import numpy as np
import pandas as pd

from credibility.evaluation import get_task
from credibility.experiments import confusion, gate_validation, holdout_coverage

from test_uncertainty import make_preds
from test_volatility import cfg as vol_cfg


def test_gate_validation_outcomes_and_naive_selection():
    dev = pd.DataFrame({"task": "return", "model": ["a", "b", "c", "d"], "horizon": 5,
                        "skill": [0.05, 0.04, -0.1, 0.02], "dm_p_adj": [0.001, 0.2, 1.0, 0.3],
                        "promote": [True, False, False, False], "beats_benchmark": [True, False, False, False],
                        "calibrated": True})
    hold = pd.DataFrame({"model": ["a", "b", "c", "d"], "horizon": 5,
                         "skill": [0.03, -0.02, -0.2, 0.05], "dm_p_adj": [0.01, 0.9, 1.0, 0.001]})
    v = gate_validation(dev, hold).set_index("model")
    assert v.loc["a", "outcome"] == "TP"          # promoted, holds
    assert v.loc["b", "outcome"] == "TN"          # rejected, indeed fails
    assert v.loc["c", "outcome"] == "TN"
    assert v.loc["d", "outcome"] == "FN"          # rejected on dev but beats on holdout
    # naive selection (dev skill>0) wrongly picks b (fails on holdout) -> FP, which the gate avoided
    assert v.loc["b", "naive_outcome"] == "FP" and v.loc["a", "naive_outcome"] == "TP"
    assert confusion(v.reset_index()) == {"TP": 1, "FP": 0, "TN": 2, "FN": 1}
    assert confusion(v.reset_index(), "naive_outcome")["FP"] == 1


def test_holdout_coverage_scores_only_holdout_rows():
    task = get_task(vol_cfg(), "return")
    full = make_preds(n=600, seed=5)
    cut = full["origin_date"].iloc[400]
    dev, hold = full[full["origin_date"] < cut], full[full["origin_date"] >= cut]
    t = holdout_coverage(dev, hold, task, str(cut.date()))
    aci = t[t["method"] == "aci"].iloc[0]
    assert 0.7 < aci["coverage"] < 0.9
    assert aci["n"] <= len(hold)
