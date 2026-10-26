# Reproducing the paper's results

Everything in the paper's results (tables, figures, in-text numbers) is generated from the stored prediction files in
`results/` by `scripts/make_artifacts.py`. Python 3.10+.

## 1. Environment
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"      # Linux/macOS: .venv/bin/python
```

## 2. Check the data and the code
```bash
python -m pytest -q                                   # 123 tests (leakage guards, purging, holdout lock, statistics)
python -m credibility.cli data verify                 # re-hash every file of the frozen snapshot nse_v1
python -m credibility.cli data report                 # per-stock quality table (20 stocks + Nifty 50 index)
```
The snapshot (`data/snapshots/nse_v1`) was retrieved once (see `manifest.json`, `created_utc`) from Yahoo Finance via
`yfinance` (split/dividend-adjusted daily OHLCV, 2016-01-01..2026-09-30). The stock list is in `configs/data.yaml`.

## 3. Regenerate every table and figure of the paper (minutes)
```bash
python scripts/make_artifacts.py                      # writes paper/generated/*.tex, *.pdf
python scripts/check_paper.py                         # static checks of paper/main.tex
```

## 4. Re-run the experiments (hours; optional)
```bash
python -m credibility.cli eval run --period dev --task return     --name ret_dev2
python -m credibility.cli eval run --period dev --task volatility --name vol_dev2
python -m credibility.cli gate build --ret-run ret_dev2 --vol-run vol_dev2 --registry registry2
```
The stored runs used to write the paper are `ret_dev`, `vol_dev`, `ret_hold`, `vol_hold`.
**Holdout runs are locked** (`results/holdout_<task>.lock`): the holdout is executed once per task by design, so the
stored `ret_hold` / `vol_hold` are the only holdout results. To repeat a holdout run, work on a copy and delete the lock.
The evaluation protocol is in `configs/eval.yaml` (origin step 10 days, 756-day burn-in, holdout from 2024-01-01).

## 5. Layout
`src/credibility/` code · `configs/` protocol and universe · `data/` frozen snapshot · `results/` stored predictions ·
`registry/` gate decisions and evidence · `scripts/` artifact generation · `tests/` test suite.

## 6. Robustness checks and the many-candidates experiment (added after the main study)
```bash
python scripts/protect_check.py --verify               # original study files unchanged (SHA-256 baseline in protected_hashes.json)
python scripts/run_many_candidates.py build            # 1. builds the extra candidates from the DEVELOPMENT runs only (minutes)
python scripts/run_many_candidates.py evaluate         # 2. applies the pre-specified rules on three splits
python scripts/make_artifacts.py                       # 3. main tables, figures, numbers
python scripts/make_extra_artifacts.py                 # 4. robustness, test-size, calibration breakdown, many-candidates table
```
Steps 1-2 are optional if you only want to regenerate the paper artifacts from the stored `results/many_candidates/` files.
`configs/many.yaml` is the protocol, written before any candidate was evaluated; its SHA-256 is stored in
`results/many_candidates/rules_summary.csv`. These steps write only new files; the holdout is not read.
`scripts/audit_checks.py` / `audit_checks2.py` are exploratory audits whose output is not used in the paper.

## 7. Package contents checklist (for reviewers)
| Item | Where |
|---|---|
| Source code | `src/credibility/` |
| Dependency versions | `requirements.lock.txt` (pinned), `pyproject.toml` |
| Exact configuration of every result | `configs/data.yaml`, `configs/eval.yaml`, `configs/many.yaml` (pre-specified many-candidates protocol) |
| Frozen data + hashes + provenance | `data/snapshots/nse_v1/`, `DATA_NOTICE.md` |
| Stored forecasts and realised outcomes | `results/{ret_dev,vol_dev,ret_hold,vol_hold,many_candidates}/` |
| Gate decisions and evidence | `registry/` |
| Scripts that regenerate every table, figure and number | `scripts/make_artifacts.py`, `scripts/make_extra_artifacts.py` |
| Tests (leakage guards, purging, holdout lock, statistics, selection rules) | `tests/` (run `python -m pytest -q`) |
| Check that the original study files are unchanged | `scripts/protect_check.py --verify` |
| Many-candidates experiment | section 6 above; `scripts/run_many_candidates.py` |

## 8. Rebuilding the zips
```bash
python scripts/make_package.py
```
Builds `stock-credibility_reproducibility.zip` and `paper_overleaf.zip` and scans them for personal paths and key-like text.
