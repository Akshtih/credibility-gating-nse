# stock-credibility

Credibility gating for stock forecasts on Indian markets (NSE). A model serves forecasts only if it beats a naive
benchmark out of sample (Diebold–Mariano, Holm-corrected) with calibrated adaptive-conformal intervals; otherwise the
benchmark is served and labelled. Includes a frozen data snapshot, a leakage-guarded backtester with a one-time
holdout, a model registry, a serving API + dashboard, and a degradation monitor with benchmark fallback.
Research code; not investment advice.

Start with [`docs/HANDOFF.md`](docs/HANDOFF.md) for status, results, decisions and next steps.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m credibility.cli serve forecast --symbol TCS.NS --task volatility --horizon 21
.venv/Scripts/python.exe -m credibility.app        # dashboard at http://127.0.0.1:5001
```

Paper: [`paper/main.tex`](paper/main.tex) (IEEEtran). All its numbers/tables/figures come from
`scripts/make_artifacts.py`.
