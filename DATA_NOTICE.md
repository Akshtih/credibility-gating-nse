# Data provenance and redistribution

* **Source.** Daily split- and dividend-adjusted OHLCV downloaded once from Yahoo Finance through the `yfinance` library
  (retrieval date in `data/snapshots/nse_v1/manifest.json`, `created_utc`), 2016-01-01 .. 2026-09-30, for the 20 stocks in
  `configs/data.yaml` plus the Nifty 50 index (regime labels only). The universe was chosen by hand (current large caps with a full
  history since 2016), which introduces survivorship bias; see the paper.
* **Integrity.** `python -m credibility.cli data verify` re-hashes every series against the SHA-256 values in the manifest.
* **Redistribution.** Yahoo Finance data are subject to Yahoo's terms of use. Before making this package public, confirm that
  redistributing the price files is permitted for your purpose. If it is not, delete `data/snapshots/nse_v1/prices/` and keep the manifest
  (hashes and quality reports). The paper's tables and figures can still be regenerated from the stored forecasts in `results/`
  (`python scripts/make_artifacts.py`), but re-downloading the prices later will NOT reproduce the hashes, because adjusted prices change
  whenever a new dividend or split is applied.
* **What the stored forecasts contain.** `results/<run>/predictions.parquet`: forecast, realised outcome, dates, stock and model for every
  forecast origin; this is derived data, not vendor data.
