# Preliminary pilot (motivation only; not part of the main study)

`backtest.py` ran Prophet (the earlier prototype's settings, price levels) against naive and ARIMA(1,1,0) baselines on
5 US + 5 NSE stocks: 2 years of daily closes from Yahoo Finance, six expanding-window origins 30 trading days apart per stock, 30-day horizon.
`backtest_rows.csv` / `backtest_summary.csv` are the stored outputs quoted in the paper's introduction
(in-sample vs out-of-sample MAPE, naive MAPE, directional accuracy, interval coverage). The script contains absolute paths from the machine it
was run on and fetches live Yahoo data, so a re-run will not reproduce the numbers exactly; the stored CSVs are the record.
