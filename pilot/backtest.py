"""Rolling-origin out-of-sample backtest: Prophet (project config) vs naive vs ARIMA.
Uses the project's own StockDataHandler / ProphetPredictor so numbers reflect the shipped code."""
import sys, json, warnings, logging
# (local helper path removed)  extra packages such as statsmodels were installed with pip --target; adjust as needed
sys.path.insert(0, "PATH/TO/earlier-prototype-project")   # the earlier Flask prototype that provided data_handler.py and model_engine.py
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from data_handler import StockDataHandler
from model_engine import ProphetPredictor

TICKERS = ["AAPL","AMZN","GOOGL","MSFT","NVDA","TCS.NS","INFY.NS","ITC.NS","HDFCBANK.NS","BHARTIARTL.NS"]
H = 30            # forecast horizon (trading days ahead, evaluated on calendar-aligned test rows)
N_ORIGINS = 6     # expanding-window origins, spaced H apart, ending at the last H rows
h = StockDataHandler()
rows = []
for t in TICKERS:
    df = h.fetch_data(t, period="2y")          # same window the app uses
    if df.empty: print("skip", t); continue
    df = df.reset_index(drop=True)
    n = len(df)
    for k in range(N_ORIGINS):
        end = n - H * (k)              # test = rows [end-H, end)
        cut = end - H
        if cut < 250: continue
        train, test = df.iloc[:cut], df.iloc[cut:end]
        last = float(train["close"].iloc[-1])
        y = test["close"].values
        # Prophet (project hyper-parameters)
        p = ProphetPredictor()
        pdf = p.preprocess(train); p.train(pdf)
        fut = p.model.predict(pd.DataFrame({"ds": test["date"].values}))
        yp = fut["yhat"].values
        lo, hi = fut["yhat_lower"].values, fut["yhat_upper"].values
        # naive (random walk)
        yn = np.full(H, last)
        # ARIMA(1,1,0) with drift-free random-walk-ish dynamics on closes
        try:
            fit = ARIMA(train["close"].values, order=(1,1,0)).fit()
            ya = np.asarray(fit.forecast(H))
        except Exception as e:
            ya = yn
        def mape(a,b): return float(np.mean(np.abs((a-b)/a))*100)
        def mae(a,b): return float(np.mean(np.abs(a-b)))
        # in-sample (what the app currently reports)
        fi = p.model.predict(pd.DataFrame({"ds": pdf["ds"]}))
        ins = float(np.mean(np.abs((pdf["y"].values-fi["yhat"].values)/pdf["y"].values))*100)
        dir_true = np.sign(y[-1]-last)
        rows.append(dict(ticker=t, origin=k, n_train=cut,
            mape_prophet=mape(y,yp), mape_naive=mape(y,yn), mape_arima=mape(y,ya),
            mae_prophet=mae(y,yp), mae_naive=mae(y,yn), mae_arima=mae(y,ya),
            dir_prophet=float(np.sign(yp[-1]-last)==dir_true),
            dir_arima=float(np.sign(ya[-1]-last)==dir_true),
            dir_up_baseline=float(dir_true>0),
            cover=float(np.mean((y>=lo)&(y<=hi))),
            mape_insample=ins))
    print("done", t, flush=True)
r = pd.DataFrame(rows)
out = "."   # output directory for backtest_rows.csv / backtest_summary.csv
r.to_csv(out + r"\backtest_rows.csv", index=False)
agg = r.drop(columns=["origin","n_train"]).groupby("ticker").mean().round(3)
agg.loc["MEAN"] = r.drop(columns=["ticker","origin","n_train"]).mean().round(3)
agg.to_csv(out + r"\backtest_summary.csv")
print(agg.to_string())
print("windows:", len(r))
