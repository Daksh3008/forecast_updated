import yfinance as yf
import pandas as pd

FRED_SERIES = {
    "us10y": "^TNX",      # proxy via Yahoo
}

YAHOO_TICKERS = {
    "brent": "BZ=F",
    "wti": "CL=F",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "vix": "^VIX",
}


def _latest_close(ticker):
    df = yf.download(ticker, period="10d", interval="1d", progress=False)
    df = df.dropna()

    # strict scalar extraction
    return float(df["Close"].iloc[-1].item())


def fetch_macro_snapshot():
    data = {}

    for k, t in YAHOO_TICKERS.items():
        try:
            data[k] = _latest_close(t)
        except Exception:
            data[k] = None

    # US 10Y yield proxy
    try:
        data["us10y"] = _latest_close(FRED_SERIES["us10y"]) / 10.0
    except Exception:
        data["us10y"] = None

    return data
