# data_pipeline/fetch_yahoo.py
import yfinance as yf
import pandas as pd

def fetch_asset(ticker, start="2007-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=False)
    df = df.rename(columns=str.lower)
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df.dropna()
