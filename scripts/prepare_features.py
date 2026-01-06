import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from db.read_daily_prices import load_asset
ASSETS = {
    "brent": "BZ=F",
    "wti": "CL=F",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "vix": "^VIX",
}

PARQUET_DIR = "data/parquet"
_lr = LinearRegression()

# ============================================================
# FAST LOADER (PARQUET)
# ============================================================

def fast_load_asset(raw_path, asset, ticker):
    os.makedirs(PARQUET_DIR, exist_ok=True)
    pq = f"{PARQUET_DIR}/{asset}.parquet"
    csv = f"{raw_path}/{asset}.csv"

    if os.path.exists(pq):
        df = pd.read_parquet(pq)
        df.index = pd.to_datetime(df.index)
        return df

    df = pd.read_csv(csv, header=[0,1], index_col=0, dayfirst=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna().sort_index()

    out = pd.DataFrame(index=df.index)
    for f in ["Open","High","Low","Close","Volume"]:
        out[f] = df[(f, ticker)]

    out.to_parquet(pq)
    return out

# ============================================================
# HELPERS
# ============================================================

def ema_dist(close, n):
    ema = close.ewm(span=n, adjust=False).mean()
    return (close - ema) / ema

def ma_ret(ret, n):
    return ret.rolling(n).mean()

def slope(series, n):
    arr = series.values
    out = np.full(len(arr), np.nan)
    for i in range(n, len(arr)):
        w = arr[i-n:i]
        if not np.isfinite(w).all():
            continue
        X = np.arange(n).reshape(-1,1)
        _lr.fit(X, w.reshape(-1,1))
        out[i] = _lr.coef_[0][0]
    return pd.Series(out, index=series.index)

def momentum(close, n):
    return close.pct_change(n)

def roc(close, n):
    return close.diff(n)

def acc(mom):
    return mom.diff()

def obv(close, vol):
    return (np.sign(close.diff()) * vol).fillna(0).cumsum()

# ============================================================
# MAIN BUILDER (PRUNED)
# ============================================================

def build_feature_matrix(raw_path="data"):

    frames = {}

    for asset, ticker in ASSETS.items():
        df = load_asset(asset)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]

        close = df["Close"]
        ret = np.log(close / close.shift(1))

        out = pd.DataFrame(index=df.index)

        # --- PRICE LEVELS ---
        out[f"{asset}_Open"] = df["Open"]
        out[f"{asset}_High"] = df["High"]
        out[f"{asset}_Low"] = df["Low"]
        out[f"{asset}_Close"] = close

        # --- RETURNS ---
        out[f"{asset}_ret"] = ret

        # --- EMA DISTANCES ---
        out[f"{asset}_ema10_dist"] = ema_dist(close, 10)
        out[f"{asset}_ema20_dist"] = ema_dist(close, 20)
        out[f"{asset}_ema50_dist"] = ema_dist(close, 50)

        # --- MOMENTUM ---
        out[f"{asset}_mom_5"] = momentum(close, 5)
        out[f"{asset}_mom_7"] = momentum(close, 7)
        out[f"{asset}_mom_10"] = momentum(close, 10)

        # --- ROC ---
        out[f"{asset}_roc_7"] = roc(close, 7)
        out[f"{asset}_roc_10"] = roc(close, 10)

        # --- ACC ---
        out[f"{asset}_acc_5"] = acc(out[f"{asset}_mom_5"])
        out[f"{asset}_acc_10"] = acc(out[f"{asset}_mom_10"])

        # --- MA RETURNS ---
        out[f"{asset}_ma_ret_5"] = ma_ret(ret, 5)
        out[f"{asset}_ma_ret_10"] = ma_ret(ret, 10)

        # --- SLOPES ---
        out[f"{asset}_ret_slope_5"] = slope(ret, 5)
        out[f"{asset}_ret_slope_10"] = slope(ret, 10)

        # --- OBV ---
        out[f"{asset}_obv"] = obv(close, df["Volume"])

        frames[asset] = out

    # ========================================================
    # MERGE
    # ========================================================

    full = pd.concat(frames.values(), axis=1).sort_index().ffill()

    # --- TARGET ---
    full["close"] = full["brent_Close"]
    full["log_ret"] = np.log(full["close"] / full["close"].shift(1))

    # 20-day rolling volatility of brent log returns
    full["brent_vol_20"] = full["log_ret"].rolling(20).std()

    # --- SPREAD ---
    full["spread_ratio_dxy"] = full["brent_Close"] / full["dxy_Close"]

    # ========================================================
    # NA HANDLING (STRICT)
    # ========================================================

    na = full.isna()
    full = full.bfill()
    full = full.mask(na, full * 0.001).fillna(1e-9)

    return full

# ============================================================

def save_feature_matrix(df, path="data/feature_matrix.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"[OK] Saved feature matrix | shape={df.shape}")

if __name__ == "__main__":
    fm = build_feature_matrix()
    save_feature_matrix(fm)
