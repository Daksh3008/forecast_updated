# db/write_daily_prices.py
import pandas as pd
from db.connection import get_conn
from data_pipeline.yahoo_fetch import fetch_asset

ASSETS = {
    "brent": "BZ=F",
    "wti": "CL=F",
    "dxy": "DX-Y.NYB",
    "usdinr": "INR=X",
    "vix": "^VIX",
}

INSERT_SQL = """
INSERT INTO daily_prices (asset, date, open, high, low, close, volume)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (asset, date)
DO UPDATE SET
    open=EXCLUDED.open,
    high=EXCLUDED.high,
    low=EXCLUDED.low,
    close=EXCLUDED.close,
    volume=EXCLUDED.volume;
"""

def run():
    conn = get_conn()
    cur = conn.cursor()

    for asset, ticker in ASSETS.items():
        df = fetch_asset(ticker)
        for d, r in df.iterrows():
            open_   = float(r["open"].iloc[0])   if hasattr(r["open"], "iloc")   else float(r["open"])
            high_   = float(r["high"].iloc[0])   if hasattr(r["high"], "iloc")   else float(r["high"])
            low_    = float(r["low"].iloc[0])    if hasattr(r["low"], "iloc")    else float(r["low"])
            close_  = float(r["close"].iloc[0])  if hasattr(r["close"], "iloc")  else float(r["close"])

            vol_raw = r["volume"]
            volume_ = (
                float(vol_raw.iloc[0])
                if hasattr(vol_raw, "iloc") and not pd.isna(vol_raw.iloc[0])
                else None
            )

            cur.execute(
                INSERT_SQL,
                (
                    asset,
                    d.date(),
                    open_,
                    high_,
                    low_,
                    close_,
                    volume_,
                )
            )


    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    run()
