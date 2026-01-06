# db/read_daily_prices.py
import pandas as pd
from db.connection import get_conn

def load_asset(asset):
    sql = """
    SELECT date, open, high, low, close, volume
    FROM daily_prices
    WHERE asset = %s
    ORDER BY date
    """
    conn = get_conn()
    df = pd.read_sql(sql, conn, params=(asset,))
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")
