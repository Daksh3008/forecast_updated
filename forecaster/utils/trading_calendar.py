# forecaster/utils/trading_calendar.py

import pandas as pd


def snap_to_previous_trading_day(target_date, market_index):
    """
    Snap a calendar date to the previous available trading day
    using an existing DatetimeIndex (market data index).

    Args:
        target_date (datetime.date)
        market_index (pd.DatetimeIndex)

    Returns:
        pd.Timestamp
    """
    ts = pd.Timestamp(target_date)

    if ts in market_index:
        return ts

    # walk backwards until we find a trading day
    while ts not in market_index:
        ts -= pd.Timedelta(days=1)

    return ts
