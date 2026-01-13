# forecaster/utils/visualizations.py

import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast_path(
    ensemble_series: pd.Series,
    anchor_price: float,
    out_path: str,
):
    """
    Plots forecast price trajectory up to target date.
    """

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # prepend anchor
    idx = [ensemble_series.index[0] - pd.Timedelta(days=1)] + list(ensemble_series.index)
    prices = [anchor_price] + list(ensemble_series.values)

    plt.figure(figsize=(10, 5))
    plt.plot(idx, prices)
    plt.title("Brent Crude Forecast Path")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
