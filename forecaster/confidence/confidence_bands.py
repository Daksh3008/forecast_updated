import numpy as np
import pandas as pd


def estimate_residual_sigma(backtest_csv, horizon=22):
    """
    Estimate residual std from historical one-shot backtest.
    """
    df = pd.read_csv(backtest_csv, parse_dates=["target_date"])
    resid = df["ensemble_pred"] - df["actual_price"]
    return resid.std()


def confidence_bands(
    forecast_price,
    sigma,
    horizon_days,
    base_horizon=22
):
    scale = np.sqrt(horizon_days / base_horizon)

    band_68 = sigma * scale
    band_95 = 1.96 * sigma * scale

    return {
        "68%": (forecast_price - band_68, forecast_price + band_68),
        "95%": (forecast_price - band_95, forecast_price + band_95),
    }
