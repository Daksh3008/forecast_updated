# forecaster/forecast_engine.py

import os
import json
import numpy as np
import pandas as pd
from datetime import date

from forecaster.models.train_lstm import train_and_get_model as train_lstm
from forecaster.models.train_tcn import train_and_get_model as train_tcn
from forecaster.models.train_ridge import train_and_get_model as train_ridge
from forecaster.models.train_lstm import predict_recursive as lstm_predict
from forecaster.models.train_tcn import predict_recursive as tcn_predict
from forecaster.models.train_ridge import predict_recursive as ridge_predict
from forecaster.models.ensemble import fit_ridge_weights, predict_with_weights

from data_pipeline.prepare_features import build_feature_matrix

from forecaster.macro.macro_fetch import fetch_macro_snapshot
from forecaster.macro.macro_context import build_macro_context
from forecaster.news.news_fetch import fetch_news
from forecaster.news.sentiment import sentiment_summary, score_headline
from forecaster.news.summarizer import top_headlines
from forecaster.report.report_builder import build_report
from forecaster.confidence.confidence_bands import (
    estimate_residual_sigma,
    confidence_bands,
)
from forecaster.regime.regime_summary import market_regime


CLOSE_COL = "brent_Close"
OUTPUT_DIR = "outputs/forecasts"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def generate_future_trading_days(anchor_ts: pd.Timestamp, target_ts: pd.Timestamp):
    """
    Generate business days strictly AFTER anchor_ts.
    - target_ts < anchor_ts  -> error
    - target_ts == anchor_ts -> zero-step forecast
    - target_ts > anchor_ts  -> normal recursion
    """
    if target_ts < anchor_ts:
        raise ValueError(
            f"Target date {target_ts.date()} is before last available data date {anchor_ts.date()}"
        )

    if target_ts == anchor_ts:
        return pd.DatetimeIndex([])

    return pd.bdate_range(
        start=anchor_ts + pd.Timedelta(days=1),
        end=target_ts,
        freq="B",
    )


# ---------------------------------------------------------
# Core Engine
# ---------------------------------------------------------

def run_forecast(asset: str, target_date: date) -> dict:
    """
    Fully causal forecast to an externally resolved target date.
    """

    # -----------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------

    target_ts = pd.Timestamp(target_date)

    df = build_feature_matrix()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    anchor_ts = df.index[-1]
    anchor_date = anchor_ts.date()

    # -----------------------------------------------------
    # 2. Determine trading days to predict
    # -----------------------------------------------------

    future_dates = generate_future_trading_days(anchor_ts, target_ts)
    start_price = float(df[CLOSE_COL].iloc[-1])

    # -----------------------------------------------------
    # 3. Train models (UNCHANGED)
    # -----------------------------------------------------

    lstm = train_lstm(df, {
        "seq_len": 5,
        "layers": 3,
        "hidden_dim": 64,
        "lr": 0.0005,
        "batch_size": 32,
        "epochs": 50,
        "dropout": 0.1,
        "early_stopping": 15,
    })

    tcn = train_tcn(df, {
        "seq_len": 7,
        "channels": 32,
        "layers": 4,
        "lr": 1e-3,
        "epochs": 40,
        "batch_size": 64,
        "dropout": 0.1,
    })

    ridge = train_ridge(df, {"alpha": 4.0})

    # -----------------------------------------------------
    # 4. Forecast (ZERO or MULTI step)
    # -----------------------------------------------------

    if len(future_dates) == 0:
        # Same-day forecast (nowcast)
        p_lstm = p_tcn = p_ridge = pd.Series(
            [start_price], index=[anchor_ts]
        )
    else:
        p_lstm = lstm_predict(lstm, df, start_price, future_dates)

        mu_tcn = tcn_predict(tcn, df, start_price, future_dates)
        p_tcn = start_price * np.exp(np.cumsum(mu_tcn.values))
        p_tcn = pd.Series(p_tcn, index=future_dates)

        p_ridge = ridge_predict(ridge, df, start_price, future_dates)

    preds = {
        "lstm": p_lstm,
        "tcn": p_tcn,
        "ridge": p_ridge,
    }

    # -----------------------------------------------------
    # 5. Forecast-safe ensemble (UNCHANGED)
    # -----------------------------------------------------

    hist_end = anchor_ts
    hist_start = df.index[-252] if len(df) > 252 else df.index[0]
    hist_df = df.loc[hist_start:hist_end]

    p_lstm_h = lstm_predict(
        lstm, hist_df, hist_df[CLOSE_COL].iloc[0], hist_df.index[1:]
    )

    mu_tcn_h = tcn_predict(
        tcn, hist_df, hist_df[CLOSE_COL].iloc[0], hist_df.index[1:]
    )

    p_tcn_h = hist_df[CLOSE_COL].iloc[0] * np.exp(np.cumsum(mu_tcn_h.values))
    p_tcn_h = pd.Series(p_tcn_h, index=hist_df.index[1:])

    p_ridge_h = ridge_predict(
        ridge, hist_df, hist_df[CLOSE_COL].iloc[0], hist_df.index[1:]
    )

    preds_hist = {
        "lstm": p_lstm_h,
        "tcn": p_tcn_h,
        "ridge": p_ridge_h,
    }

    weights = fit_ridge_weights(
        preds_hist,
        hist_df.loc[p_lstm_h.index, CLOSE_COL],
    )

    ensemble = predict_with_weights(preds, weights)
    final_price = float(ensemble.iloc[-1])

    # -----------------------------------------------------
    # 6. Context (OUTPUT ONLY)
    # -----------------------------------------------------

    macro = fetch_macro_snapshot()
    macro_lines = build_macro_context(macro)

    news = fetch_news()
    scores = [score_headline(h["title"]) for h in news]
    sentiment = sentiment_summary(news)
    top_news = top_headlines(news, scores, top_k=10)

    model_prices = {
        "lstm": float(p_lstm.iloc[-1]),
        "tcn": float(p_tcn.iloc[-1]),
        "ridge": float(p_ridge.iloc[-1]),
        "ensemble": final_price,
    }

    sigma = estimate_residual_sigma("outputs/one_shot_backtest.csv")

    bands = confidence_bands(
        forecast_price=model_prices["ensemble"],
        sigma=sigma,
        horizon_days=len(future_dates),
    )

    _, regime_text = market_regime(macro, df[CLOSE_COL])

    # -----------------------------------------------------
    # 7. Report + persistence
    # -----------------------------------------------------

    text_report = build_report(
        model_prices=model_prices,
        anchor_date=anchor_date,
        target_date=target_ts.date(),
        weights=weights,
        macro_lines=macro_lines,
        news_lines=top_news,
        sentiment=sentiment,
        bands=bands,
        regime_text=regime_text,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = target_ts.strftime("%Y%m%d")

    txt_path = f"{OUTPUT_DIR}/brent_{tag}.txt"
    json_path = f"{OUTPUT_DIR}/brent_{tag}.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_report)

    with open(json_path, "w") as f:
        json.dump(
            {
                "asset": asset,
                "anchor_date": str(anchor_date),
                "target_date": str(target_ts.date()),
                "forecast_price": final_price,
                "weights": weights,
            },
            f,
            indent=2,
        )

    return {
        "forecast_price": final_price,
        "weights": weights,
        "text_report": text_report,
        "txt_path": txt_path,
        "json_path": json_path,
    }
