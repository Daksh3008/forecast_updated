# forecaster/models/train_ridge.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

TARGET = "log_ret"


def train_and_get_model(train_df, params):
    feature_cols = [c for c in train_df.columns if c != TARGET]
    X = train_df[feature_cols].values[:-1]
    y = train_df[TARGET].shift(-1).values[:-1]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = Ridge(alpha=float(params.get("alpha", 1.0)))
    model.fit(Xs, y)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols
    }


def predict_recursive(bundle, full_df, start_price, dates):
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_cols = bundle["feature_cols"]

    df = full_df.sort_index()
    last_row = df.iloc[-1][feature_cols].astype(float).copy()

    close_idx = feature_cols.index("brent_Close")
    price = float(start_price)
    prices = []

    for _ in dates:
        xs = scaler.transform(last_row.values.reshape(1, -1))
        logret = float(model.predict(xs)[0])
        price *= np.exp(logret)
        prices.append(price)
        last_row.iloc[close_idx] = price

    return pd.Series(prices, index=dates, name="ridge_pred")
