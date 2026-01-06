# scripts/train_ridge.py

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TARGET = "log_ret"


# ---------------------------------------------------------
# Build X, y for one-step-ahead regression
# ---------------------------------------------------------

def _build_xy(df, feature_cols):
    X = df[feature_cols].values
    y = df[TARGET].shift(-1).values

    return X[:-1], y[:-1]


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_and_get_model(train_df, params):

    if TARGET not in train_df.columns:
        raise ValueError(f"{TARGET} missing")

    feature_cols = [c for c in train_df.columns if c != TARGET]

    X, y = _build_xy(train_df, feature_cols)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    alpha = float(params.get("alpha", 1.0))
    model = Ridge(alpha=alpha)
    model.fit(Xs, y)

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------
# STRICTLY CAUSAL RECURSIVE RIDGE
# ---------------------------------------------------------

def predict_recursive(model_bundle, full_df, start_price, predict_dates):
    """
    Strictly causal Ridge recursion.

    - Uses ONLY information available at anchor date
    - Recursively updates price
    - Freezes all non-price features
    """

    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_cols = model_bundle["feature_cols"]

    df = full_df.sort_index()

    # -------- anchor state --------
    first_date = predict_dates[0]
    anchor_pos = df.index.get_loc(first_date) - 1
    if anchor_pos < 0:
        raise ValueError("Not enough history for Ridge prediction")

    # last known feature vector (at anchor date)
    last_row = df.iloc[anchor_pos][feature_cols].astype(float).copy()

    close_idx = feature_cols.index("brent_Close")

    curr_price = float(start_price)
    logrets = []

    for _ in predict_dates:
        # build model input
        x = last_row.values.reshape(1, -1)
        xs = scaler.transform(x)

        # predict next log return
        logret = float(model.predict(xs)[0])
        logrets.append(logret)

        # update price
        curr_price *= np.exp(logret)

        # update ONLY price-derived feature
        last_row.iloc[close_idx] = curr_price
        # all other features remain frozen (causal)

    prices = start_price * np.exp(np.cumsum(logrets))

    return pd.Series(prices, index=predict_dates, name="ridge_pred")
