# one_shot_backtest.py

import os, warnings
import pandas as pd
import numpy as np

from scripts.train_lstm import train_and_get_model as train_lstm, predict_recursive as lstm_predict
from scripts.train_tcn import train_and_get_model as train_tcn, predict_recursive as tcn_predict
from scripts.train_ridge import train_and_get_model as train_ridge, predict_recursive as ridge_predict
from scripts.ensemble import fit_ridge_weights, predict_with_weights

warnings.filterwarnings("ignore")

PRED_HORIZON = 22
RETRAIN_EVERY = 10
CLOSE = "brent_Close"

def load_feature_matrix():
    df = pd.read_csv("data/feature_matrix.csv", index_col=0, parse_dates=True)
    return df

def run_one_shot_backtest():
    df = load_feature_matrix()
    results = []

    anchors = df.loc["2023-12-31":"2025-10-31"].index
    last_models = None

    for i, anchor in enumerate(anchors):
        retrain = (i % RETRAIN_EVERY == 0)

        print(f"[ANCHOR] {anchor.date()} | retrain={retrain}")

        idx = df.index.get_loc(anchor)
        if idx + PRED_HORIZON >= len(df):
            continue

        train_df = df.iloc[:idx+1]
        future = df.index[idx+1:idx+1+PRED_HORIZON]
        start_price = train_df[CLOSE].iloc[-1]

        if retrain or last_models is None:
            lstm = train_lstm(train_df, {
                "seq_len":5,"layers":3,"hidden_dim":64,
                "lr":0.0005,"batch_size":32,"epochs":50,"dropout":0.1,"early_stopping":15
            })
            tcn = train_tcn(train_df, {
                "seq_len":7,"channels":32,"layers":4,
                "lr":1e-3,"epochs":40,"batch_size":64,"dropout":0.1
            })
            ridge = train_ridge(train_df, {"alpha":4.0})
            last_models = (lstm,tcn,ridge)

        lstm,tcn,ridge = last_models

        p_lstm = lstm_predict(lstm, df, start_price, future)
        mu = tcn_predict(tcn, df, start_price, future)
        p_tcn = start_price * np.exp(np.cumsum(mu.values))
        p_tcn = pd.Series(p_tcn, index=future)

        p_ridge = ridge_predict(ridge, df, start_price, future)

        preds = {"lstm":p_lstm,"tcn":p_tcn,"ridge":p_ridge}
        w = fit_ridge_weights(preds, df[CLOSE])
        ens = predict_with_weights(preds, w)

        target_date = future[-1]

        lstm_last = p_lstm.iloc[-1]
        tcn_last = p_tcn.iloc[-1]
        ridge_last = p_ridge.iloc[-1]
        ensemble_last = ens.iloc[-1]
        actual_last = df.loc[target_date, CLOSE]

        pct_diff = (ensemble_last - actual_last) / actual_last * 100

        results.append({
            "anchor_date": anchor,
            "target_date": target_date,
            "lstm_pred": lstm_last,
            "tcn_pred": tcn_last,
            "ridge_pred": ridge_last,
            "ensemble_pred": ensemble_last,
            "actual_price": actual_last,
            "pct_diff_ensemble": pct_diff
        })


    out = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    out.to_csv("outputs/one_shot_backtest.csv", index=False)
    print("[DONE] Backtest finished")

if __name__ == "__main__":
    run_one_shot_backtest()
