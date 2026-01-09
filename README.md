# Brent Forecasting – Rolling Backtest

Rolling monthly forecasts for Brent Crude using:
- LSTM + Attention
- TCN + Attention
- Ridge Regression
- Linear Ensemble

Daily data:
- Brent (close target)
- WTI, DXY, USD/INR, VIX
- Yahoo Finance primary
- FRED secondary

Target:
recursive predictions for t+22 days(22trading days =  1 month)

Output for backtest:
one year backtest results with lstm, tcn, ridge, ensemble predictions and %difference between actual and ensemble


# Brent Crude Rolling Backtester
A quantitative research framework for recursively forecasting Brent crude prices using deep learning and structured ensemble methods.

The system performs **yearly rolling backtests**:
- Train on historical market data up to date t
- predict next date log returns and feed it again and predict for t+2, keep doing this for t+22 days 
- convert final predicted log returns to price
- Ensemble the models using historical NNLS weights (no look-ahead)

This engine is designed for **robust, leakage-free forecasting** suitable for production quant research.

---

## ✔ Current Models

### 1. LSTM + Attention
- Recursive forecasting of daily log-returns
- Multi-step prediction via sequence roll-forward

### 2. Temporal Convolutional Network (TCN)
- Causal convolutions
- Attention head 
 
### 3. Ridge Regression (coming next)
Planned addition:
- Monthly horizon drift predictor
- Low-variance baseline component for ensemble stability

---

## ✔ Ensemble Logic

### Current Method:
- NNLS non-negative weighting
- Fit using **previous month actuals**
- Apply weights to next month predictions

### Upcoming Enhancements:
- 12-month exponentially-decayed weight memory
- Ridge-constrained non-negative regression
- Monthly regime-based prior weighting

**No future data is ever used.**

---

Models run on GPU (PyTorch Lightning)

COMMANDS AND OUTPUT LOCATION
1. download yfinance data
python -m scripts.download_data
output to data/

if we want to use db as a datasource
python -m db.init_db
python -m db.write_daily_prices

2. prepare features
python -m scripts.prepare_features
output to data/feature_matrix.csv

if datasource is db then feature matrix will be made using
python -m data_pipeline.prepare_features

3. training models individually (optional, models will be called internally)
python -m scripts.train_lstm
python -m scripts.train_tcn
python -m scripts.train_ridge

4. Backtesting (Also used for confidence bands so run once before running predict.py)
python -m one_shot_backtest
outputs to output/one_shot_backtest.csv

5. shap ridge analysis
python -m analysis.shap_ridge
outputs to analysis/shap_feature_ranking.csv

6. Run agent/Predict
(sample inputs:
    predict crude after 2 months
                        Q1 2026
                        next quarter
                        January 2026
                        after 10 days
)
python -m forecaster.run_forecast
outputs to:
outputs/forecasts/brent_YYYYMMDD.txt
outputs/forecasts/brent_YYYYMMDD.json


-------------------------------------------------------------------------
Now run only script files
1. database pipeline
chmod +x run_db_pipeline.sh
./run_db_pipeline.sh

2. yfinance pipeline
chmod +x run_yfinance_pipeline.sh
./run_csv_pipeline.sh
-------------------------------------------------------------------------
Steps to follow:

for db pipeline: 
docker compose up -d
docker compose ps 
docker compose exec app bash 
chmod +x run_db_pipeline.sh 
./run_db_pipeline.sh 
-----------------------------------------------------------
for yfinance pipeline: 
docker compose up -d 
docker compose ps 
docker compose exec app bash 

chmod +x run_yfinance_pipeline.sh 
./run_yfinance_pipeline.sh