#!/bin/bash
set -e

echo "======================================"
echo " DB-backed Full Pipeline"
echo "======================================"

echo "Initializing database..."
python -m db.init_db

echo "Fetching and storing market data..."
python -m db.write_daily_prices

echo "Building feature matrix (DB-backed)..."
python -m data_pipeline.prepare_features

echo "Running one-shot backtest..."
python -m one_shot_backtest

echo "Launching forecast agent..."
python -m forecaster.run_forecast
