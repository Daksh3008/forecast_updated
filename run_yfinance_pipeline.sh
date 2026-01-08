#!/bin/bash
set -e

echo "======================================"
echo " CSV/Yahoo-backed Pipeline (No DB)"
echo "======================================"

echo "Downloading market data from Yahoo..."
python -m scripts.download_data

echo "Building feature matrix (CSV-backed)..."
python -m scripts.prepare_features

echo "Running one-shot backtest..."
python -m one_shot_backtest

echo "Launching forecast agent..."
python -m forecaster.run_forecast
