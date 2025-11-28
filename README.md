# Stock Prediction System

A comprehensive system for long-term stock price prediction using advanced machine learning models (LSTM, Transformer, XGBoost, Ensemble).

## Features
- **Data Pipeline**: Fetches minute-level data from Yahoo Finance, handles 7-day limits, and calculates advanced features (Volume in INR, Technical Indicators).
- **Models**: Implements and compares multiple architectures:
  - Bidirectional LSTM with Attention
  - Transformer Encoder
  - XGBoost
  - Ensemble
- **Evaluation**: Tracks Execution Rate (90% target) and Alpha generation against 9:30 AM baseline.
- **Cohort Analysis**: Analyzes performance across Market Cap and Volume segments.

## Installation

```bash
pip install .
```

## Usage

Run the full pipeline:

```bash
python run_pipeline.py
```

Fetch new stock list:

```bash
python src/data/ticker_fetcher.py
```
# stock_prediction
