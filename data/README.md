# Data Directory

This folder stores the raw data files. They are **not tracked by Git** due to their size.

## How to obtain the data

The data is downloaded automatically when you run:

```python
from src.data_pipeline import load_data
returns = load_data()
```

This downloads daily closing prices for 23 S&P 500 tickers from Yahoo Finance (starting 1945-01-01) and computes log-returns.

## Expected output

- `returns` DataFrame: ~16,181 rows × 23 columns
- Each column is the daily log-return of one asset
