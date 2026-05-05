# Neural Network Forecasting

## Objective

Apply classical neural network architectures (Dense, Recurrent, Convolutional, and Mixed) to financial time series forecasting. Models use historical S&P 500 closing prices (23 assets, since 1960) to predict the average future closing price over different time horizons.

## Data

- **Assets:** 23 S&P 500 tickers (AEP, BA, CAT, CNP, CVX, DIS, DTE, ED, GD, GE, HON, HPQ, IBM, IP, JNJ, KO, KR, MMM, MO, MRK, MSI, PG, XOM)
- **Features:** Log-returns of daily closing prices
- **Input windows:** 5, 10, 30, 90 days
- **Output windows:** 1, 5, 30, 90 days (average of future closing returns)
- **Split:** 90% train / 10% test (chronological, seed=42)

## Project Structure

```
├── config.py                  # Shared constants (windows, tickers, seed)
├── requirements.txt           # Python dependencies
├── src/
│   ├── data_pipeline.py       # Data loading, window generation, train/test split
│   ├── evaluation.py          # MAE computation, standardized results saving
│   ├── plotting.py            # Training curves, comparisons, competition matrix
│   ├── baselines.py           # Buy & Hold, Naive, Zero baselines
│   └── portfolio.py           # Portfolio construction (research section)
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_baselines.ipynb
│   ├── 02_dense_models.ipynb
│   ├── 03_recurrent_models.ipynb
│   ├── 04_cnn_models.ipynb
│   ├── 05_mixed_models.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_research.ipynb
├── data/                      # Raw data (not tracked by git)
├── results/
│   ├── tables/                # CSV results per model
│   └── figures/               # Convergence curves, comparisons, matrix
├── models/                    # Saved model weights
└── report/                    # Final PDF presentation
```

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

## Workflow

1. All shared infrastructure is in `src/` — do not modify without team agreement.
2. Each team member works on their own branch (`feature/dense-models`, `feature/recurrent-models`, `feature/cnn-models`).
3. Merge to `develop` via Pull Request with at least 1 review.
4. `main` only receives stable, reviewed code from `develop`.

## Quick Start

```python
from src.data_pipeline import load_data, get_train_test
from src.evaluation import compute_mae, save_results

returns = load_data()
X_train, X_test, y_train, y_test = get_train_test(returns, input_window=30, output_window=5)

# Train your model...
# mae_test = compute_mae(y_test, predictions)
# save_results("MyModel_v1", "dense", 30, 5, mae_train, mae_test, n_params)
```

## Metric

**MAE (Mean Absolute Error)** — as specified in the assignment.


