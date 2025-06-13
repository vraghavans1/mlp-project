# MLP Project T22025

This repository contains code for the Kaggle competition
"Engage-2 Value from Clicks to Conversions". The goal is to
predict customer purchase value from multi-session behavioural data.

## Contents

- `Copy_of_22f3002203_notebook_t22025.ipynb`: initial notebook with baseline model.
- `lightgbm_regressor.py`: LightGBM model with basic preprocessing.
- `xgb_regressor.py`: XGBoost pipeline with one-hot encoding and scaling.
- `xgb_te_regressor.py`: XGBoost model using simple target encoding.

## Usage

Run one of the Python scripts inside the Kaggle notebook environment:

```bash
python lightgbm_regressor.py

# or

python xgb_regressor.py

# or

python xgb_te_regressor.py
```

A file named `submission.csv` will be produced with the required
`id` and `purchaseValue` columns for submission.

## Configuration

Create a `.env` file in the project root based on `.env.example` and fill in your Kaggle API credentials. These variables are used when downloading competition data.
