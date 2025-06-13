import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Paths assume Kaggle environment or downloaded data in working directory
TRAIN_PATH = os.getenv('TRAIN_PATH', '/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv')
TEST_PATH = os.getenv('TEST_PATH', '/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv')

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = 'purchaseValue'

# Identify categorical columns
cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]

# Simple target encoding
prior = train_df[TARGET].mean()
for col in cat_cols:
    mapping = train_df.groupby(col)[TARGET].mean()
    train_df[col] = train_df[col].map(mapping)
    test_df[col] = test_df[col].map(mapping).fillna(prior)

# Fill numeric NA
for col in train_df.columns:
    if train_df[col].dtype.kind in 'biufc' and train_df[col].isnull().any():
        median = train_df[col].median()
        train_df[col].fillna(median, inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(median, inplace=True)

X = train_df.drop(columns=[TARGET])
y = np.log1p(train_df[TARGET])

params = {
    'n_estimators': 1000,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': 1,
}
model = XGBRegressor(**params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
    pred = model.predict(X_val)
    score = r2_score(np.expm1(y_val), np.expm1(pred))
    cv_scores.append(score)
print(f"Mean CV R²: {np.mean(cv_scores):.4f}")

model.fit(X, y)

pred_test = np.expm1(model.predict(test_df))
submission = pd.DataFrame({'id': range(test_df.shape[0]), 'purchaseValue': pred_test})
submission.to_csv('submission.csv', index=False)
print('submission.csv saved')
