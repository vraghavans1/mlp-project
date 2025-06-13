import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
wqvk8f-codex/improve-model-to-achieve-r2_score-0.50
import lightgbm as lgb
main

# Data paths (Kaggle defaults can be overridden by environment variables)
TRAIN_PATH = os.getenv('TRAIN_PATH', '/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv')
TEST_PATH = os.getenv('TEST_PATH', '/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv')

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET = 'purchaseValue'

def preprocess(df):
    if 'date' in df.columns:
        dt = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = dt.dt.year
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        df.drop(columns=['date'], inplace=True)
    if 'sessionStart' in df.columns:
        df['sessionStart'] = pd.to_datetime(df['sessionStart'], errors='coerce')
wqvk8f-codex/improve-model-to-achieve-r2_score-0.50
        df['sessionStart'] = df['sessionStart'].astype('int64') // 10**9
main
    return df

train_df = preprocess(train_df)
test_df = preprocess(test_df)

cat_cols = train_df.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    if col in test_df.columns:
        test_df[col] = test_df[col].astype('category')

for col in train_df.columns:
    if col != TARGET and train_df[col].dtype.kind in 'biufc':
        median = train_df[col].median()
wqvk8f-codex/improve-model-to-achieve-r2_score-0.50
        train_df[col] = train_df[col].fillna(median)
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(median)
main

X = train_df.drop(columns=[TARGET])
y = np.log1p(train_df[TARGET])

params = {
    'n_estimators': 1200,
    'learning_rate': 0.03,
    'num_leaves': 128,
    'max_depth': -1,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42
}
model = LGBMRegressor(**params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for tr_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
wqvk8f-codex/improve-model-to-achieve-r2_score-0.50
        callbacks=[lgb.early_stopping(50)],
main
        verbose=False,
        categorical_feature=cat_cols
    )
    pred = model.predict(X_val)
    score = r2_score(np.expm1(y_val), np.expm1(pred))
    cv_scores.append(score)
print(f"Mean CV R²: {np.mean(cv_scores):.4f}")

model.fit(X, y, categorical_feature=cat_cols)

test_pred = np.expm1(model.predict(test_df))
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({'id': range(test_df.shape[0]), 'purchaseValue': test_pred})
submission.to_csv('submission.csv', index=False)
print('submission.csv saved')
