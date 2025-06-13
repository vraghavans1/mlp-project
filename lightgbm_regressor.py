import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

# Paths assume Kaggle environment
train_path = '/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv'
test_path = '/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Target and features
y = train_data['purchaseValue']
X = train_data.drop(columns=['purchaseValue'])

# Basic preprocessing: fill numeric NA with median and convert categoricals to 'category'
for col in X.columns:
    if X[col].dtype.kind in 'biufc':
        median = X[col].median()
        X[col] = X[col].fillna(median)
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(median)
    else:
        X[col] = X[col].astype('category')
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('category')

# Log-transform target to reduce skew
y_log = np.log1p(y)

params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': -1,
    'num_leaves': 64,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}
model = LGBMRegressor(**params)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[train_idx], y_log.iloc[val_idx]
    model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)
    preds = model.predict(X_val)
    score = r2_score(np.expm1(y_val), np.expm1(preds))
    scores.append(score)
print(f'CV R2: {np.mean(scores):.4f}')

# Train on full data
model.fit(X, y_log)

# Predict on test
preds_test = np.expm1(model.predict(test_data))
submission = pd.DataFrame({'id': range(test_data.shape[0]), 'purchaseValue': preds_test})
submission.to_csv('submission.csv', index=False)
print('submission.csv saved')
