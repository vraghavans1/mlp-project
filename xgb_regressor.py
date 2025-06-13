import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

train_path = '/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv'
test_path = '/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Drop columns not available at prediction time
DROP_COLS = ['purchaseValue', 'userId', 'sessionId', 'date', 'sessionStart', 'gclIdPresent']
DROP_COLS = [c for c in DROP_COLS if c in train_data.columns]

# Convert device.isMobile to binary if present
for df in (train_data, test_data):
    if 'device.isMobile' in df.columns:
        df['isMobile'] = df['device.isMobile'].map({True:1, False:0, 'TRUE':1, 'FALSE':0})

y = np.log1p(train_data['purchaseValue'])
X = train_data.drop(columns=DROP_COLS)
X_test = test_data.drop(columns=[c for c in DROP_COLS if c != 'purchaseValue'])

categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=1
    ))
])

kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    score = r2_score(np.expm1(y_val), np.expm1(preds))
    cv_scores.append(score)
    print(f"Fold R²: {score:.4f}")
print(f"Mean CV R²: {np.mean(cv_scores):.4f}")

model.fit(X, y)
test_pred = np.expm1(model.predict(X_test))

submission = pd.DataFrame({
    'id': range(0, test_data.shape[0]),
    'purchaseValue': test_pred
})
submission.to_csv('submission.csv', index=False)
print("submission.csv saved")
