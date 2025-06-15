import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

TRAIN_PATH = os.getenv(
    "TRAIN_PATH",
    "/kaggle/input/engage-2-value-from-clicks-to-conversions/train_data.csv",
)
TEST_PATH = os.getenv(
    "TEST_PATH",
    "/kaggle/input/engage-2-value-from-clicks-to-conversions/test_data.csv",
)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["dow"] = dt.dt.dayofweek
        df.drop(columns=["date"], inplace=True)
    if "sessionStart" in df.columns:
        ts = pd.to_datetime(df["sessionStart"], errors="coerce")
        df["session_ts"] = ts.astype("int64") // 10**9
        df["session_hour"] = ts.dt.hour
        df["session_dow"] = ts.dt.dayofweek
        df.drop(columns=["sessionStart"], inplace=True)
    return df

train_df = add_date_features(train_df)
test_df = add_date_features(test_df)

TARGET = "purchaseValue"
X = train_df.drop(columns=[TARGET])
y = train_df[TARGET]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore", sparse=False),
    ),
])

preprocessor = ColumnTransformer(
    [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = LGBMRegressor(random_state=42)

param_distributions = {
    "regressor__n_estimators": [500, 800, 1000],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__num_leaves": [31, 63, 127],
    "regressor__max_depth": [-1, 8, 16],
    "regressor__subsample": [0.8, 1.0],
    "regressor__colsample_bytree": [0.8, 1.0],
}

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", model),
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,

    n_iter=50,
    cv=5,

    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
val_pred = best_model.predict(X_val)
val_score = r2_score(y_val, val_pred)
print(f"Validation R²: {val_score:.4f}")

best_model.fit(X, y)

test_pred = best_model.predict(test_df)
submission = pd.DataFrame({"id": range(test_df.shape[0]), "purchaseValue": test_pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
