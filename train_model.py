"""
Train and persist the Malaysia House Price prediction pipelines.

This reproduces Phase 2 (feature engineering) and Phase 3 (preprocessing +
model training/tuning) from House_Prediction_System.ipynb exactly, then
saves the fitted pipelines + metrics so app.py can load them instantly
instead of retraining on every run.

USAGE
-----
1. Put malaysia_house_price_data_2025.csv in this same folder.
2. Run:  python train_model.py
3. Wait for it to finish (grid search over 3 models -- usually well under
   a couple of minutes on a ~2,000-row dataset).
4. Then launch the app:  streamlit run app.py

Outputs (written to ./models/):
  random_forest_pipeline.pkl       <- best model, used by default
  ridge_regression_pipeline.pkl
  gradient_boosting_pipeline.pkl
  metrics.json                     <- test-set metrics, feeds the dashboard
  feature_importance.json          <- Random Forest feature importances
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from model_utils import (
    ALL_FEATURES,
    CARDINALITY_THRESHOLD,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    FrequencyEncoder,
)

DATA_PATH = "malaysia_house_price_data_2025.csv"
OUTPUT_DIR = "models"

# Filenames are also referenced from app.py -- keep these two in sync if you
# rename anything.
MODEL_FILENAMES = {
    "Random_Forest": "random_forest_pipeline.pkl",
    "Ridge_Regression": "ridge_regression_pipeline.pkl",
    "Gradient_Boosting": "gradient_boosting_pipeline.pkl",
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at '{path}' is missing expected column(s): {missing}. "
            f"Expected columns: {ALL_FEATURES + [TARGET_COL]}"
        )
    return df


def build_preprocessor(X_train: pd.DataFrame):
    """Mirrors Phase 3's cardinality-aware routing: low-cardinality
    categoricals go through One-Hot Encoding, high-cardinality ones go
    through frequency encoding so the encoder doesn't explode into a huge
    sparse matrix full of categories the model will rarely see again."""
    cardinality = {col: X_train[col].nunique() for col in CATEGORICAL_FEATURES}
    low_card = [c for c in CATEGORICAL_FEATURES if cardinality[c] <= CARDINALITY_THRESHOLD]
    high_card = [c for c in CATEGORICAL_FEATURES if cardinality[c] > CARDINALITY_THRESHOLD]

    print("--- Categorical Cardinality Check ---")
    for col, n in cardinality.items():
        bucket = "One-Hot" if col in low_card else "Frequency"
        print(f"  {col}: {n} unique values -> {bucket} encoded")

    numeric_transformer = Pipeline(steps=[("scaler", RobustScaler())])
    low_card_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    high_card_transformer = Pipeline(
        steps=[("freq", FrequencyEncoder()), ("scaler", RobustScaler())]
    )

    transformers = [("num", numeric_transformer, NUMERIC_FEATURES)]
    if low_card:
        transformers.append(("cat_low", low_card_transformer, low_card))
    if high_card:
        transformers.append(("cat_high", high_card_transformer, high_card))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, low_card, high_card


def get_model_configs():
    """Same 3 candidate models + hyperparameter grids as Phase 3."""
    return {
        "Ridge_Regression": {
            "model": Ridge(random_state=42),
            "params": {
                "regressor__regressor__alpha": [0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]
            },
        },
        "Random_Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "regressor__regressor__n_estimators": [100, 200],
                "regressor__regressor__max_depth": [8, 12, 15],
                "regressor__regressor__min_samples_leaf": [2, 5],
            },
        },
        "Gradient_Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "regressor__regressor__n_estimators": [100, 150],
                "regressor__regressor__learning_rate": [0.03, 0.1],
                "regressor__regressor__max_depth": [4, 6],
            },
        },
    }


def extract_feature_importance(rf_pipeline, numeric_features, low_card, high_card):
    """Maps Random Forest's flat feature_importances_ array back onto the
    original column names (summing one-hot dummy importances per column)."""
    rf_model = rf_pipeline.named_steps["regressor"].regressor_
    ct = rf_pipeline.named_steps["preprocessor"]

    importance_map = {}
    idx = 0

    for col in numeric_features:
        importance_map[col] = float(rf_model.feature_importances_[idx])
        idx += 1

    if low_card:
        ohe = ct.named_transformers_["cat_low"].named_steps["onehot"]
        for i, col in enumerate(low_card):
            n_cats = len(ohe.categories_[i])
            importance_map[col] = float(np.sum(rf_model.feature_importances_[idx: idx + n_cats]))
            idx += n_cats

    for col in high_card:
        importance_map[col] = float(rf_model.feature_importances_[idx])
        idx += 1

    return importance_map


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[INFO] Loading dataset from '{DATA_PATH}' ...")
    df = load_data(DATA_PATH)
    print(f"[INFO] Dataset shape: {df.shape}")

    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"[INFO] Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    preprocessor, low_card, high_card = build_preprocessor(X_train)

    metrics = {}
    feature_importance = None

    print("\n" + "=" * 60)
    print("TRAINING, 5-FOLD CROSS-VALIDATION & HYPERPARAMETER TUNING")
    print("=" * 60)

    for name, config in get_model_configs().items():
        print(f"\n[RUNNING] {name} ...")

        # Log-transform the (right-skewed) target on the way in, and
        # exp-inverse it on the way out -- same as Phase 3.
        target_regressor = TransformedTargetRegressor(
            regressor=config["model"], func=np.log1p, inverse_func=np.expm1
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", target_regressor)])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        best_pipeline = grid.best_estimator_

        y_train_pred = best_pipeline.predict(X_train)
        y_pred = best_pipeline.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        print(f"  Best params : {grid.best_params_}")
        print(f"  Train R2={train_r2:.4f}  Test R2={test_r2:.4f}  MAE=RM{mae:,.0f}  RMSE=RM{rmse:,.0f}")

        metrics[name] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(rmse, 2),
            "R2": round(float(test_r2), 4),
            "Train_R2": round(float(train_r2), 4),
        }

        out_path = os.path.join(OUTPUT_DIR, MODEL_FILENAMES[name])
        joblib.dump(best_pipeline, out_path)
        print(f"  Saved -> {out_path}")

        if name == "Random_Forest":
            feature_importance = extract_feature_importance(best_pipeline, NUMERIC_FEATURES, low_card, high_card)

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if feature_importance:
        with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w") as f:
            json.dump(feature_importance, f, indent=2)

    best_name = max(metrics, key=lambda k: metrics[k]["R2"])
    print("\n" + "=" * 60)
    print(f"[RESULT] Best performing model on test data: {best_name} (Test R2 = {metrics[best_name]['R2']})")
    print("=" * 60)
    print("\n[SUCCESS] All models + metrics saved to ./models/")
    print("You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
