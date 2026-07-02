"""
Shared utilities for the Malaysia House Price Prediction project.

Why this file exists on its own (instead of living inside train_model.py):
scikit-learn pipelines that use a custom transformer can only be unpickled
with joblib.load() if that transformer's class is importable from the exact
same module path it was pickled from. If FrequencyEncoder were defined inside
train_model.py's __main__ script, app.py would crash with something like:

    AttributeError: Can't get attribute 'FrequencyEncoder' on <module '__main__'>

By keeping it here and having both train_model.py and app.py do
`from model_utils import FrequencyEncoder`, the saved .pkl files load cleanly
in either script.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Feature configuration -- shared between training and the Streamlit app so
# the two can never silently drift apart.
# ---------------------------------------------------------------------------
TARGET_COL = "Median_Price"
CATEGORICAL_FEATURES = ["Township", "Area", "State", "Tenure", "Type"]
NUMERIC_FEATURES = ["Median_PSF", "Transactions"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
CARDINALITY_THRESHOLD = 15  # matches Phase 3 of the notebook


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes a categorical column by how often each category appears in the
    TRAINING data (fit only on train folds, so no leakage across CV splits).
    Unlike One-Hot Encoding this stays low-dimensional and degrades gracefully
    (frequency 0) for categories never seen during training, instead of
    zeroing out an entire block of dummy columns.

    This is copied verbatim from Phase 3 of House_Prediction_System.ipynb.
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = {col: X[col].value_counts(normalize=True) for col in X.columns}
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0.0)
        return X.to_numpy(dtype=float)
