# Malaysia House Price Predictor — Streamlit App

Deploys the Phase 3 model (`House_Prediction_System.ipynb`) from the
Malaysian House Price Prediction System project as an interactive Streamlit
app: a prediction form, a market dashboard, and a model performance/feature
importance view.

## Files

| File | Purpose |
|---|---|
| `model_utils.py` | Shared feature config + the custom `FrequencyEncoder` used for high-cardinality columns (`Township`, `Area`). Kept separate so the saved model files can be loaded back from any script. |
| `train_model.py` | Reproduces Phase 2/3 of the notebook: feature engineering, the cardinality-aware preprocessing pipeline, and `GridSearchCV` tuning for Ridge / Random Forest / Gradient Boosting. Saves the fitted pipelines + metrics to `models/`. |
| `app.py` | The Streamlit app itself. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put the dataset in this same folder
#    (malaysia_house_price_data_2025.csv — the one used in the notebook)

# 3. Train and save the models (one-time step, ~1-2 minutes)
python train_model.py

# 4. Launch the app
streamlit run app.py
```

After step 3 you should see a new `models/` folder containing:
- `random_forest_pipeline.pkl`, `ridge_regression_pipeline.pkl`, `gradient_boosting_pipeline.pkl`
- `metrics.json` (MAE / RMSE / R² / train-test gap for each model)
- `feature_importance.json` (Random Forest feature importances)

The app reads those files directly, so it loads instantly instead of
retraining every time it starts. If you change the dataset, just re-run
`train_model.py` to refresh them.

## What the app does

**🔮 Predict Price** — pick a State, then Area and Township narrow down
automatically to only the ones that actually exist for that State (pulled
live from the dataset). Choose a property Type and Tenure, adjust the
price-per-square-foot and transaction count (pre-filled from local
historical data where available), and get a predicted `Median_Price`. A
"Compare all 3 models" toggle in the sidebar runs the same input through
Ridge, Random Forest, and Gradient Boosting side by side.

**📊 Market Dashboard** — price distribution, average price by state, price
by tenure type, and a correlation heatmap, all filterable by state.

**📈 Model Performance** — the same MAE/RMSE/R²/overfitting-gap comparison
from Phase 4 of the notebook, plus the Random Forest feature importance
ranking, rendered as interactive charts instead of static notebook output.

## Notes on fidelity to the notebook

- `Estimated_Size_SQFT` is recreated for the dashboard's correlation heatmap
  only (matching Phase 2's EDA use) — it is never passed into the model,
  since it's mathematically derived from `Median_Price` and would leak the
  target, exactly as flagged in the notebook.
- The same train/test split (`test_size=0.20, random_state=42`), the same
  cardinality threshold (15 unique values) for choosing One-Hot vs.
  Frequency encoding, and the same `TransformedTargetRegressor`
  (`log1p` / `expm1`) are used, so metrics here should match the notebook's
  Phase 4 results (small differences are possible if your CSV has been
  modified since the notebook ran).
- Random Forest is set as the default model in the sidebar because it had
  the best test R² in Phase 4 — but Ridge had the smallest train-test gap
  (most stable generalisation), which is why the app also lets you compare
  all three rather than hiding that nuance.

## Next steps (other bonus items from your brief)

This `app.py` + `requirements.txt` structure is already in the shape
Streamlit Community Cloud expects, so "cloud deployment" is mostly a matter
of pushing this folder to a GitHub repo and pointing Streamlit Cloud at it
(you'd need to either commit the trained `models/*.pkl` files or run
`train_model.py` as a startup step). Happy to help with that, a FastAPI
layer, or wiring in generative AI explanations for predictions whenever
you're ready to tackle those.
