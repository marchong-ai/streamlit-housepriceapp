"""
Streamlit app for the Malaysia House Price Prediction System
(CSM3601 -- Group 2: Ammar, Hariz, Alieq, Helmi)

SETUP
-----
1. pip install -r requirements.txt
2. Put malaysia_house_price_data_2025.csv in this folder.
3. python train_model.py        (one-time -- trains & saves the models)
4. streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import FrequencyEncoder  # noqa: F401  (needed so joblib can unpickle the pipeline)

DATA_PATH = "malaysia_house_price_data_2025.csv"
MODEL_DIR = "models"

# Keep these filenames in sync with MODEL_FILENAMES in train_model.py
MODEL_FILES = {
    "Random Forest (Recommended)": "random_forest_pipeline.pkl",
    "Ridge Regression": "ridge_regression_pipeline.pkl",
    "Gradient Boosting": "gradient_boosting_pipeline.pkl",
}

st.set_page_config(page_title="Malaysia House Price Predictor", page_icon="🏠", layout="wide")


# --------------------------------------------------------------------------- 
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    # Same engineered column as Phase 2 -- EDA/dashboard use only, never fed
    # to the model (it's a direct function of the target -> target leakage).
    df["Estimated_Size_SQFT"] = df["Median_Price"] / df["Median_PSF"]
    return df


@st.cache_resource
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_json(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


df = load_data()
metrics = load_json("metrics.json")
feature_importance = load_json("feature_importance.json")

st.title("🏠 Malaysia House Price Prediction System")
st.caption("CSM3601 — Problem Solving Using Artificial Intelligence | Group 2: Ammar, Hariz, Alieq, Helmi")

if df is None:
    st.error(
        f"Couldn't find **{DATA_PATH}**. Place the dataset in this folder (next to app.py) "
        "and refresh the page."
    )
    st.stop()

no_models_trained = all(load_model(f) is None for f in MODEL_FILES.values())
if no_models_trained:
    st.warning(
        "No trained models found in `models/`. Run **`python train_model.py`** in this "
        "folder first (it only needs to be done once), then refresh the page. "
        "You can still browse the Market Dashboard below in the meantime."
    )

with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/home.png",
        width=80
    )

    st.title("Housing Analytics")

    st.markdown("---")

    st.header("Prediction Settings")

    model_label = st.selectbox(
        "Choose Model",
        list(MODEL_FILES.keys()),
        index=0
    )

    compare_all = st.checkbox(
        "Compare All Models",
        value=False
    )

    st.markdown("---")

    st.info(
        """
        **Random Forest** is recommended because it achieved
        the highest test R² score.
        """
    )

pipeline = load_model(MODEL_FILES[model_label])

tab_predict, tab_dashboard, tab_performance = st.tabs(
    ["🔮 Predict Price", "📊 Market Dashboard", "📈 Model Performance"]
)

# =============================================================================
# TAB 1 -- Predict
# =============================================================================
with tab_predict:
    st.subheader("Estimate a property cluster's median price")
    st.caption(
        "Fill in the location and property details below. Township and Area "
        "options narrow down automatically once you pick a State."
    )

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("State", sorted(df["State"].unique()))

        areas = sorted(df.loc[df["State"] == state, "Area"].unique())
        area = st.selectbox("Area", areas)

        townships = sorted(
            df.loc[(df["State"] == state) & (df["Area"] == area), "Township"].unique()
        )
        township = st.selectbox("Township", townships)

    with col2:
        prop_type = st.selectbox("Property Type", sorted(df["Type"].unique()))
        tenure = st.selectbox("Tenure", sorted(df["Tenure"].unique()))

        # Pre-fill sensible defaults from whatever historical rows match the
        # chosen location, so the user isn't typing in numbers from scratch.
        local_subset = df[
            (df["State"] == state) & (df["Area"] == area) & (df["Township"] == township)
        ]
        if len(local_subset):
            default_psf = float(local_subset["Median_PSF"].mean())
            default_txn = int(round(local_subset["Transactions"].mean()))
        else:
            default_psf = float(df["Median_PSF"].median())
            default_txn = int(df["Transactions"].median())

        median_psf = st.number_input(
            "Median price per square foot (RM)",
            min_value=0.0,
            value=round(default_psf, 2),
            step=10.0,
        )
        transactions = st.number_input(
            "Number of recorded transactions", min_value=0, value=default_txn, step=1
        )

    predict_clicked = st.button("Predict Median Price", type="primary", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame(
            [
                {
                    "Township": township,
                    "Area": area,
                    "State": state,
                    "Tenure": tenure,
                    "Type": prop_type,
                    "Median_PSF": median_psf,
                    "Transactions": transactions,
                }
            ]
        )

        if compare_all:
            rows = []
            for label, filename in MODEL_FILES.items():
                model = load_model(filename)
                if model is not None:
                    rows.append({"Model": label, "Predicted Median Price (RM)": float(model.predict(input_df)[0])})

            if rows:
                comp_df = pd.DataFrame(rows)
                st.dataframe(
                    comp_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Predicted Median Price (RM)": st.column_config.NumberColumn(format="RM %.0f")
                    },
                )
                fig = px.bar(
                    comp_df, x="Model", y="Predicted Median Price (RM)", color="Model", text_auto=".2s"
                )
                fig.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No trained models found. Run `python train_model.py` first.")

        elif pipeline is not None:
            prediction = float(pipeline.predict(input_df)[0])
            st.metric("Predicted Median Price", f"RM {prediction:,.0f}")

            state_avg = df.loc[df["State"] == state, "Median_Price"].mean()
            overall_median = df["Median_Price"].median()
            diff_state = prediction - state_avg
            diff_national = prediction - overall_median
            c1, c2 = st.columns(2)
            c1.metric(f"{state} average", f"RM {state_avg:,.0f}", delta=f"{diff_state:+,.0f} RM vs your prediction", delta_color="off")
            c2.metric("National median", f"RM {overall_median:,.0f}", delta=f"{diff_national:+,.0f} RM vs your prediction", delta_color="off")
        else:
            st.error(f"No trained model found for **{model_label}**. Run `python train_model.py` first.")

        st.caption(
            "⚠️ This is a statistical estimate based on township-level aggregated data, "
            "not a formal valuation -- treat it as a ballpark figure. Actual prices "
            "depend on property condition, renovations, and negotiated terms."
        )

# =============================================================================
# TAB 2 -- Dashboard
# =============================================================================
with tab_dashboard:
    st.subheader("📊 Malaysian Housing Market Overview")

    # -------------------------------------------------------------------------
    # KPI Dashboard
    # -------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            "Property Clusters",
            f"{len(df):,}"
        )

    with kpi2:
        st.metric(
            "Average Price",
            f"RM {df['Median_Price'].mean():,.0f}"
        )

    with kpi3:
        st.metric(
            "Median Price / SQFT",
            f"RM {df['Median_PSF'].median():,.0f}"
        )

    with kpi4:
        st.metric(
            "States Covered",
            df["State"].nunique()
        )

    st.divider()

    # -------------------------------------------------------------------------
    # Dashboard Filters
    # -------------------------------------------------------------------------
    st.markdown("### Dashboard Filters")

    filter1, filter2, filter3 = st.columns(3)

    with filter1:
        state_filter = st.multiselect(
            "State",
            sorted(df["State"].unique())
        )

    with filter2:
        type_filter = st.multiselect(
            "Property Type",
            sorted(df["Type"].unique())
        )

    with filter3:
        tenure_filter = st.multiselect(
            "Tenure",
            sorted(df["Tenure"].unique())
        )

    min_price = int(df["Median_Price"].min())
    max_price = int(df["Median_Price"].max())

    price_range = st.slider(
        "Median Price Range (RM)",
        min_price,
        max_price,
        (min_price, max_price),
    )

    # -------------------------------------------------------------------------
    # Apply Filters
    # -------------------------------------------------------------------------
    view_df = df.copy()

    if state_filter:
        view_df = view_df[
            view_df["State"].isin(state_filter)
        ]

    if type_filter:
        view_df = view_df[
            view_df["Type"].isin(type_filter)
        ]

    if tenure_filter:
        view_df = view_df[
            view_df["Tenure"].isin(tenure_filter)
        ]

    view_df = view_df[
        (view_df["Median_Price"] >= price_range[0]) &
        (view_df["Median_Price"] <= price_range[1])
    ]

    st.divider()

    # -------------------------------------------------------------------------
    # Price Distribution & State Comparison
    # -------------------------------------------------------------------------
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            view_df,
            x="Median_Price",
            nbins=35,
            title="Distribution of Median House Prices",
            labels={
                "Median_Price": "Median Price (RM)"
            },
            color_discrete_sequence=["#1f77b4"],
        )

        fig.add_vline(
            x=view_df["Median_Price"].mean(),
            line_dash="dash",
            annotation_text="Average",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:
        state_avg = (
            view_df
            .groupby("State")["Median_Price"]
            .mean()
            .sort_values()
            .reset_index()
        )

        fig = px.bar(
            state_avg,
            x="Median_Price",
            y="State",
            orientation="h",
            title="Average Median Price by State",
            labels={
                "Median_Price": "Average Median Price (RM)"
            },
            color="Median_Price",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Tenure & Correlation
    # -------------------------------------------------------------------------
    left2, right2 = st.columns(2)

    with left2:
        fig = px.box(
            view_df,
            x="Tenure",
            y="Median_Price",
            color="Tenure",
            title="Median Price by Tenure Type",
            labels={
                "Median_Price": "Median Price (RM)"
            },
        )

        fig.update_layout(showlegend=False)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right2:
        corr_cols = [
            "Median_Price",
            "Median_PSF",
            "Transactions",
            "Estimated_Size_SQFT",
        ]

        corr = view_df[corr_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Top 10 Most Expensive Areas
    # -------------------------------------------------------------------------
    left3, right3 = st.columns(2)

    with left3:

        expensive = (
            view_df.groupby("Area")["Median_Price"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig = px.bar(
            expensive,
            x="Median_Price",
            y="Area",
            orientation="h",
            title="Top 10 Most Expensive Areas",
            color="Median_Price",
            color_continuous_scale="Reds",
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right3:

        property_avg = (
            view_df.groupby("Type")["Median_Price"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            property_avg,
            x="Type",
            y="Median_Price",
            title="Average Price by Property Type",
            color="Median_Price",
            color_continuous_scale="Viridis",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # State Summary Table
    # -------------------------------------------------------------------------
    st.subheader("State Market Summary")

    summary = (
        view_df.groupby("State")
        .agg(
            Average_Price=("Median_Price", "mean"),
            Median_PSF=("Median_PSF", "median"),
            Transactions=("Transactions", "sum"),
            Property_Clusters=("State", "count"),
        )
        .round(0)
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# TAB 3 -- Model performance
# =============================================================================
with tab_performance:
    st.subheader("How the three trained models compare")

    if metrics is None:
        st.warning("No metrics found. Run `python train_model.py` first.")
    else:
        rows = []
        for name, m in metrics.items():
            gap = m["Train_R2"] - m["R2"]
            if gap > 0.30:
                status = "🔴 Severe overfitting"
            elif gap > 0.10:
                status = "🟡 Mild overfitting"
            else:
                status = "🟢 Healthy"
            rows.append(
                {
                    "Model": name.replace("_", " "),
                    "MAE (RM)": m["MAE"],
                    "RMSE (RM)": m["RMSE"],
                    "Test R2": m["R2"],
                    "Train R2": m["Train_R2"],
                    "Train-Test Gap": round(gap, 4),
                    "Status": status,
                }
            )
        metrics_df = pd.DataFrame(rows)
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "MAE (RM)": st.column_config.NumberColumn(format="RM %.0f"),
                "RMSE (RM)": st.column_config.NumberColumn(format="RM %.0f"),
            },
        )

        pc1, pc2 = st.columns(2)
        with pc1:
            fig = px.bar(metrics_df, x="Model", y="Test R2", color="Model", text_auto=".3f", title="Test R² Score by Model")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with pc2:
            fig = px.bar(metrics_df, x="Model", y="MAE (RM)", color="Model", text_auto=".2s", title="Mean Absolute Error by Model")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    if feature_importance:
        st.subheader("What drives price the most? (Random Forest)")
        fi_df = (
            pd.DataFrame.from_dict(feature_importance, orient="index", columns=["Importance"])
            .sort_values("Importance", ascending=True)
            .reset_index()
            .rename(columns={"index": "Feature"})
        )
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Ethical note:** this model is trained on township-level aggregated data and, "
        "at best, explains under half of real-world price variance. It should support, "
        "not replace, professional property valuation -- predictions are least reliable "
        "in states with fewer recorded transactions."
    )
