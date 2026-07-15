"""
Divar Real Estate — Price Explorer & Predictor
Step 5: Interactive dashboard combining EDA, a live prediction tool, and a
model-transparency (feature importance) panel.

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

st.set_page_config(page_title="Divar Real Estate Explorer", layout="wide")


# ---------------------------------------------------------------------------
# Data & model loading (cached so it only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    train_df = pd.read_parquet("data/processed/train_clean.parquet")
    return train_df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load("models/price_model.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    neighborhood_encoding = joblib.load("models/neighborhood_encoding.pkl")
    return model, feature_cols, neighborhood_encoding


train_df = load_data()
model, feature_cols, neighborhood_encoding = load_model_artifacts()

amenity_cols = [c for c in train_df.columns if c.startswith("has_")]
city_wide_mean_price_per_sqm = train_df["price_per_sqm"].mean()

st.title("🏠 Divar Real Estate — Price Explorer & Predictor")
st.caption(
    "Apartment-sale listings, Tehran. Prices reflect asking prices at time of "
    "listing, not confirmed transaction prices."
)

tab_explore, tab_predict, tab_model = st.tabs(
    ["📊 Explore the Market", "💰 Predict a Price", "🔍 How the Model Works"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Filterable EDA view
# ---------------------------------------------------------------------------

with tab_explore:
    st.subheader("Filter listings")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_price, max_price = st.slider(
            "Price range (toman)",
            min_value=int(train_df["price_value"].min()),
            max_value=int(train_df["price_value"].max()),
            value=(int(train_df["price_value"].quantile(0.05)),
                   int(train_df["price_value"].quantile(0.95))),
        )

    with col2:
        rooms_options = sorted(train_df["rooms_count"].dropna().unique().tolist())
        selected_rooms = st.multiselect(
            "Rooms", options=rooms_options, default=rooms_options
        )

    with col3:
        neighborhoods = sorted(train_df["neighborhood_slug"].dropna().unique().tolist())
        selected_neighborhoods = st.multiselect(
            "Neighborhood (leave empty for all)", options=neighborhoods, default=[]
        )

    filtered = train_df[
        (train_df["price_value"] >= min_price)
        & (train_df["price_value"] <= max_price)
        & (train_df["rooms_count"].isin(selected_rooms))
    ]
    if selected_neighborhoods:
        filtered = filtered[filtered["neighborhood_slug"].isin(selected_neighborhoods)]

    st.markdown(f"**{len(filtered):,} listings match your filters**")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_price = px.histogram(
            filtered, x="price_per_sqm", nbins=60,
            title="Price per sqm distribution"
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with col_b:
        monthly = (
            filtered.groupby(filtered["created_at_month"].dt.to_period("M"))
            .agg(median_price_per_sqm=("price_per_sqm", "median"))
            .reset_index()
        )
        monthly["created_at_month"] = monthly["created_at_month"].dt.to_timestamp()
        fig_trend = px.line(
            monthly, x="created_at_month", y="median_price_per_sqm",
            title="Median price/sqm over time", markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Amenity price premium")
    amenity_rows = []
    for col in amenity_cols:
        with_amenity = filtered.loc[filtered[col] == True, "price_per_sqm"].median()
        without_amenity = filtered.loc[filtered[col] == False, "price_per_sqm"].median()
        if pd.notna(with_amenity) and pd.notna(without_amenity) and without_amenity > 0:
            amenity_rows.append({
                "amenity": col.replace("has_", "").replace("_", " ").title(),
                "pct_premium": round((with_amenity / without_amenity - 1) * 100, 1),
            })
    amenity_df = pd.DataFrame(amenity_rows).sort_values("pct_premium", ascending=False)
    fig_amenity = px.bar(
        amenity_df, x="pct_premium", y="amenity", orientation="h",
        title="% price/sqm premium by amenity (within current filter)"
    )
    st.plotly_chart(fig_amenity, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Live prediction tool
# ---------------------------------------------------------------------------

with tab_predict:
    st.subheader("Estimate a price")
    st.caption(
        "Enter property details below. The estimate is based on historical listing "
        "patterns and reflects likely asking price, not a guaranteed sale price."
    )

    col1, col2 = st.columns(2)

    with col1:
        building_size = st.number_input("Building size (sqm)", min_value=20, max_value=1000, value=80)
        rooms_count = st.selectbox("Rooms", options=[0, 1, 2, 3, 4, 5], index=2)
        floor = st.number_input("Floor", min_value=0, max_value=40, value=3)
        unit_per_floor = st.number_input("Units per floor", min_value=1, max_value=10, value=2)
        construction_year = st.number_input(
            "Construction year (Gregorian)", min_value=1950, max_value=2026, value=2010
        )

    with col2:
        neighborhood = st.selectbox(
            "Neighborhood", options=sorted(train_df["neighborhood_slug"].dropna().unique())
        )
        selected_amenities = st.multiselect(
            "Amenities",
            options=[c.replace("has_", "").replace("_", " ").title() for c in amenity_cols],
        )

    if st.button("Estimate price", type="primary"):
        listing_year = pd.Timestamp.now().year
        building_age = max(listing_year - construction_year, 0)
        amenity_count = len(selected_amenities)
        floor_ratio = floor / unit_per_floor if unit_per_floor > 0 else np.nan
        neighborhood_price_encoding = neighborhood_encoding.get(
            neighborhood, city_wide_mean_price_per_sqm
        )

        input_row = {col: 0 for col in feature_cols}
        input_row.update({
            "building_size": building_size,
            "rooms_count": rooms_count,
            "floor": floor,
            "unit_per_floor": unit_per_floor,
            "building_age": building_age,
            "amenity_count": amenity_count,
            "floor_ratio": floor_ratio,
            "neighborhood_price_encoding": neighborhood_price_encoding,
        })
        for amenity_display in selected_amenities:
            amenity_col = "has_" + amenity_display.lower().replace(" ", "_")
            if amenity_col in input_row:
                input_row[amenity_col] = 1

        X_input = pd.DataFrame([input_row])[feature_cols]
        pred_log = model.predict(X_input)[0]
        pred_price = np.expm1(pred_log)

        # Rough confidence range using the model's known test-set error spread
        # (replace with a saved residual std from Step 4 for a tighter estimate)
        lower, upper = pred_price * 0.85, pred_price * 1.15

        st.success(f"**Estimated price: {pred_price:,.0f} toman**")
        st.caption(f"Likely range: {lower:,.0f} – {upper:,.0f} toman")


# ---------------------------------------------------------------------------
# Tab 3 — Model transparency panel
# ---------------------------------------------------------------------------

with tab_model:
    st.subheader("What drives the price estimate")
    st.caption(
        "SHAP values show how much each feature pushes a prediction up or down, "
        "computed on a sample of the training data."
    )

    @st.cache_data
    def compute_shap_summary():
        sample = train_df[feature_cols].sample(min(1000, len(train_df)), random_state=42)
        for col in amenity_cols:
            if col in sample.columns:
                sample[col] = sample[col].astype(int)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        mean_abs_shap = pd.DataFrame({
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        return mean_abs_shap

    shap_summary = compute_shap_summary()
    fig_shap = px.bar(
        shap_summary.head(15), x="mean_abs_shap", y="feature", orientation="h",
        title="Top 15 features by average impact on predicted price"
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.subheader("Limitations")
    st.markdown(
        """
        - Prices are **asking prices** at time of listing, not confirmed sale prices.
        - `rooms_count` caps at "5 or more," so the model can't distinguish a 5-room
          from an 8-room apartment.
        - Neighborhood effects are captured via a target-encoded average — new or
          rare neighborhoods fall back to the city-wide mean, which may understate
          local variation.
        - This model is scoped to one city and property type; it should not be
          used to estimate prices outside that scope.
        """
    )
