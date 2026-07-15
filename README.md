# Tehran Apartment Price Prediction \& Market Explorer

Price prediction model and interactive dashboard for Tehran's apartment-sale market, built on 1M+ anonymized real estate listings from Divar (Iran's largest online classifieds platform).

[**Live Dashboard**](https://divar-price-prediction.streamlit.app) (VPN required)

## Overview

This project scopes down a large, multi-category real estate dataset to a single coherent market segment (Tehran apartment sales), builds and compares regression models to predict listing price, and packages the results into an interactive dashboard. It also documents where the model's predictions break down.

**Full write-up:** [`report.md`](./report.md)

## Key Results

|Model|R² (log scale)|MAE (toman)|
|-|-|-|
|Linear Regression|0.43|5,603,917,291|
|Random Forest|0.59|3,114,900,846|
|**XGBoost**|**0.59**|3,277,396,914|

* Parking, elevator, and storage carry the largest price premiums (+45% to +65%).
* The model's error grows sharply (1,000%+ mean absolute error) in Tehran's highest-end neighborhoods, a segmented limitation documented in the full report.

## Dataset

* **Source:** [Divar real estate listings](https://www.kaggle.com/datasets/valakhorasani/divar-real-estate-dataset1m-persian-property-ads) — 1,000,000+ anonymized listings across cities, property types, and transaction modes.
* **Scope for this project:** `residential-sell` / `apartment-sell` / Tehran → 84,586 listings.
* **Note:** prices are seller-listed asking prices, not confirmed transaction prices.

## Methodology

1. **Scope \& split** — filtered to one segment; time-based train/test split to reflect realistic forecasting conditions.
2. **Clean \& engineer** — handled Persian-language fields (word-based room counts, Jalali-calendar construction years, Persian-Indic numerals), outlier clipping, target-encoded neighborhoods.
3. **Explore** — price distributions, amenity premiums, time trends, individual-vs-agency listing behavior.
4. **Model \& evaluate** — Linear Regression baseline; Random Forest ; XGBoost, with SHAP feature importance and neighborhood-level residual analysis.
5. **Dashboard** — Streamlit app with a filterable market explorer, a live price estimator, and a model-transparency panel.

See the [full report](./report.md) for details, results, and limitations.

## Running Locally

```bash
pip install -r requirements.txt
# Run the notebook first to generate data/processed/ and models/, then:
streamlit run app.py
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · Plotly

## Limitations

* Listed prices are asking prices, not confirmed sale prices.
* Model accuracy degrades significantly in the luxury segment (see report).
* `rooms\\\_count` caps at "5 or more."

Full limitations and methodology notes are in [`report.md`](./report.md).

