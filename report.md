# Predicting Apartment Sale Prices in Tehran: A Data Analysis Case Study

*Analysis of 1M+ Divar real estate listings, scoped to the Tehran apartment-sale segment*

## Overview

This project analyzes anonymized real estate listings from Divar, one of Iran's largest online classifieds platforms, to understand what drives apartment sale prices in Tehran and to build a price prediction model. The work covers data cleaning of a messy, Persian-language dataset, exploratory analysis of pricing patterns, and a comparison of regression models.

## Dataset

The source dataset contains 1,000,000 listings across 60 fields, spanning multiple cities, property types, and transaction modes (sale, long-term rent, short-term rent). Because these different listing types follow fundamentally different pricing logics the analysis was scoped narrowly to one coherent segment: **apartment-sale listings in Tehran** (`residential-sell` / `apartment-sell` / `tehran`), yielding 84,586 listings, or 8.5% of the raw data.

A time-based train/test split was used rather than a random split, to reflect the realistic task of predicting future prices from historical listings rather than leaking information across time:

|Split|Rows|Date range|
|-|-|-|
|Train|72,529|Feb 2020 – Nov 2024|
|Test|12,057|Dec 2024 – Feb 2025|

Listing volume was heavily concentrated in the most recent year (over 90% of listings fall between April 2024 and February 2025), consistent with Divar's more recent growth in structured listing data.

## Data Cleaning

The raw export required substantial cleaning, typical of real-world, Persian-language production data:

* **29 columns were 100% missing** for this segment (e.g. `rent\_value`, `has\_pool`, `has\_jacuzzi`, `land\_size`). these belonged to other property types (short-term rentals, land, rentals) that shared the same export schema, and were dropped rather than imputed.
* **`rooms\_count`** was stored as Persian word categories (`یک`, `دو`, `سه`, `چهار`, `پنج یا بیشتر`, `بدون اتاق`) rather than numbers, and was mapped to a 0–5 numeric scale, where 5 represents "5 or more". a modeling limitation, since the model cannot distinguish a 5-room apartment from a much larger one.
* **`construction\_year`** was recorded in Persian-Indic numerals using the Jalali calendar, and was converted to Western digits and approximated to Gregorian years for feature engineering (e.g. building age).
* **`user\_type`** (individual vs. agency seller) was missing for 49% of listings. critically, this pattern is not random: only 1.6% of listings are confirmed "individual" sellers, 49.5% are confirmed agency listings, and the remaining 49% are unlabeled. Missing values were encoded as their own `"unknown"` category rather than imputed toward either class, to avoid manufacturing a signal that isn't in the data.
* **Outliers** in `price\_value` and `building\_size` were clipped at the 1st/99th percentiles (fit on the training set only, then applied to test) to limit the influence of extreme values without discarding rows outright.

## Exploratory Findings

**Price distribution:** Median price per square meter across the dataset was approximately 84.6 million toman, with a mean of 99.3 million toman. the gap reflects a right-skewed distribution driven by a smaller number of high-end listings, which is why prices were modeled on a log scale.

**Amenities and price:** Comparing median price-per-sqm for listings with vs. without each amenity produced a genuinely interesting split:

|Amenity|Price premium|
|-|-|
|Parking|+65.5%|
|Elevator|+60.6%|
|Warehouse/storage|+44.8%|
|Balcony|−2.2%|
|Restroom|−4.1%|
|Heating system|−8.5%|
|Warm water provider|−8.9%|
|Cooling system|−9.1%|

Parking, elevator, and storage carry large, intuitive price premiums. Heating, cooling, and warm-water systems showed a *negative* association with price, the likely explanation is that these amenities are close to universal in newer, moderately-priced construction, while the priciest listings (older, larger, more established addresses) mention them less often in the listing text even if the units may still have them, or true luxury units simply don't foreground utilities as a selling point. This is a pattern worth investigating further rather than a straightforward causal amenity effect, and is flagged as a limitation rather than presented as a clean finding.

**Individual vs. agency listings:** Confirmed agency listings had both a higher median price (7.6 billion toman) and a higher median price-per-sqm (92.6 million toman) than confirmed individual listings (5.25 billion toman and 70.0 million toman respectively). Given the individual-seller sample is small (1,136 listings vs. 35,877 agency listings), this comparison is suggestive rather than conclusive.

## Modeling

Three regression approaches were compared, predicting log-transformed sale price from 19 features (property characteristics, amenities, a target-encoded neighborhood effect, and seller type), evaluated on the time-based holdout:

|Model|R² (log scale)|MAE (toman)|
|-|-|-|
|Linear Regression|0.43|5,603,917,291|
|Random Forest|0.59|3,114,900,846|
|XGBoost|0.59|3,277,396,914|

Both tree-based models substantially outperform linear regression, roughly halving the mean absolute error and explaining meaningfully more variance in price, evidence that price in this market is driven by non-linear interactions between features (e.g. how amenities matter differently at different size/location combinations) that a linear model can't capture.

Random Forest and XGBoost perform almost identically on R², and Random Forest actually achieves a slightly *lower* MAE in real toman terms despite XGBoost's marginally higher R² on the log scale. Both are reasonable choices; this analysis proceeded with XGBoost for the feature-importance and residual analysis, primarily for its built-in SHAP compatibility.

## Where the Model Breaks Down

The most important finding from this project came from the residual analysis, not the headline accuracy numbers. When broken down by neighborhood, prediction error was wildly uneven: the ten worst-predicted neighborhoods (by mean absolute percentage error, among neighborhoods with at least 10 test listings) were **Jamaran, Aghdasieh, Hekmat, Velenjak, Elahiyeh, Zafaraniyeh, Jordan, Hasan-Abad-Shomali, Tehran-Jolfa, and Dezashib** with mean absolute errors ranging from roughly 1,080% to over 2,500%.

These are, without exception, among Tehran's most affluent, high-end neighborhoods. This is a meaningful and honest limitation: the model, trained predominantly on the broader market, systematically fails to capture the price dynamics of Tehran's luxury segment, where price is likely driven by factors this dataset doesn't capture well (specific address prestige, view, finish quality, brand of building) rather than the structural features available here (size, room count, standard amenities). A production or research use of this model would need either a separate luxury-segment model or an explicit caveat restricting its reliable range to non-luxury listings.

## Limitations

* **Listed prices are asking prices, not transaction prices.** The dataset reflects what sellers requested, not what buyers paid, the model predicts market listing behavior, not confirmed sale value.
* **`rooms\_count` caps at "5 or more,"** collapsing all larger apartments into a single category.
* **The luxury segment is poorly predicted** and should be treated as out of scope for this model's practical range.
* **Neighborhood effects are captured via a single target-encoded average,** which likely understates within-neighborhood variation, particularly in large or diverse neighborhoods.
* **The `user\_type` field is missing for roughly half of listings,** and the confirmed "individual seller" sample is small, limiting confidence in any agency-vs-individual comparison.

## 

