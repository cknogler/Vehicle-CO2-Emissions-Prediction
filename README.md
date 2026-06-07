# 🚗 Vehicle CO₂ Emissions Prediction

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vehicle-co2-cknogler.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data science project analysing the **ADEME Car Labelling Dataset** (France, 2013) to identify the key drivers of vehicle CO₂ emissions. The project covers data preprocessing, exploratory data analysis, clustering, and predictive modelling — deployed as an interactive Streamlit dashboard.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Dashboard](#dashboard)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Technologies](#technologies)
- [Author](#author)

---

## Project Overview

This project investigates which technical vehicle characteristics explain CO₂ emissions and how accurately machine learning models can predict them. It is based on 44,850 vehicle configurations from the French ADEME Car Labelling registry, deduplicated to 5,700 unique mechanical configurations for analysis.

---

## Research Questions

**Clustering**
> Which natural vehicle segments can be identified based on technical characteristics (fuel type, body style, gearbox, power, mass) in the French vehicle market 2013, and how do these segments differ in their CO₂ emissions?

**Prediction**
> What is the relative contribution of vehicle mass, engine power, fuel type, body style and gearbox type in explaining CO₂ emissions, and which minimal feature set achieves the best predictive performance?

---

## Dataset

**Source:** [ADEME Car Labelling Dataset](https://www.data.gouv.fr/en/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/) — French Agency for Ecological Transition

| Property | Value |
|----------|-------|
| Raw records | 44,850 |
| Fuel filter (ES + GO) | 43,935 |
| Unique configurations (deduplicated) | 5,700 |
| Features | 25 (renamed from French) |
| Target variable | CO₂ (g/km) |
| Year | 2013 |

**Key variables used:**

| Feature | Type | Description |
|---------|------|-------------|
| `Empty Mass Euro Avg (kg)` | Numerical | Average kerb weight |
| `Maximum Power (kW)` | Numerical | Engine peak power |
| `GearType` | Categorical | Manual / Automatic / CVT / DCT |
| `GearCount` | Numerical | Number of gears |
| `Fuel` | Categorical | Petrol (ES) / Diesel (GO) |
| `Body` | Categorical | Berline / Break / SUV / Minibus etc. |
| `CO2 (g/km)` | Target | CO₂ emissions in grams per kilometre |

---

## Dashboard

The interactive Streamlit app is structured across seven tabs:

| Tab | Content |
|-----|---------|
| 📋 **Preprocessing** | Missing value heatmap, dataset summary, data cleaning steps |
| 📊 **EDA** | Fleet distribution, primary CO₂ drivers, boxplots by category |
| 🔗 **Correlation Analysis** | Pearson vs. Spearman heatmap, scatter plots with regression lines, hexbin density plots |
| 📉 **Deduplication** | Engineering fleet diversity infographic, outlier analysis (IQR), CO₂ before/after deduplication |
| 🔵 **Clustering** | Elbow method, K-Prototypes (k=4), cluster profiles (radar + heatmap), categorical distribution |
| 🤖 **Prediction** | Feature set comparison (5-fold CV), model performance (R² / MAE), feature importance, partial dependence plots |
| 🎯 **CO₂ Calculator** | Consumer-facing tool: segment, fuel, gearbox, power → median CO₂ from real data + brand comparison |

---

## Methodology

### Preprocessing
- Column renaming (French → English)
- HC/NOX imputation from HC+NOX sum
- Electric vehicle pollutant NaN → 0
- Average kerb weight from Min/Max
- Gearbox split: `"A 6"` → `GearType="Automatic"`, `GearCount=6`

### Deduplication
Vehicles are filtered to petrol and diesel (`ES`, `GO`) and grouped by unique mechanical configuration (brand, model, fuel, body, gearbox, power, mass, CO₂, consumption). This reduces 43,935 rows to **5,700 unique configurations**, removing duplicate trim variants that share identical technical parameters.

### Clustering — K-Prototypes
K-Prototypes (from `kmodes`) is used instead of K-Means because it natively handles mixed numeric and categorical data without requiring label encoding. Numeric features are standardised (StandardScaler); categorical features (Body, Fuel, Gearbox) are handled via Hamming distance.

```
Categorical features : Body, Fuel, Gearbox
Numerical features   : Maximum Power (kW), Empty Mass Euro Avg (kg)
Algorithm            : KPrototypes (init='Cao', n_init=5, k=4)
```

The Elbow Method (k=2–9) confirms k=3 as the mathematical optimum; k=4 is used for richer segment granularity.

### Predictive Modelling
Four feature sets are compared using 5-fold cross-validation on a Random Forest to select the optimal input combination. Five regression models are then trained on an 80/20 train/test split.

**Feature sets compared:**

| Set | Features | CV MAE |
|-----|----------|--------|
| `all_features` | Mass + Power + Fuel + GearType + GearCount + Body | **11.5 g/km** |
| `no_body` | Mass + Power + Fuel + GearType + GearCount | 12.1 g/km |
| `mass_power_fuel` | Mass + Power + Fuel | 14.7 g/km |
| `mass_power_only` | Mass + Power | 16.7 g/km |

**Optimised hyperparameters (RandomizedSearchCV, 30 iterations, 5-fold CV):**

```python
RandomForestRegressor(
    n_estimators=300, max_depth=20,
    max_features=0.8, min_samples_leaf=1
)

GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.2,
    max_depth=6, subsample=1.0, max_features=0.5
)
```

---

## Results

### Clustering — Four Vehicle Segments

| Cluster | Size | Avg CO₂ | Profile |
|---------|------|---------|---------|
| 0 | 2,400 (42%) | ~148 g/km | Light mid-range — low mass & power, below fleet average |
| 1 | 1,430 (25%) | ~210 g/km | Heavy commercial — high mass, mostly diesel, well above fleet average |
| 2 | 1,130 (20%) | ~126 g/km | **Efficiency cluster** — lowest CO₂, light petrol vehicles |
| 3 | 740 (13%) | ~243 g/km | High-performance — highest power & mass, widest spread |

Fleet average: **171.3 g/km**

### Predictive Modelling — Model Performance

| Model | Test R² | Test MAE |
|-------|---------|---------|
| **Gradient Boosting** | **0.955** | **7.50 g/km** |
| Random Forest | 0.951 | 7.64 g/km |
| Linear Regression | 0.864 | 13.72 g/km |
| Ridge | 0.864 | 13.72 g/km |
| Lasso | 0.862 | 13.80 g/km |

### Feature Importance (Random Forest)

| Feature | Importance |
|---------|-----------|
| Empty Mass Euro Avg (kg) | **46.9%** |
| Maximum Power (kW) | **37.2%** |
| GearCount | 4.6% |
| Fuel_GO | 2.6% |
| Fuel_ES | 2.6% |
| Body / GearType | <1% each |

**Key finding:** Mass and power together explain ~84% of CO₂ variance. The gearbox type (manual vs. automatic) has a negligible isolated effect (~0–2 g/km) when controlling for vehicle class and power.

---

## Project Structure

```
Vehicle-CO2-Emissions-Prediction/
│
├── app.py                          # Streamlit dashboard (7 tabs)
├── pipeline.py                     # Standalone ML pipeline script
├── requirements.txt                # Python dependencies
├── cl_JUIN_2013-complet3.csv       # ADEME dataset (raw)
├── CO2_Predictions_full.ipynb      # Original analysis notebook
└── README.md
```

---

## Installation

**Run locally:**

```bash
# Clone the repository
git clone https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction.git
cd Vehicle-CO2-Emissions-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

**Run the pipeline (without UI):**

```bash
python pipeline.py --data cl_JUIN_2013-complet3.csv --output results/
```

The pipeline outputs:
- `results/df_clean.csv` — preprocessed dataset
- `results/df_unique.csv` — deduplicated dataset
- `results/df_clustered.csv` — with cluster labels
- `results/model_rf.pkl` — trained model
- `results/model_meta.json` — metrics and feature importances

---

## Technologies

| Category | Library |
|----------|---------|
| Data processing | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Statistics | `scipy` |
| Machine learning | `scikit-learn` |
| Clustering | `kmodes` (K-Prototypes) |
| Dashboard | `streamlit` |

---

## Author

**Christian Knogler**
Data Analyst · Munich, Germany

[![GitHub](https://img.shields.io/badge/GitHub-cknogler-black?logo=github)](https://github.com/cknogler)
