"""
app.py  –  Vehicle CO₂ Emissions Dashboard
Streamlit App – basierend auf dem originalen Notebook-Code
ADEME Car Labelling Dataset (cl_JUIN_2013-complet3.csv)
"""
import io
import urllib.request
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.stats import pearsonr, spearmanr
try:
    from kmodes.kprototypes import KPrototypes
    KPROTO_AVAILABLE = True
except ImportError:
    KPROTO_AVAILABLE = False
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle CO₂ Emissions",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#1f77b4"
RANDOM_STATE = 42

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
})

CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)

COLUMN_MAPPING = {
    "Marque": "Brand",
    "Modèle dossier": "Folder Model",
    "Modèle UTAC": "Utac Model",
    "Désignation commerciale": "Commerical Designation",
    "CNIT": "cnit",
    "Type Variante Version (TVV)": "Type Variant Version",
    "Carburant": "Fuel",
    "Hybride": "Hybrid",
    "Puissance administrative": "Administrative Power",
    "Puissance maximale (kW)": "Maximum Power (kW)",
    "Boîte de vitesse": "Gearbox",
    "Consommation urbaine (l/100km)": "Urban Consumption (l/100km)",
    "Consommation extra-urbaine (l/100km)": "Extra Urban Consumption (l/100km)",
    "Consommation mixte (l/100km)": "Combined Consumption (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)",
    "CO type I (g/km)": "CO type 1 (g/km)",
    "HC (g/km)": "HC (g/km)",
    "NOX (g/km)": "NOX (g/km)",
    "HC+NOX (g/km)": "HC+NOX (g/km)",
    "Particules (g/km)": "Particles (g/km)",
    "masse vide euro min (kg)": "Empty Mass Euro Min (kg)",
    "masse vide euro max (kg)": "Empty Mass Euro Max (kg)",
    "Champ V9": "Field V9",
    "Date de mise à jour": "Update Date",
    "Carrosserie": "Body",
    "gamme": "Range",
}

UNIQUE_COLS = [
    'Brand', 'Folder Model', 'Fuel', 'Body', 'Gearbox',
    'Maximum Power (kW)', 'Empty Mass Euro Avg (kg)',
    'CO2 (g/km)', 'Combined Consumption (l/100km)', 'Range'
]

FEATURE_SETS = {
    "all_features":    ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel", "GearType", "GearCount", "Body"],
    "no_body":         ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel", "GearType", "GearCount"],
    "mass_power_fuel": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel"],
    "mass_power_only": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)"],
}

# ── Data loading & preprocessing ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_preprocess(source) -> pd.DataFrame:
    if isinstance(source, str):
        with urllib.request.urlopen(source) as r:
            raw = r.read()
    else:
        raw = source

    df = None
    for enc in ["latin1", "utf-8", "cp1252"]:
        for sep in [";", ","]:
            try:
                tmp = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if tmp.shape[1] > 5:
                    df = tmp
                    break
            except Exception:
                continue
        if df is not None:
            break
    if df is None:
        raise ValueError("CSV konnte nicht gelesen werden.")

    # Rename
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    # HC/NOX imputation
    if all(c in df.columns for c in ["HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)"]):
        df['hc_calc']  = df['HC+NOX (g/km)'] - df['NOX (g/km)']
        df['nox_calc'] = df['HC+NOX (g/km)'] - df['HC (g/km)']
        df['hc_calc']  = df['hc_calc'].fillna(df['HC (g/km)'])
        df['nox_calc'] = df['nox_calc'].fillna(df['NOX (g/km)'])
        df["HC (g/km)"]    = df["hc_calc"]
        df["NOX (g/km)"]   = df["nox_calc"]
        df["HC+NOX (g/km)"] = df["hc_calc"] + df["nox_calc"]
        df.drop(columns=['hc_calc', 'nox_calc'], inplace=True)

    # Gearbox fix
    if "Gearbox" in df.columns:
        df['Gearbox'] = df['Gearbox'].replace(['N 0', 'N 1'], 'A 0')
        df['Gearbox'] = df['Gearbox'].replace(['S 6'], 'D 6')

    # Electric → 0
    electric_cols = ["CO type 1 (g/km)", "Urban Consumption (l/100km)",
                     "Extra Urban Consumption (l/100km)", "Combined Consumption (l/100km)",
                     "CO2 (g/km)", "HC+NOX (g/km)", "HC (g/km)", "Particles (g/km)"]
    if "Fuel" in df.columns:
        el_mask = df["Fuel"] == "EL"
        for c in electric_cols:
            if c in df.columns:
                df.loc[el_mask, c] = df.loc[el_mask, c].fillna(0)

    # Mass avg
    if "Empty Mass Euro Min (kg)" in df.columns and "Empty Mass Euro Max (kg)" in df.columns:
        df["Empty Mass Euro Avg (kg)"] = (
            pd.to_numeric(df["Empty Mass Euro Min (kg)"], errors="coerce") +
            pd.to_numeric(df["Empty Mass Euro Max (kg)"], errors="coerce")
        ) / 2
        df.drop(columns=["Empty Mass Euro Min (kg)", "Empty Mass Euro Max (kg)"], inplace=True)

    # Numeric types
    for col in ["CO2 (g/km)", "Combined Consumption (l/100km)",
                "Maximum Power (kW)", "Empty Mass Euro Avg (kg)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


GEAR_TYPE_MAP = {"M": "Manual", "A": "Automatic", "V": "CVT",
                  "D": "DCT", "N": "Automatic", "S": "Manual"}

@st.cache_data(show_spinner=False)
def make_df_unique(df: pd.DataFrame) -> pd.DataFrame:
    if "Fuel" not in df.columns:
        return df
    df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy()

    # Split Gearbox "A 6" -> GearType="Automatic", GearCount=6
    if "Gearbox" in df_combus.columns:
        gear_split = df_combus["Gearbox"].astype(str).str.split(" ", expand=True)
        df_combus["GearType"]  = gear_split[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_combus["GearCount"] = pd.to_numeric(
            gear_split[1] if 1 in gear_split.columns else pd.Series([np.nan]*len(df_combus)),
            errors="coerce"
        )

    cols = [c for c in UNIQUE_COLS if c in df_combus.columns]
    df_unique = (
        df_combus.groupby(cols, dropna=False)
        .size()
        .reset_index(name='Clone_Count')
        .sort_values('Clone_Count', ascending=False)
        .reset_index(drop=True)
    )

    # Re-add GearType + GearCount after groupby (merge back)
    if "Gearbox" in df_unique.columns:
        gear_split2 = df_unique["Gearbox"].astype(str).str.split(" ", expand=True)
        df_unique["GearType"]  = gear_split2[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_unique["GearCount"] = pd.to_numeric(
            gear_split2[1] if 1 in gear_split2.columns else pd.Series([np.nan]*len(df_unique)),
            errors="coerce"
        )

    return df_unique


@st.cache_data(show_spinner=False)
def run_clustering(_df: pd.DataFrame, k: int = 4):
    """
    K-Prototypes clustering for mixed numeric + categorical data.
    Numeric columns are standardized; categoricals handled natively.
    """
    categorical_cols = [c for c in ['Body', 'Fuel', 'Gearbox'] if c in _df.columns]
    numeric_cols     = [c for c in ['Maximum Power (kW)', 'Empty Mass Euro Avg (kg)'] if c in _df.columns]
    feature_cols     = categorical_cols + numeric_cols
    target_col       = 'CO2 (g/km)'

    df_c = _df[feature_cols + [target_col]].dropna().copy()

    # Scale numeric columns
    scaler = StandardScaler()
    df_kp  = df_c.copy()
    df_kp[numeric_cols] = scaler.fit_transform(df_kp[numeric_cols])

    # Categoricals must be object dtype for KPrototypes
    for col in categorical_cols:
        df_kp[col] = df_kp[col].astype(str)

    X_matrix       = df_kp[feature_cols].to_numpy(dtype=object)
    categorical_idx = [feature_cols.index(col) for col in categorical_cols]

    if not KPROTO_AVAILABLE:
        st.error("kmodes not installed. Please add 'kmodes>=0.12.2' to requirements.txt.")
        return df_c

    kproto = KPrototypes(
        n_clusters=k,
        init='Cao',
        n_init=5,
        verbose=0,
        random_state=RANDOM_STATE
    )
    df_c['Cluster'] = kproto.fit_predict(X_matrix, categorical=categorical_idx)
    return df_c


@st.cache_resource(show_spinner=False)
def train_all_models(_df: pd.DataFrame):
    target_col = "CO2 (g/km)"
    all_needed = sorted(set(
        [target_col] + [c for cols in FEATURE_SETS.values() for c in cols]
    ))
    df_model = _df[[c for c in all_needed if c in _df.columns]].dropna().copy()

    def get_types(features):
        num = df_model[features].select_dtypes(include=["int64","float64"]).columns.tolist()
        cat = df_model[features].select_dtypes(include=["object","category"]).columns.tolist()
        return num, cat

    def build_preprocessors(num, cat):
        scaled = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num)
        ])
        tree = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num)
        ])
        return scaled, tree

    # Feature set comparison (CV on RF)
    fs_results = []
    for fs_name, fs_feats in FEATURE_SETS.items():
        feats_avail = [f for f in fs_feats if f in df_model.columns]
        if not feats_avail:
            continue
        num, cat = get_types(feats_avail)
        _, tree_pre = build_preprocessors(num, cat)
        pipe = Pipeline([("pre", tree_pre),
                         ("m", RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1))])
        scores = cross_val_score(pipe, df_model[feats_avail], df_model[target_col],
                                 cv=5, scoring="neg_mean_absolute_error")
        fs_results.append({"Feature_Set": fs_name, "Features": ", ".join(feats_avail),
                            "CV_MAE_mean": -np.mean(scores), "CV_MAE_std": np.std(scores)})

    fs_df = pd.DataFrame(fs_results).sort_values("CV_MAE_mean")
    best_fs = fs_df.iloc[0]["Feature_Set"]
    feature_cols = [f for f in FEATURE_SETS[best_fs] if f in df_model.columns]

    X = df_model[feature_cols]
    y = df_model[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=RANDOM_STATE)

    num_f, cat_f = get_types(feature_cols)
    scaled_pre, tree_pre = build_preprocessors(num_f, cat_f)

    model_defs = {
        "Linear Regression": Pipeline([("pre", scaled_pre), ("m", LinearRegression())]),
        "Ridge":             Pipeline([("pre", scaled_pre), ("m", Ridge(alpha=1.0))]),
        "Lasso":             Pipeline([("pre", scaled_pre), ("m", Lasso(alpha=0.1))]),
        "Random Forest":     Pipeline([("pre", tree_pre),
                                       ("m", RandomForestRegressor(
                                           n_estimators=300,
                                           max_depth=20,
                                           max_features=0.8,
                                           min_samples_split=2,
                                           min_samples_leaf=1,
                                           random_state=RANDOM_STATE,
                                           n_jobs=-1
                                       ))]),
        "Gradient Boosting": Pipeline([("pre", tree_pre),
                                       ("m", GradientBoostingRegressor(
                                           n_estimators=200,
                                           learning_rate=0.2,
                                           max_depth=6,
                                           min_samples_split=10,
                                           subsample=1.0,
                                           max_features=0.5,
                                           random_state=RANDOM_STATE
                                       ))]),
    }

    results = []
    fitted = {}
    for name, pipe in model_defs.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        ytr_pred = pipe.predict(X_train)
        yte_pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "Train_R2": r2_score(y_train, ytr_pred),
            "Test_R2":  r2_score(y_test, yte_pred),
            "Train_MAE": mean_absolute_error(y_train, ytr_pred),
            "Test_MAE":  mean_absolute_error(y_test, yte_pred),
        })

    results_df = pd.DataFrame(results).sort_values("Test_R2", ascending=False)

    # RF details
    rf_pipe = fitted["Random Forest"]
    rf_pre  = rf_pipe.named_steps["pre"]
    rf_model = rf_pipe.named_steps["m"]
    feat_names = rf_pre.get_feature_names_out()
    fi_df = pd.DataFrame({"Feature": feat_names,
                           "Importance": rf_model.feature_importances_})\
              .sort_values("Importance", ascending=False)

    return (fitted, results_df, fs_df, best_fs, feature_cols,
            X_train, X_test, y_train, y_test, rf_pipe, fi_df, num_f, cat_f)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 CO₂ Dashboard")
    st.markdown("**ADEME Car Labelling Dataset**")
    st.markdown("---")
    st.caption("Dataset is loaded automatically from the repository.")
    uploaded = st.file_uploader("Eigene CSV hochladen (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("**Projekt:** [GitHub ↗](https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction)",
                unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
source = uploaded.read() if uploaded is not None else CSV_URL

with st.spinner("Loading and preprocessing data …"):
    try:
        df      = load_and_preprocess(source)
        df_unique = make_df_unique(df)
        df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy() if 'Fuel' in df.columns else df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

with st.sidebar:
    st.markdown("---")
    st.caption(f"Raw data: {len(df):,} rows")
    st.caption(f"Unique (ES/GO): {len(df_unique):,} configurations")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Preprocessing",
    "📊 EDA",
    "🔗 Correlation Analysis",
    "📉 Deduplication",
    "🔵 Clustering",
    "🤖 Prediction",
    "🎯 CO₂ Calculator",
])

# ═══════════════════════ TAB 0 – PREPROCESSING ═══════════════════════════════
with tabs[0]:
    st.header("📋 Preprocessing & Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Spalten", len(df.columns))
    n_esgo = len(df_combus)
    c3.metric("ES+GO Vehicles", f"{n_esgo:,}")
    c4.metric("Unique Configurations", f"{len(df_unique):,}")

    st.markdown("---")

    # Missing values heatmap + barplot (wie im Notebook)
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    missing_sorted = missing_values[missing_values > 0].sort_values(ascending=False)

    if len(missing_sorted) > 0:
        cols_with_na = missing_sorted.index.tolist()
        from matplotlib.colors import ListedColormap
        cmap_mv = ListedColormap(["lightgrey", "#1f77b4"])

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        sns.heatmap(df[cols_with_na].isna(), cmap=cmap_mv, cbar=False,
                    yticklabels=False, ax=axes[0])
        axes[0].set_title("Missing Values Pattern (Blue = Missing)")
        axes[0].tick_params(axis='x', rotation=90)

        sns.barplot(x=missing_sorted.index, y=missing_sorted.values,
                    color=BLUE, ax=axes[1])
        axes[1].set_title("Number of Missing Values per Column")
        axes[1].tick_params(axis='x', rotation=90)

        plt.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.success("No missing values after preprocessing!")

    st.markdown("---")
    st.subheader("Dataset Summary")
    desc = df.describe(include='all').T
    # Only format numeric columns to avoid ValueError on string columns
    num_cols_desc = desc.select_dtypes(include='number').columns.tolist()
    fmt = {c: "{:.2f}" for c in num_cols_desc}
    st.dataframe(desc.style.format(fmt, na_rep="-"), width='stretch')

    st.subheader("First Rows")
    st.dataframe(df.head(10), width='stretch')


# ═══════════════════════ TAB 1 – EDA ═════════════════════════════════════════
with tabs[1]:
    st.header("📊 Fleet-wide Distribution and Frequency Analysis")

    # CO2 analysis before deduplication (2x2 wie im Notebook)
    st.subheader("CO₂ Emissions Analysis – Target Variable (before Deduplication)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle('CO2 Emissions Analysis - Target Variable', fontsize=16, fontweight='bold')

    co2_data = df['CO2 (g/km)'].dropna()

    axes[0,0].hist(co2_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(co2_data.mean(), color='red', linestyle='--',
                      label=f'Mean: {co2_data.mean():.1f}')
    axes[0,0].axvline(co2_data.median(), color='green', linestyle='--',
                      label=f'Median: {co2_data.median():.1f}')
    axes[0,0].set_title('Distribution of CO2 Emissions')
    axes[0,0].set_xlabel('CO2 (g/km)'); axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()

    axes[0,1].boxplot(co2_data)
    axes[0,1].set_title('CO2 Emissions Box Plot')
    axes[0,1].set_ylabel('CO2 (g/km)')

    stats.probplot(co2_data, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot (Normality Check)')

    axes[1,1].axis('off')
    txt = (f"Mean: {co2_data.mean():.1f} g/km\n"
           f"Median: {co2_data.median():.1f} g/km\n"
           f"Std: {co2_data.std():.1f} g/km\n"
           f"Min: {co2_data.min():.0f} g/km\n"
           f"Max: {co2_data.max():.0f} g/km\n"
           f"N: {len(co2_data):,}")
    axes[1,1].text(0.5, 0.5, txt, fontsize=14, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8))
    axes[1,1].set_title('Summary Statistics')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # 5x2 Fleet Distribution (wie im Notebook)
    st.subheader("Fleet-wide Distribution and Frequency Analysis")
    fig, axes = plt.subplots(5, 2, figsize=(20, 30))
    fig.suptitle('Fleet-wide Distribution and Frequency Analysis', fontsize=20, y=1.01)

    sns.histplot(df['Empty Mass Euro Avg (kg)'], bins=100, kde=True, ax=axes[0,0], color=BLUE)
    axes[0,0].set_title('Distribution of Vehicle Mass (kg)', fontsize=14)

    sns.histplot(df['Maximum Power (kW)'], bins=100, kde=True, ax=axes[0,1], color=BLUE)
    axes[0,1].set_title('Distribution of Maximum Power (kW)', fontsize=14)

    sns.histplot(df['Combined Consumption (l/100km)'].dropna(), bins=100, kde=True,
                 ax=axes[1,0], color=BLUE)
    axes[1,0].set_title('Combined Consumption Distribution (l/100km)', fontsize=14)

    sns.histplot(df['CO2 (g/km)'].dropna(), bins=100, kde=True, ax=axes[1,1], color=BLUE)
    axes[1,1].set_title('Distribution of CO2 Emissions (g/km)', fontsize=14)

    if 'Fuel' in df.columns:
        sns.countplot(data=df, x='Fuel', ax=axes[2,0], color=BLUE,
                      order=df['Fuel'].value_counts().index)
        axes[2,0].set_title('Fuel Type Frequency', fontsize=14)

    if 'Body' in df.columns:
        sns.countplot(data=df, x='Body', ax=axes[2,1], color=BLUE,
                      order=df['Body'].value_counts().index)
        axes[2,1].set_title('Body Type Frequency', fontsize=14)
        axes[2,1].tick_params(axis='x', rotation=45)

    if 'Gearbox' in df.columns:
        sns.countplot(data=df, x='Gearbox', ax=axes[3,0], color=BLUE,
                      order=df['Gearbox'].value_counts().index)
        axes[3,0].set_title('Gearbox Frequency', fontsize=14)

    if 'Range' in df.columns:
        sns.countplot(data=df, x='Range', ax=axes[3,1], color=BLUE,
                      order=df['Range'].value_counts().index)
        axes[3,1].set_title('Vehicle Range Frequency', fontsize=14)
        axes[3,1].tick_params(axis='x', rotation=30)

    top_brands = df['Brand'].value_counts().nlargest(25)
    sns.barplot(x=top_brands.values, y=top_brands.index, ax=axes[4,0], color=BLUE)
    axes[4,0].set_title('Top 25 Brands by Frequency', fontsize=14)

    if 'Commerical Designation' in df.columns:
        top_models = df['Commerical Designation'].value_counts().nlargest(15)
        sns.barplot(x=top_models.values, y=top_models.index, ax=axes[4,1], color=BLUE)
        axes[4,1].set_title('Top 15 Vehicle Models by Frequency', fontsize=14)

    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Primary Drivers (4x2 wie im Notebook)
    st.subheader("Primary Drivers of CO₂ Emissions")
    fig, axes = plt.subplots(4, 2, figsize=(20, 28))
    fig.suptitle('Finalized Analysis: Primary Drivers of CO2 Emissions', fontsize=20, y=1.01)

    sns.histplot(df['CO2 (g/km)'].dropna(), bins=50, kde=True, ax=axes[0,0], color=BLUE)
    axes[0,0].set_title('Distribution of CO2 Emissions', fontsize=14)
    axes[0,0].set_xlabel('CO2 (g/km)')

    sns.scatterplot(data=df, x='Empty Mass Euro Avg (kg)', y='CO2 (g/km)',
                    alpha=0.4, ax=axes[0,1], color=BLUE)
    axes[0,1].set_title('Vehicle Mass vs CO2 Emissions', fontsize=14)

    sns.scatterplot(data=df, x='Maximum Power (kW)', y='CO2 (g/km)',
                    alpha=0.4, ax=axes[1,0], color=BLUE)
    axes[1,0].set_title('Maximum Power vs CO2 Emissions', fontsize=14)

    sns.scatterplot(data=df, x='Combined Consumption (l/100km)', y='CO2 (g/km)',
                    alpha=0.4, ax=axes[1,1], color=BLUE)
    axes[1,1].set_title('Combined Consumption vs CO2 Emissions', fontsize=14)

    sns.boxplot(data=df, x='Fuel', y='CO2 (g/km)', ax=axes[2,0],
                hue='Fuel', palette='pastel', legend=False)
    axes[2,0].set_title('CO2 Emissions by Fuel Type', fontsize=14)

    sns.boxplot(data=df, x='Body', y='CO2 (g/km)', ax=axes[2,1],
                hue='Body', palette='pastel', legend=False)
    axes[2,1].set_title('CO2 Emissions by Body Type', fontsize=14)
    axes[2,1].tick_params(axis='x', rotation=45)

    sns.boxplot(data=df, x='Gearbox', y='CO2 (g/km)', ax=axes[3,0],
                hue='Gearbox', palette='pastel', legend=False)
    axes[3,0].set_title('CO2 Emissions by Gearbox Type', fontsize=14)
    axes[3,0].tick_params(axis='x', rotation=45)

    axes[3,1].axis('off')
    summary_text = ("Summary of Drivers:\n\n"
                    "1. Consumption: Highest correlation with CO2.\n"
                    "2. Mass/Power: Significant secondary drivers.\n"
                    "3. Fuel: Significant variance between types.\n"
                    "4. Gearbox/Body: Impact visible in distribution spreads.")
    axes[3,1].text(0.5, 0.5, summary_text, fontsize=14, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════ TAB 2 – KORRELATIONEN ═══════════════════════════════
with tabs[2]:
    st.header("🔗 Correlation & Statistical Analysis")

    # Pearson + Spearman heatmap (wie im Notebook)
    st.subheader("Pearson vs. Spearman Correlation Heatmap")
    st.markdown(
        "**Methodology:** Two correlation measures are computed and compared in parallel. "
        "**Pearson** measures linear relationships (assumes normality). "
        "**Spearman** measures monotonic relationships (rank-based, robust to outliers). "
        "Comparing both reveals where non-linear relationships exist \u2014 "
        "indicated by large differences between Pearson and Spearman coefficients."
    )
    df_numeric = df.select_dtypes(include=np.number).copy()
    pearson_corr  = df_numeric.corr(method='pearson')
    spearman_corr = df_numeric.corr(method='spearman')

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax[0])
    ax[0].set_title('Pearson Correlation Heatmap (Numeric-Only)')
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax[1])
    ax[1].set_title('Spearman Correlation Heatmap (Numeric-Only)')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(
        "**Interpretation:** CO\u2082 correlates most strongly with Combined Consumption "
        "(Pearson r=0.96, Spearman r=0.98) \u2014 near-perfect linear and monotonic relationship. "
        "Empty mass shows a strong Pearson correlation (r=0.69) and even stronger "
        "Spearman (r=0.65) \u2014 the relationship is predominantly monotonic. "
        "Maximum power has moderate Pearson (r=0.36) but weaker Spearman (r=0.18) "
        "\u2014 indicates a non-linear relationship. "
        "HC and NOX correlate negatively with CO\u2082 (r\u2248-0.17) \u2014 diesel vehicles emit "
        "more NOX at lower CO\u2082 than petrol vehicles."
    )

    st.markdown("---")

    # Detailed scatter: Mass, Consumption, Power vs CO2 (2x2 each)
    SCATTER_INTERP = {
        "Empty Mass": (
            "**Interpretation:** Strong positive correlation (Pearson r=0.68, Spearman r=0.78, R²=0.46). "
            "46% of CO₂ variance is explained by empty mass alone. "
            "Spearman exceeds Pearson — slightly non-linear relationship: "
            "for very heavy vehicles (>2,500 kg) the CO₂ increase per kg diminishes. "
            "The hexbin shows data density at 1,200–2,000 kg / 100–220 g/km — "
            "the core market of compact and mid-range vehicles."
        ),
        "Combined Consumption": (
            "**Interpretation:** Near-perfect linear correlation (Pearson r=0.98, R²=0.96). "
            "96% of CO₂ variance is explained by fuel consumption — physically expected, "
            "as CO₂ is directly proportional to combustion (petrol ≈ 2.31 kg/l, diesel ≈ 2.64 kg/l). "
            "The narrow data path in the hexbin confirms a quasi-deterministic relationship. "
            "Note: Combined Consumption is deliberately excluded from the prediction model — "
            "including it would reduce the model to a trivial conversion factor."
        ),
        "Maximum Power": (
            "**Interpretation:** Moderate correlation (Pearson r=0.67, Spearman r=0.54, R²=0.45). "
            "Pearson notably higher than Spearman — predominantly linear relationship with high scatter. "
            "High-performance vehicles (>300 kW) span 200–550 g/km — "
            "power alone explains CO₂ less precisely than mass, "
            "because power is strongly correlated with mass, which is the actual physical driver."
        ),
    }

    for var_name, var_col in [
        ("Empty Mass", "Empty Mass Euro Avg (kg)"),
        ("Combined Consumption", "Combined Consumption (l/100km)"),
        ("Maximum Power", "Maximum Power (kW)"),
    ]:
        if var_col not in df_unique.columns:
            continue
        st.subheader(f"{var_name} vs CO₂ (Deduplicated)")
        d = df_unique[[var_col, 'CO2 (g/km)']].dropna()
        pc, _ = pearsonr(d[var_col], d['CO2 (g/km)'])
        sc, _ = spearmanr(d[var_col], d['CO2 (g/km)'])
        r2 = pc**2

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))

        sns.regplot(x=d[var_col], y=d['CO2 (g/km)'],
                    scatter_kws={'alpha':0.6,'s':20,'color':'blue'},
                    line_kws={'color':'red','alpha':0.8,'linewidth':2},
                    ax=axes[0,0])
        axes[0,0].set_title(f'r = {pc:.4f}, R² = {r2:.4f}')
        axes[0,0].set_xlabel(var_col); axes[0,0].set_ylabel('CO2 (g/km)')
        axes[0,0].grid(True, alpha=0.3)

        hb = axes[0,1].hexbin(d[var_col], d['CO2 (g/km)'], gridsize=30, cmap='Blues')
        axes[0,1].set_title('Density Plot (Hexbin)')
        axes[0,1].set_xlabel(var_col); axes[0,1].set_ylabel('CO2 (g/km)')
        plt.colorbar(hb, ax=axes[0,1])

        axes[1,0].hist(d[var_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,0].set_xlabel(var_col); axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'Distribution of {var_col}')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].hist(d['CO2 (g/km)'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1,1].set_xlabel('CO2 (g/km)'); axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of CO2 Emissions')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout(); st.pyplot(fig); plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Pearson r", f"{pc:.4f}")
        col2.metric("Spearman r", f"{sc:.4f}")
        col3.metric("R²", f"{r2:.4f}")

        if var_name in SCATTER_INTERP:
            st.caption(SCATTER_INTERP[var_name])

        st.markdown("---")


# ═══════════════════════ TAB 3 – DEDUPLICATION ═══════════════════════════════
with tabs[3]:
    st.header("📉 Data Deduplication – Unique Mechanical Configurations")

    st.markdown("""
    ### Why deduplicate?

    The raw ADEME dataset contains **44,850 records** — but most of them are not unique vehicles.
    The same mechanical configuration (e.g. a Mercedes Viano 2.2 CDI with 120 kW, 2,130 kg,
    manual 6-speed, 200 g/km CO₂) appears hundreds of times under different trim names,
    option packages or registration variants. Including these duplicates would **bias every
    statistical analysis and machine learning model** towards the most common configurations
    (in this dataset: Mercedes-Benz Minibuses dominate ~86% of raw records).

    ### How it works — three steps

    **Step 1 — Fuel filter:**
    Only petrol (`ES`) and diesel (`GO`) vehicles are kept. Electric, hybrid and gas
    vehicles are excluded because they follow fundamentally different emission physics
    and would require separate models. This reduces the dataset from **44,850 → 43,935 records**.

    **Step 2 — Define a unique mechanical configuration:**
    A vehicle is considered unique if it has a distinct combination of:
    `Brand · Folder Model · Fuel · Body · Gearbox · Maximum Power (kW) ·
    Empty Mass Euro Avg (kg) · CO₂ (g/km) · Combined Consumption (l/100km) · Range`

    This means: two cars with identical technical parameters but different commercial
    names (e.g. "Viano Trend" vs "Viano Ambiente") count as **one configuration**.

    **Step 3 — Group and count:**
    For each unique configuration, the number of duplicate rows (`Clone_Count`) is recorded.
    The resulting dataset contains **5,700 unique mechanical configurations** —
    the true analytical unit for understanding CO₂ emissions.

    > **Result:** {redundancy_pct:.1f}% of the filtered dataset were duplicates.
    > The most redundant configuration appeared **{top_clone:,} times** in the raw data.
    """.format(
        redundancy_pct=(len(df[df["Fuel"].isin(["ES","GO"])]) - len(df_unique)) /
                        len(df[df["Fuel"].isin(["ES","GO"])]) * 100
                        if len(df[df["Fuel"].isin(["ES","GO"])]) > 0 else 0,
        top_clone=int(df_unique["Clone_Count"].iloc[0]) if len(df_unique) > 0 else 0
    ))

    st.markdown("---")

    # ── Key metrics ──────────────────────────────────────────────────────────
    # total_obs      : all records in the raw dataset (all fuel types)
    # filtered_obs   : after keeping only petrol (ES) + diesel (GO)
    # unique_designs : after groupby on UNIQUE_COLS — the true analytical unit
    # redundancy_pct : share of filtered records that were duplicates
    # top_clone      : largest Clone_Count — most repeated configuration
    total_obs          = len(df)
    filtered_obs       = len(df_combus)
    unique_designs     = len(df_unique)
    filtered_out       = total_obs - filtered_obs
    duplicates_removed = filtered_obs - unique_designs
    redundancy_pct     = (duplicates_removed / filtered_obs * 100) if filtered_obs > 0 else 0
    top_clone          = int(df_unique['Clone_Count'].iloc[0]) if len(df_unique) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{total_obs:,}")
    c2.metric("ES+GO Filter",     f"{filtered_obs:,}")
    c3.metric("Unique Designs",   f"{unique_designs:,}")
    c4.metric("Redundancy Rate",  f"{redundancy_pct:.1f}%")

    # ── Bar chart: Engineering Fleet Diversity ───────────────────────────────
    # Visualises the reduction from raw records → unique mechanical designs.
    # The annotation box shows redundancy %, unique count and total count.
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ['Total Records in Data', 'Unique Mechanical Designs']
    values = [total_obs, unique_designs]
    sns.barplot(x=categories, y=values, palette="viridis", ax=ax, hue=categories, legend=False)
    for i, v in enumerate(values):
        ax.text(i, v + total_obs * 0.01, f'{v:,}', ha='center', va='bottom',
                fontweight='bold', fontsize=14)
    stats_text = (f"Redundancy: {redundancy_pct:.1f}%\n"
                  f"Unique: {unique_designs:,}\nTotal: {total_obs:,}")
    ax.text(1.1, total_obs * 0.70, stats_text, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.4'))
    ax.set_title('Engineering Fleet Diversity', fontsize=18, pad=20)
    ax.set_ylabel('Number of Observations', fontsize=12)
    ax.set_ylim(0, total_obs * 1.15)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── CO₂ analysis after deduplication ────────────────────────────────────
    # Re-plots the CO₂ distribution on df_unique instead of df_combus.
    # Removing duplicates shifts the distribution: the mean drops because
    # Mercedes-Benz Minibus clones (high CO₂, high Clone_Count) are collapsed
    # into single rows — revealing the true spread across vehicle types.
    st.subheader("CO₂ Analysis after Deduplication")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle('CO2 Emissions Analysis (Deduplicated Data)', fontsize=16, fontweight='bold')

    co2_u = df_unique['CO2 (g/km)'].dropna()
    axes[0,0].hist(co2_u, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(co2_u.mean(), color='red', linestyle='--',
                      label=f'Mean: {co2_u.mean():.1f}')
    axes[0,0].axvline(co2_u.median(), color='green', linestyle='--',
                      label=f'Median: {co2_u.median():.1f}')
    axes[0,0].set_title('Distribution of CO2 Emissions (Deduplicated)')
    axes[0,0].set_xlabel('CO2 (g/km)'); axes[0,0].legend()

    axes[0,1].boxplot(co2_u)
    axes[0,1].set_title('CO2 Emissions Box Plot (Deduplicated)')
    axes[0,1].set_ylabel('CO2 (g/km)')

    stats.probplot(co2_u, dist="norm", plot=axes[1,0])
    axes[1,0].set_title('Q-Q Plot (Normality Check - Deduplicated)')

    axes[1,1].axis('off')
    txt = (f"Mean: {co2_u.mean():.1f} g/km\nMedian: {co2_u.median():.1f} g/km\n"
           f"Std: {co2_u.std():.1f} g/km\nN unique: {len(co2_u):,}")
    axes[1,1].text(0.5, 0.5, txt, fontsize=14, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Outlier analysis (IQR method) ────────────────────────────────────────
    # IQR (Interquartile Range) method: outliers are values below Q1 - 1.5×IQR
    # or above Q3 + 1.5×IQR.  Applied separately to CO₂, Power and Mass.
    # Outliers are NOT removed — they are real vehicles (e.g. Lexus LFA, 379 g/km).
    # They are documented here for transparency and to inform model interpretation.
    st.subheader("Outlier Analysis (IQR Method)")
    for col_name in ['CO2 (g/km)', 'Maximum Power (kW)', 'Empty Mass Euro Avg (kg)']:
        if col_name not in df_unique.columns:
            continue
        Q1 = df_unique[col_name].quantile(0.25)
        Q3 = df_unique[col_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_unique[(df_unique[col_name] < Q1 - 1.5*IQR) |
                              (df_unique[col_name] > Q3 + 1.5*IQR)]
        st.markdown(f"**{col_name}**: {len(outliers)} outliers "
                    f"(IQR bounds: >{Q3 + 1.5*IQR:.1f} or <{Q1 - 1.5*IQR:.1f})")

    st.subheader("Top 5 Redundant Mechanical Bases")
    st.dataframe(df_unique.head(5), width='stretch')


# ═══════════════════════ TAB 4 – CLUSTERING ══════════════════════════════════
with tabs[4]:
    st.header("🔵 K-Prototypes Clustering")

    st.markdown("""
    > **Research Question:** Which natural vehicle segments can be identified based on
    > technical characteristics (fuel type, body style, gearbox, power, mass)
    > in the French vehicle market 2013, and how do these segments differ
    > in their CO₂ emissions?
    """)

    # ── Elbow Method ─────────────────────────────────────────────────────────
    st.subheader("Elbow Method – Optimal Number of Clusters")
    st.markdown(
        "The Elbow Method computes the **total cost** (intra-cluster distance) "
        "for k=2 to k=9. The 'elbow' in the cost curve indicates the optimal k."
    )

    @st.cache_data(show_spinner=False)
    def compute_elbow(_df: pd.DataFrame):
        if not KPROTO_AVAILABLE:
            return None
        categorical_cols = [c for c in ["Body", "Fuel", "Gearbox"] if c in _df.columns]
        numeric_cols     = [c for c in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)"] if c in _df.columns]
        feature_cols     = categorical_cols + numeric_cols
        target_col       = "CO2 (g/km)"

        df_c = _df[feature_cols + [target_col]].dropna().copy()
        scaler = StandardScaler()
        df_kp  = df_c.copy()
        df_kp[numeric_cols] = scaler.fit_transform(df_kp[numeric_cols])
        for col in categorical_cols:
            df_kp[col] = df_kp[col].astype(str)

        X_matrix        = df_kp[feature_cols].to_numpy(dtype=object)
        categorical_idx = [feature_cols.index(col) for col in categorical_cols]

        costs = []
        k_range = range(2, 10)
        for k_val in k_range:
            model = KPrototypes(
                n_clusters=k_val, init="Cao",
                n_init=3, verbose=0, random_state=RANDOM_STATE
            )
            model.fit_predict(X_matrix, categorical=categorical_idx)
            costs.append(model.cost_)
        return list(k_range), costs

    with st.spinner("Computing Elbow Method (k=2–9) …"):
        elbow_result = compute_elbow(df_unique)

    if elbow_result is not None:
        k_range, costs = elbow_result
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_range, costs, marker="o", color=BLUE, linewidth=2, markersize=8)
        ax.fill_between(k_range, costs, alpha=0.1, color=BLUE)

        # Mark the elbow: largest second derivative
        diffs2 = np.diff(np.diff(costs))
        elbow_k = k_range[np.argmax(diffs2) + 1]
        ax.axvline(elbow_k, color="red", lw=1.5, linestyle="--",
                   label=f"Recommended k = {elbow_k}")
        ax.scatter([elbow_k], [costs[k_range.index(elbow_k)]],
                   color="red", zorder=5, s=120)

        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Cost (intra-cluster distance)")
        ax.set_title("Elbow Method for K-Prototypes", fontsize=13)
        ax.set_xticks(list(k_range))
        ax.legend()
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"Elbow method suggests k = {elbow_k}. This analysis uses k = 4 for richer segment granularity.")
    else:
        st.warning("Elbow Method requires the kmodes package.")

    st.markdown("---")

    k = st.slider("Number of Clusters (k)", 2, 8, 4)
    with st.spinner("Clustering in progress …"):
        df_cluster_raw = run_clustering(df_unique, k=k)

    cluster_order = sorted(df_cluster_raw['Cluster'].unique())
    palette_clust = sns.color_palette("Paired", n_colors=len(cluster_order))
    cluster_colors = dict(zip(cluster_order, palette_clust))

    st.subheader("Cluster Sizes")
    print(df_cluster_raw['Cluster'].value_counts().sort_index())

    # 2x2 Dashboard (wie im Notebook)
    fleet_mean = df_cluster_raw['CO2 (g/km)'].mean()
    cluster_means = df_cluster_raw.groupby('Cluster', as_index=False)['CO2 (g/km)'].mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    sns.countplot(data=df_cluster_raw, x='Cluster', hue='Cluster',
                  hue_order=cluster_order, palette=cluster_colors, ax=ax1, legend=False)
    ax1.set_title("Observations per Cluster", pad=15)
    ax1.set_xlabel("Cluster"); ax1.set_ylabel("Count")

    sns.boxplot(data=df_cluster_raw, x='Cluster', y='CO2 (g/km)', hue='Cluster',
                hue_order=cluster_order, palette=cluster_colors, ax=ax2, legend=False)
    ax2.set_title("CO2 Distribution per Cluster", pad=15)
    ax2.set_xlabel("Cluster"); ax2.set_ylabel("CO2 (g/km)")

    sns.scatterplot(data=df_cluster_raw, x='Maximum Power (kW)', y='Empty Mass Euro Avg (kg)',
                    hue='Cluster', hue_order=cluster_order, palette=cluster_colors,
                    alpha=0.6, ax=ax3)
    ax3.set_title("Power vs Mass", pad=15)
    ax3.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.barplot(data=cluster_means, x='Cluster', y='CO2 (g/km)', hue='Cluster',
                hue_order=cluster_order, palette=cluster_colors, ax=ax4, legend=False)
    ax4.axhline(fleet_mean, color='red', linestyle='--', label=f'Fleet Avg {fleet_mean:.1f}')
    ax4.set_title("Mean CO2 per Cluster", pad=15); ax4.legend()

    plt.subplots_adjust(hspace=0.5, wspace=0.35)
    st.pyplot(fig); plt.close()

    # ── Cluster-Interpretation ────────────────────────────────────────────────
    st.markdown("#### Cluster Interpretation")
    st.markdown("""
    The visualisations reveal **four distinct vehicle segments**:

    | Cluster | Size | Avg CO₂ | Profile |
    |---------|-------|--------|--------|
    | **0** | ~2,400 (42%) | ~148 g/km | Light mid-range — low mass & power, below fleet average |
    | **1** | ~1,430 (25%) | ~210 g/km | Heavy commercial — high mass, mostly diesel, well above fleet average |
    | **2** | ~1,130 (20%) | ~126 g/km | **Efficiency cluster** — lowest CO₂, light petrol vehicles |
    | **3** | ~740 (13%)   | ~243 g/km | High-performance — highest power & mass, widest spread |

    **Fleet average: 171.3 g/km** — Clusters 0 and 2 are clearly below, Clusters 1 and 3 above.

    **Answer to the research question:** Yes, natural vehicle segments can be identified.
    The strongest driver of cluster membership is the combination of **mass and power** —
    visible in the Power-vs-Mass scatterplot. Fuel type and body style provide additional differentiation:
    Cluster 2 (efficiency) is strongly petrol-dominated, Cluster 1 (commercial) almost exclusively diesel.
    """)

    st.markdown("---")

    # Categorical distribution per cluster (wie im Notebook)
    st.subheader("Categorical Distribution per Cluster")
    cat_cols_clust = [c for c in ['Body', 'Fuel', 'Gearbox'] if c in df_cluster_raw.columns]
    fig, axes = plt.subplots(1, len(cat_cols_clust), figsize=(18, 6))
    if len(cat_cols_clust) == 1: axes = [axes]
    for i, feature in enumerate(cat_cols_clust):
        counts = pd.crosstab(df_cluster_raw['Cluster'], df_cluster_raw[feature])
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        pct.plot(kind='bar', stacked=True, colormap='viridis', ax=axes[i])
        axes[i].set_title(f'{feature} Distribution', pad=15)
        axes[i].set_xlabel('Cluster'); axes[i].set_ylabel('Percentage (%)')
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].legend(title=feature, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    st.pyplot(fig); plt.close()

    st.caption(
        "Interpretation: Cluster 2 is almost exclusively petrol (ES) — the efficiency cluster "
        "consists mainly of light petrol saloons with manual gearbox (M 5/M 6). "
        "Cluster 1 is nearly entirely diesel (GO) — heavy minibuses and vans dominate this segment. "
        "Cluster 3 shows the broadest body style variety (Berline, Break, TS TERRAINS/CHEMINS) — "
        "typical for high-performance vehicles across all categories."
    )

    st.markdown("---")

    # Radar + Heatmap (wie im Notebook)
    st.subheader("Cluster Profiles – Radar & Heatmap")
    profile_features = ['Maximum Power (kW)', 'Empty Mass Euro Avg (kg)', 'CO2 (g/km)']
    profile_labels = ['Power', 'Mass', 'CO2']

    df_profile = df_cluster_raw.copy()
    for f in profile_features:
        mn, mx = df_profile[f].min(), df_profile[f].max()
        df_profile[f] = (df_profile[f] - mn) / (mx - mn + 1e-9)

    cluster_profiles = df_profile.groupby('Cluster')[profile_features].mean()
    cluster_profiles.columns = profile_labels
    cluster_profiles = cluster_profiles.reindex(cluster_order)

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])

    ax_radar = fig.add_subplot(gs[0], projection='polar')
    labels = profile_labels
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for cid in cluster_profiles.index:
        vals = cluster_profiles.loc[cid].tolist() + [cluster_profiles.loc[cid].tolist()[0]]
        col = cluster_colors.get(cid, "gray")
        ax_radar.plot(angles, vals, color=col, linewidth=2, label=f'Cluster {cid}')
        ax_radar.fill(angles, vals, color=col, alpha=0.12)
    ax_radar.set_theta_offset(np.pi/2); ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(labels, fontsize=11)
    ax_radar.set_ylim(0, 1); ax_radar.set_title("Cluster Profiles (Radar)", fontsize=13, pad=30)
    ax_radar.legend(loc='upper left', bbox_to_anchor=(1.10, 1.02), frameon=False, fontsize=10)

    ax_heat = fig.add_subplot(gs[1])
    hm_data = cluster_profiles.values
    nr, nc = hm_data.shape
    ax_heat.imshow(hm_data, cmap='Greys', aspect='auto', alpha=0.20)
    ax_heat.set_xticks(np.arange(nc)); ax_heat.set_xticklabels(profile_labels, fontsize=11)
    ax_heat.set_yticks(np.arange(nr))
    ax_heat.set_yticklabels([f'Cluster {c}' for c in cluster_profiles.index], fontsize=11)
    for i, cid in enumerate(cluster_profiles.index):
        col = cluster_colors.get(cid, "gray")
        for j in range(nc):
            ax_heat.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color=col, alpha=0.35))
            ax_heat.text(j, i, f"{hm_data[i,j]:.2f}", ha='center', va='center', fontsize=10)
    ax_heat.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax_heat.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax_heat.tick_params(which='minor', bottom=False, left=False)
    ax_heat.set_title("Cluster Profiles (Heatmap)", fontsize=13, pad=12)

    fig.suptitle("Vehicle Cluster Profiles Dashboard", fontsize=16, y=1.02)
    st.pyplot(fig); plt.close()

    st.caption(
        "Radar & Heatmap show normalised values (0=minimum, 1=maximum of the dataset). Cluster 3 has the highest normalised values across all three dimensions (Power=0.50, Mass=0.47, CO₂=0.33) — this is the high-performance segment. Cluster 2 has the lowest values (Power=0.07, Mass=0.18, CO₂=0.09) — this is the efficiency cluster. Cluster 1 stands out with an unusually high Mass score (0.67) at moderate power (0.13) — typical for heavy commercial vehicles with diesel engines."
    )


# ═══════════════════════ TAB 5 – PREDICTION ══════════════════════════════════
with tabs[5]:
    st.header("🤖 Predictive Modeling")

    st.markdown("""
    > **Research Question:** What is the relative contribution of vehicle mass, engine power,
    > fuel type, body style and gearbox type in explaining CO₂ emissions,
    > and which minimal feature set achieves the best predictive performance?
    """)

    with st.spinner("Training models (Feature Sets + CV + 5 models) …"):
        try:
            (fitted, results_df, fs_df, best_fs, feature_cols,
             X_train, X_test, y_train, y_test,
             rf_pipe, fi_df, num_f, cat_f) = train_all_models(df_unique)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    # best_model_name available for the rest of this tab
    best_model_name = results_df.iloc[0]["Model"]

    # ── 1. Feature Set Comparison ────────────────────────────────────────────
    st.subheader("1️⃣ Feature Set Comparison (5-Fold CV, Random Forest)")
    st.markdown(
        "**Methodology:** Four feature combinations are compared using **5-fold cross-validation** "
        "with a Random Forest. MAE measures the average deviation in g/km "
        "(**lower is better**). This selects the most informative feature set without overfitting risk."
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_fs = fs_df.sort_values("CV_MAE_mean", ascending=True)
    ax.barh(plot_fs["Feature_Set"], plot_fs["CV_MAE_mean"], color=BLUE, alpha=0.9)
    ax.set_xlabel("CV MAE (lower = better)"); ax.set_ylabel("Feature Set")
    ax.set_title("Feature Set Comparison (5-Fold CV, Random Forest)")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.success(f"✅ Best Feature Set: **{best_fs}** | Features: {', '.join(feature_cols)}")
    st.dataframe(fs_df[["Feature_Set","Features","CV_MAE_mean","CV_MAE_std"]]
                 .style.format({"CV_MAE_mean": "{:.2f}", "CV_MAE_std": "{:.2f}"}),
                 width='stretch')
    st.caption(
        "Interpretation: `all_features` (mass + power + fuel + gearbox + body) "
        "achieves the lowest MAE — every feature contributes to predictive performance. "
        "Removing body type (`no_body`) costs ~0.6 g/km, "
        "dropping gearbox (`mass_power_fuel`) adds ~3 g/km more error."
    )

    st.markdown("---")

    # ── 2. Model Performance ─────────────────────────────────────────────────
    st.subheader(f"2️⃣ Model Comparison: R² and MAE ({best_fs})")
    st.markdown(
        "Five models are evaluated on the same train/test split (80/20). "
        "**R\u00b2** measures the proportion of explained variance (1.0 = perfect). "
        "**MAE** is the average deviation in g/km. "
        "A large gap between train and test metrics indicates **overfitting**."
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_r = results_df.sort_values("Test_R2", ascending=True)
    axes[0].barh(plot_r["Model"], plot_r["Test_R2"], color=BLUE, alpha=0.9)
    axes[0].set_title(f"Test R² ({best_fs})"); axes[0].set_xlabel("Test R²")
    axes[1].barh(plot_r["Model"], plot_r["Test_MAE"], color=BLUE, alpha=0.9)
    axes[1].set_title(f"Test MAE ({best_fs})"); axes[1].set_xlabel("Test MAE")
    for ax in axes:
        ax.grid(False)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    fig.suptitle(f"Regression Model Performance – {best_fs}", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95]); st.pyplot(fig); plt.close()

    num_cols_res = results_df.select_dtypes(include='number').columns.tolist()
    st.dataframe(results_df.style
                 .highlight_max(subset=["Test_R2"], color="#c8e6c9")
                 .highlight_min(subset=["Test_MAE"], color="#c8e6c9")
                 .format({c: "{:.4f}" for c in num_cols_res}),
                 width='stretch')
    st.caption(
        "Interpretation: Gradient Boosting and Random Forest achieve R²≈0.95 at ~7–8 g/km MAE — "
        "meaning the model explains 95% of CO₂ variance with an average error of just 7 g/km. "
        "Linear models (Ridge, Lasso, Linear Regression) plateau at R\u00b2\u22480.86, "
        "as they cannot capture non-linear relationships (e.g. mass \u00d7 power interactions)."
    )

    st.markdown("---")

    # ── 3. Feature Importance ────────────────────────────────────────────────
    st.subheader(f"3️⃣ Feature Importance – Random Forest ({best_fs})")
    st.markdown(
        "Feature Importance (Mean Decrease Impurity) measures how strongly each feature "
        "contributes to reducing the prediction error. "
        "Categorical features were one-hot encoded "
        "(e.g. `cat__Fuel_GO`, `cat__Body_MINIBUS`). "
        "Numerical features contribute directly (`num__` prefix)."
    )
    top15 = fi_df.head(15).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.barh(top15["Feature"], top15["Importance"], color=BLUE, alpha=0.9)
    ax.set_xlabel("Importance"); ax.set_ylabel("Feature")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    fig.suptitle(f"Top 15 Random Forest Feature Importances ({best_fs})", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.95]); st.pyplot(fig); plt.close()
    st.dataframe(fi_df.head(15), width='stretch')
    st.caption(
        "Interpretation: **Empty mass (46.9%)** and **engine power (37.2%)** together dominate ~84% "
        "of explained variance — these are the primary physical drivers of CO₂ emissions. "
        "Gear count (4.6%) and fuel type (~2.6% each) provide additional predictive information. "
        "Body style and gearbox type play a minor role (<1% per feature)."
    )

    st.markdown("---")

    # ── 4. Partial Dependence Plots ──────────────────────────────────────────
    st.subheader("4️⃣ Partial Dependence Plots – Random Forest")
    st.markdown(
        "PDPs show the **marginal effect** of a single feature on the predicted CO\u2082 value \u2014 "
        "all other features are held at their mean (ceteris paribus). "
        "This reveals the isolated, non-linear influence of each individual feature."
        ""
    )
    # Fix: GearCount must be float for PDP
    X_train_pdp = X_train.copy()
    if "GearCount" in X_train_pdp.columns:
        X_train_pdp["GearCount"] = X_train_pdp["GearCount"].astype(float)

    pdp_features = [f for f in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
                                 "GearType", "GearCount"] if f in feature_cols]
    try:
        fig, ax = plt.subplots(figsize=(16, 7))
        PartialDependenceDisplay.from_estimator(
            rf_pipe, X_train_pdp, features=pdp_features,
            categorical_features=[f for f in cat_f if f in pdp_features], ax=ax)
        for axis in fig.axes:
            axis.grid(False)
            for sp in ["top","right"]: axis.spines[sp].set_visible(False)
        fig.suptitle(f"Partial Dependence Plots – Random Forest ({best_fs})", fontsize=16)
        fig.tight_layout(rect=[0,0,1,0.95]); st.pyplot(fig); plt.close()
        st.caption(
            "Interpretation: The CO₂ increase with mass and power is non-linear — "
            "the effect is stronger at lower values than at higher ones (diminishing marginal returns). "
            "More gears correlate slightly negatively with CO₂ (more efficient gear spacing). "
            "Automatic gearbox shows marginally higher CO₂ than manual — after controlling for all other features."
        )
    except Exception as e:
        st.warning(f"PDP not available: {e}")



# ═══════════════════════ TAB 6 – CO₂-RECHNER ═════════════════════════════════
with tabs[6]:
    st.header("🎯 CO₂ Calculator & Brand Comparison")
    st.markdown(
        "Choose your vehicle by **everyday criteria** — "
        "the app shows the **real CO₂ median** from comparable vehicles "
        "in the ADEME dataset and which brand is most efficient in your segment."
    )

    # ── Mappings: consumer language → dataset values ─────────────────────────
    # ── Mappings: AutoScout24-style Body + Fuel + PS ────────────────────────
    BODY_MAP = {
        "Saloon":          "BERLINE",
        "Estate":          "BREAK",
        "SUV / Off-Road":  "TS TERRAINS/CHEMINS",
        "Compact MPV":     "COMBISPACE",
        "Van / Minibus":   "MINIBUS",
        "Coupé":           "COUPE",
        "Convertible":     "CABRIOLET",
        "Minivan":         "MONOSPACE",
        "Small Minivan":   "MONOSPACE COMPACT",
        "City Van":        "MINISPACE",
    }

    BODY_INFO = {
        "Saloon":         ("~135 g/km", "1,610 configs"),
        "Estate":         ("~148 g/km", "699 configs"),
        "SUV / Off-Road": ("~162 g/km", "552 configs"),
        "Compact MPV":    ("~149 g/km", "258 configs"),
        "Van / Minibus":  ("~210 g/km", "1,167 configs"),
        "Coupé":          ("~179 g/km", "415 configs"),
        "Convertible":    ("~160 g/km", "297 configs"),
        "Minivan":        ("~149 g/km", "91 configs"),
        "Small Minivan":  ("~134 g/km", "190 configs"),
        "City Van":       ("~124 g/km", "72 configs"),
    }

    POWER_MAP = {
        "Up to 75 HP (≤55 kW)":      (0,    55),
        "76–130 HP (56–96 kW)":       (56,   96),
        "131–200 HP (97–147 kW)":     (97,  147),
        "Over 200 HP (>147 kW)":      (148, 600),
    }

    FUEL_MAP = {"Petrol": "ES", "Diesel": "GO"}
    GEAR_MAP = {"Manual": "M", "Automatic": "A"}

    # ── Input form ───────────────────────────────────────────────────────────
    with st.form("co2_consumer_form"):
        st.subheader("Your Vehicle")
        col1, col2 = st.columns(2)
        with col1:
            body_sel  = st.selectbox(
                "Body Style",
                list(BODY_MAP.keys()),
                index=0,
                help="Vehicle body type — same categories as AutoScout24 / mobile.de"
            )
            antrieb   = st.radio("Fuel Type", ["Petrol", "Diesel"], horizontal=True)
            getriebe  = st.radio("Gearbox",   ["Manual", "Automatic"], horizontal=True)
        with col2:
            power_sel = st.select_slider(
                "Engine Power",
                options=list(POWER_MAP.keys()),
                value="76–130 HP (56–96 kW)"
            )
            st.markdown("**Segment reference**")
            for body, (co2_ref, n_ref) in BODY_INFO.items():
                icon = "👉 " if body == body_sel else "　"
                st.caption(f"{icon}**{body}:** {co2_ref} median · {n_ref}")

        submitted = st.form_submit_button(
            "Calculate CO₂ & Compare Brands 🚀", use_container_width=True
        )

    if submitted:
        body_val    = BODY_MAP[body_sel]
        kw_range    = POWER_MAP[power_sel]
        fuel_val    = FUEL_MAP[antrieb]
        gear_prefix = GEAR_MAP[getriebe]
        ps_lo       = round(kw_range[0] * 1.36)
        ps_hi       = round(kw_range[1] * 1.36) if kw_range[1] < 600 else None

        # ── Filter: Body + Fuel + Power + Gearbox ────────────────────────────
        mask = (
            df_unique["Body"].eq(body_val) &
            df_unique["Fuel"].eq(fuel_val) &
            df_unique["Maximum Power (kW)"].between(kw_range[0], kw_range[1]) &
            df_unique["Gearbox"].astype(str).str.startswith(gear_prefix)
        )
        df_match = df_unique[mask].copy()

        # Fallback 1: without gearbox filter
        used_gear_filter = True
        if len(df_match) < 5:
            mask2 = (
                df_unique["Body"].eq(body_val) &
                df_unique["Fuel"].eq(fuel_val) &
                df_unique["Maximum Power (kW)"].between(kw_range[0], kw_range[1])
            )
            df_match = df_unique[mask2].copy()
            used_gear_filter = False

        # Fallback 2: without power filter
        used_power_filter = True
        if len(df_match) < 3:
            mask3 = (
                df_unique["Body"].eq(body_val) &
                df_unique["Fuel"].eq(fuel_val)
            )
            df_match = df_unique[mask3].copy()
            used_gear_filter  = False
            used_power_filter = False


        co2_vals = df_match["CO2 (g/km)"].dropna()

        if len(co2_vals) == 0:
            st.error("No vehicles found. Please try a different configuration.")
            st.stop()

        # ── Key metrics from real data ─────────────────────────────────────
        co2_median = co2_vals.median()
        co2_mean   = co2_vals.mean()
        co2_p25    = co2_vals.quantile(0.25)
        co2_p75    = co2_vals.quantile(0.75)
        co2_min    = co2_vals.min()
        co2_max    = co2_vals.max()
        n_match    = len(df_match)
        n_brands   = df_match["Brand"].nunique()

        # Fleet-wide median as reference
        fleet_median = df_unique["CO2 (g/km)"].median()
        pct_better   = (df_unique["CO2 (g/km)"] <= co2_median).mean() * 100
        delta_fleet  = co2_median - fleet_median
        jahres_co2   = co2_median * 15000 / 1000

        euro  = ("A (≤100 g/km)" if co2_median <= 100 else
                 "B (101–120)"   if co2_median <= 120 else
                 "C (121–140)"   if co2_median <= 140 else
                 "D (141–160)"   if co2_median <= 160 else
                 "E (161–200)"   if co2_median <= 200 else "F/G (>200)")
        color = "green" if co2_median <= 120 else "orange" if co2_median <= 160 else "red"

        st.markdown("---")
        st.subheader("Result")

        ps_str = f"{ps_lo}–{ps_hi} HP" if ps_hi else f"{ps_lo}+ HP"
        filter_info = f"{body_sel} · {antrieb} · {ps_str}"
        if used_gear_filter:
            filter_info += f" · {getriebe}"
        else:
            filter_info += " · all gearbox types"
        if not used_gear_filter:
            st.caption("Note: Gearbox filter was broadened (too few matches).")
        if not used_power_filter:
            st.caption(f"Note: Power filter was broadened — showing all {body_sel} {antrieb} vehicles.")

        # ── Result card ─────────────────────────────────────────────────────
        st.markdown(
            f"<div style='background:{color}22;border-left:6px solid {color};"
            f"padding:20px;border-radius:8px;margin:8px 0;'>"
            f"<h2 style='color:{color};margin:0'>"
            f"🚗 {co2_median:.0f} g CO₂/km <span style='font-size:16px;font-weight:normal'>"
            f"(Median from {n_match} real vehicles)</span></h2>"
            f"<p style='font-size:16px;margin:6px 0'>"
            f"EU Efficiency Class: <strong>{euro}</strong>"
            f"&nbsp;·&nbsp; Annual CO₂: approx. <strong>{jahres_co2:.0f} kg</strong> "
            f"(at 15,000 km/year)</p>"
            f"<p style='color:gray;font-size:12px;margin:0'>"
            f"Data: {filter_info} · {n_brands} brands · "
            f"Range: {co2_min:.0f}–{co2_max:.0f} g/km"
            f"</p></div>",
            unsafe_allow_html=True
        )

        # ── 4 Kennzahlen ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median CO₂", f"{co2_median:.0f} g/km")
        c2.metric("Range (25–75%)", f"{co2_p25:.0f}–{co2_p75:.0f} g/km")
        c3.metric("vs. Fleet", f"{delta_fleet:+.0f} g/km",
                  delta_color="inverse")
        c4.metric("Better than", f"{pct_better:.0f}% of all vehicles")

        # ── Segment distribution ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(df_unique["CO2 (g/km)"].dropna(), bins=60,
                color="lightgrey", alpha=0.7, label="All vehicles")
        ax.hist(co2_vals, bins=30, color=color, alpha=0.75,
                label=f"{body_sel} · {antrieb} ({n_match} vehicles)")
        ax.axvline(co2_median, color=color, lw=2.5, linestyle="--",
                   label=f"Segment median: {co2_median:.0f} g/km")
        ax.axvline(fleet_median, color="gray", lw=1.5, linestyle=":",
                   label=f"Fleet median: {fleet_median:.0f} g/km")
        ax.set_xlabel("CO₂ (g/km)"); ax.set_ylabel("Häufigkeit")
        ax.set_title("Your Segment vs. Full Fleet")
        ax.legend(fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Brand comparison ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏷️ Which brand is most efficient in your segment?")
        st.caption(f"{n_match} vehicles · {n_brands} brands · {filter_info}")

        brand_summary = (
            df_match.groupby("Brand")["CO2 (g/km)"]
            .agg(Modelle="count", CO2_Min="min",
                 CO2_Median="median", CO2_Mittel="mean", CO2_Max="max")
            .round(1)
            .sort_values("CO2_Median")
            .reset_index()
        )
        # Only brands with ≥2 models (unless too few results)
        if (brand_summary["Modelle"] >= 2).sum() >= 3:
            brand_summary = brand_summary[brand_summary["Modelle"] >= 2]

        top_n   = min(15, len(brand_summary))
        plot_b  = brand_summary.head(top_n)

        bar_colors = []
        for rank in range(len(plot_b)):
            if rank < 3:          bar_colors.append("#2ecc71")
            elif rank >= top_n-3: bar_colors.append("#e74c3c")
            else:                 bar_colors.append(BLUE)

        fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.55)))
        bars = ax.barh(plot_b["Brand"], plot_b["CO2_Median"],
                       color=bar_colors, alpha=0.85, edgecolor="white")

        for bar, val, n in zip(bars, plot_b["CO2_Median"], plot_b["Modelle"]):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f} g/km  ({int(n)} Modelle)",
                    va="center", fontsize=9)

        ax.axvline(co2_median, color="black", lw=1.5, linestyle="--",
                   label=f"Segment median: {co2_median:.0f} g/km")

        ax.set_xlabel("Median CO₂ (g/km)")
        ax.set_title(
            f"Brand Comparison: {body_sel} · {antrieb} · {getriebe} · {power_sel}",
            fontsize=12, pad=12
        )
        ax.set_xlim(0, plot_b["CO2_Median"].max() * 1.28)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#2ecc71", label="Top 3 most efficient brands"),
            Patch(facecolor=BLUE,      label="Mid-field"),
            Patch(facecolor="#e74c3c", label="Top 3 highest CO₂"),
            plt.Line2D([0],[0], color="black", lw=1.5, linestyle="--",
                       label=f"Segment median: {co2_median:.0f} g/km"),
        ]
        ax.legend(handles=legend_els, loc="lower right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Top 3 Empfehlungskarten ──────────────────────────────────────────
        st.subheader("🏆 Top 3 Recommendations")
        top3  = brand_summary.head(3)
        cols3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for idx, (col_ui, (_, rb)) in enumerate(zip(cols3, top3.iterrows())):
            saving    = co2_median - rb["CO2_Median"]
            saving_kg = saving * 15000 / 1000
            saving_str = (f"↓ {saving_kg:.0f} kg CO₂/year saved"
                          if saving > 1 else "At segment median")
            col_ui.markdown(
                f"<div style='background:#f0fdf4;border:2px solid #2ecc71;"
                f"padding:16px;border-radius:10px;text-align:center;'>"
                f"<div style='font-size:28px'>{medals[idx]}</div>"
                f"<div style='font-size:17px;font-weight:bold'>{rb['Brand']}</div>"
                f"<div style='font-size:24px;color:#2ecc71;font-weight:bold'>"
                f"{rb['CO2_Median']:.0f} g/km</div>"
                f"<div style='font-size:11px;color:gray;margin-top:4px'>"
                f"Min {rb['CO2_Min']:.0f} · Max {rb['CO2_Max']:.0f} g/km<br>"
                f"{int(rb['Modelle'])} models in segment<br>"
                f"<strong>{saving_str}</strong>"
                f"</div></div>",
                unsafe_allow_html=True
            )

        # ── Detailtabelle ────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 Show all matching vehicles"):
            show_cols = [c for c in
                         ["Brand", "Folder Model", "Fuel", "Body", "Gearbox",
                          "Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
                          "CO2 (g/km)", "Combined Consumption (l/100km)"]
                         if c in df_match.columns]
            disp = df_match[show_cols].copy()
            if "Maximum Power (kW)" in disp.columns:
                disp.insert(disp.columns.get_loc("Maximum Power (kW)")+1,
                            "HP", (disp["Maximum Power (kW)"] * 1.36).round(0).astype("Int64"))
            st.dataframe(
                disp.sort_values("CO2 (g/km)").reset_index(drop=True),
                width="stretch"
            )
