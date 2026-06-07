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
    'CO2 (g/km)', 'Combined Consumption (l/100km)'
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
        st.error("kmodes nicht installiert. Bitte requirements.txt mit 'kmodes>=0.12.2' updaten.")
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
    st.caption("Datensatz wird automatisch aus dem Repo geladen.")
    uploaded = st.file_uploader("Eigene CSV hochladen (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("**Projekt:** [GitHub ↗](https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction)",
                unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
source = uploaded.read() if uploaded is not None else CSV_URL

with st.spinner("Daten werden geladen und vorverarbeitet …"):
    try:
        df      = load_and_preprocess(source)
        df_unique = make_df_unique(df)
        df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy() if 'Fuel' in df.columns else df
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        st.stop()

with st.sidebar:
    st.markdown("---")
    st.caption(f"Rohdaten: {len(df):,} Zeilen")
    st.caption(f"Unique (ES/GO): {len(df_unique):,} Konfigurationen")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Preprocessing",
    "📊 EDA",
    "🔗 Korrelationen",
    "📉 Deduplication",
    "🔵 Clustering",
    "🤖 Prediction",
    "🎯 CO₂-Rechner",
])

# ═══════════════════════ TAB 0 – PREPROCESSING ═══════════════════════════════
with tabs[0]:
    st.header("📋 Preprocessing & Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gesamt Zeilen", f"{len(df):,}")
    c2.metric("Spalten", len(df.columns))
    n_esgo = len(df_combus)
    c3.metric("ES+GO Fahrzeuge", f"{n_esgo:,}")
    c4.metric("Unique Konfigurationen", f"{len(df_unique):,}")

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
        st.success("Keine fehlenden Werte nach Preprocessing!")

    st.markdown("---")
    st.subheader("Dataset Summary")
    desc = df.describe(include='all').T
    # Only format numeric columns to avoid ValueError on string columns
    num_cols_desc = desc.select_dtypes(include='number').columns.tolist()
    fmt = {c: "{:.2f}" for c in num_cols_desc}
    st.dataframe(desc.style.format(fmt, na_rep="-"), width='stretch')

    st.subheader("Erste Zeilen")
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
    df_numeric = df.select_dtypes(include=np.number).copy()
    pearson_corr  = df_numeric.corr(method='pearson')
    spearman_corr = df_numeric.corr(method='spearman')

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax[0])
    ax[0].set_title('Pearson Correlation Heatmap (Numeric-Only)')
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax[1])
    ax[1].set_title('Spearman Correlation Heatmap (Numeric-Only)')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Detailed scatter: Mass, Consumption, Power vs CO2 (2x2 each)
    for var_name, var_col in [
        ("Empty Mass", "Empty Mass Euro Avg (kg)"),
        ("Combined Consumption", "Combined Consumption (l/100km)"),
        ("Maximum Power", "Maximum Power (kW)"),
    ]:
        if var_col not in df_unique.columns:
            continue
        st.subheader(f"{var_name} vs CO₂ (Dedupliziert)")
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
        st.markdown("---")


# ═══════════════════════ TAB 3 – DEDUPLICATION ═══════════════════════════════
with tabs[3]:
    st.header("📉 Data Deduplication – Unique Mechanical Configurations")

    total_obs    = len(df)
    filtered_obs = len(df_combus)
    unique_designs = len(df_unique)
    filtered_out   = total_obs - filtered_obs
    duplicates_removed = filtered_obs - unique_designs
    redundancy_pct = (duplicates_removed / filtered_obs * 100) if filtered_obs > 0 else 0
    top_clone = int(df_unique['Clone_Count'].iloc[0]) if len(df_unique) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total_obs:,}")
    c2.metric("ES+GO Filter", f"{filtered_obs:,}")
    c3.metric("Unique Designs", f"{unique_designs:,}")
    c4.metric("Redundancy Rate", f"{redundancy_pct:.1f}%")

    # Bar chart: Engineering Fleet Diversity (wie im Notebook)
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

    # CO2 after dedup (2x2)
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

    # Outlier analysis
    st.subheader("Outlier Analysis (IQR Method)")
    for col_name in ['CO2 (g/km)', 'Maximum Power (kW)', 'Empty Mass Euro Avg (kg)']:
        if col_name not in df_unique.columns:
            continue
        Q1 = df_unique[col_name].quantile(0.25)
        Q3 = df_unique[col_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_unique[(df_unique[col_name] < Q1 - 1.5*IQR) |
                              (df_unique[col_name] > Q3 + 1.5*IQR)]
        st.markdown(f"**{col_name}**: {len(outliers)} Ausreißer "
                    f"(Grenze: >{Q3 + 1.5*IQR:.1f} oder <{Q1 - 1.5*IQR:.1f})")

    st.subheader("Top 5 Redundant Mechanical Bases")
    st.dataframe(df_unique.head(5), width='stretch')


# ═══════════════════════ TAB 4 – CLUSTERING ══════════════════════════════════
with tabs[4]:
    st.header("🔵 K-Prototypes Clustering")

    st.markdown("""
    > **Forschungsfrage:** Welche natürlichen Fahrzeugsegmente lassen sich anhand von
    > technischen Merkmalen (Antriebsart, Karosserie, Getriebe, Leistung, Masse)
    > im französischen Fahrzeugmarkt 2013 identifizieren, und wie unterscheiden sich
    > diese Segmente in ihrem CO₂-Ausstoß?
    """)

    # ── Elbow Method ─────────────────────────────────────────────────────────
    st.subheader("Elbow Method – Optimale Clusteranzahl")
    st.markdown(
        "Die Elbow-Methode berechnet die **Gesamtkosten** (intra-cluster distance) "
        "für k=2 bis k=9. Der 'Knick' im Kostenverlauf zeigt das optimale k."
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

    with st.spinner("Elbow-Methode wird berechnet (k=2–9) …"):
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
                   label=f"Empfohlenes k = {elbow_k}")
        ax.scatter([elbow_k], [costs[k_range.index(elbow_k)]],
                   color="red", zorder=5, s=120)

        ax.set_xlabel("Anzahl Cluster (k)")
        ax.set_ylabel("Kosten (intra-cluster distance)")
        ax.set_title("Elbow Method für K-Prototypes", fontsize=13)
        ax.set_xticks(list(k_range))
        ax.legend()
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"Empfohlene Clusteranzahl laut Elbow-Methode: **k = {elbow_k}**")
    else:
        st.warning("Elbow-Methode benötigt das kmodes-Paket.")

    st.markdown("---")

    k = st.slider("Anzahl Cluster (k)", 2, 8, 4)
    with st.spinner("Clustering läuft …"):
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
    st.markdown("#### Interpretation der Cluster")
    st.markdown("""
    Aus den Grafiken lassen sich **vier distinkte Fahrzeugsegmente** ablesen:

    | Cluster | Größe | Ø CO₂ | Profil |
    |---------|-------|--------|--------|
    | **0** | ~2.400 (42%) | ~148 g/km | Leichte Mittelklasse — niedrige Masse & Leistung, unter Flottenø |
    | **1** | ~1.430 (25%) | ~210 g/km | Schwere Nutzfahrzeuge — hohe Masse, überwiegend Diesel, stark über Flottenø |
    | **2** | ~1.130 (20%) | ~126 g/km | **Effizienzcluster** — niedrigste CO₂-Werte, leichte Benziner |
    | **3** | ~740 (13%)   | ~243 g/km | Hochleistungsfahrzeuge — höchste Leistung & Masse, weiteste Streuung |

    **Flottenø: 171.3 g/km** — Cluster 0 und 2 liegen deutlich darunter, Cluster 1 und 3 darüber.

    **Antwort auf die Forschungsfrage:** Ja, es lassen sich natürliche Fahrzeugsegmente identifizieren.
    Der stärkste Treiber der Clusterzugehörigkeit ist die Kombination aus **Masse und Leistung** —
    sichtbar im Power-vs-Mass-Scatterplot. Kraftstoffart und Karosserie differenzieren zusätzlich:
    Cluster 2 (Effizienz) ist stark Benzin-dominiert, Cluster 1 (Nutzfahrzeuge) fast ausschließlich Diesel.
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
        "Interpretation: Cluster 2 ist fast ausschließlich Benzin (ES) — der Effizienzcluster "
        "besteht hauptsächlich aus leichten Benzin-Limousinen mit Schaltgetriebe (M 5/M 6). "
        "Cluster 1 ist nahezu vollständig Diesel (GO) — schwere Minibuse und Transporter "
        "dominieren dieses Segment. Cluster 3 zeigt die breiteste Karosserie-Vielfalt "
        "(Berline, Break, TS TERRAINS/CHEMINS) — typisch für Hochleistungsfahrzeuge quer durch alle Klassen."
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
        "Radar & Heatmap zeigen normalisierte Werte (0=Minimum, 1=Maximum des Datensatzes). "
        "Cluster 3 hat die höchsten normalisierten Werte in allen drei Dimensionen (Power=0.50, Mass=0.47, CO₂=0.33) "
        "— das ist das Hochleistungssegment. "
        "Cluster 2 hat die niedrigsten Werte (Power=0.07, Mass=0.18, CO₂=0.09) "
        "— das ist der Effizienzcluster. "
        "Cluster 1 fällt durch einen ungewöhnlich hohen Mass-Wert (0.67) bei moderater Leistung (0.13) auf "
        "— typisch für schwere Nutzfahrzeuge mit Dieselmotor."
    )


# ═══════════════════════ TAB 5 – PREDICTION ══════════════════════════════════
with tabs[5]:
    st.header("🤖 Predictive Modeling")

    st.markdown("""
    > **Forschungsfrage:** Welchen relativen Beitrag leisten Fahrzeugmasse, Motorleistung,
    > Kraftstoffart, Karosserie und Getriebetyp zur Erklärung von CO₂-Emissionen,
    > und welches minimale Feature-Set erreicht die beste Vorhersagegüte?
    """)

    with st.spinner("Modelle werden trainiert (Feature Sets + CV + 5 Modelle) …"):
        try:
            (fitted, results_df, fs_df, best_fs, feature_cols,
             X_train, X_test, y_train, y_test,
             rf_pipe, fi_df, num_f, cat_f) = train_all_models(df_unique)
        except Exception as e:
            st.error(f"Training fehlgeschlagen: {e}")
            st.stop()

    # best_model_name available for the rest of this tab
    best_model_name = results_df.iloc[0]["Model"]

    # ── 1. Feature Set Comparison ────────────────────────────────────────────
    st.subheader("1️⃣ Feature Set Comparison (5-Fold CV, Random Forest)")
    st.markdown(
        "Vier Feature-Kombinationen werden per **5-facher Kreuzvalidierung** mit einem "
        "Random Forest verglichen. Der MAE (Mean Absolute Error) misst die "
        "durchschnittliche Abweichung in g/km — **niedriger ist besser**. "
        "So wird das informativste Feature-Set ohne Overfitting-Risiko ausgewählt."
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_fs = fs_df.sort_values("CV_MAE_mean", ascending=True)
    ax.barh(plot_fs["Feature_Set"], plot_fs["CV_MAE_mean"], color=BLUE, alpha=0.9)
    ax.set_xlabel("CV MAE (lower = better)"); ax.set_ylabel("Feature Set")
    ax.set_title("Feature Set Comparison (5-Fold CV, Random Forest)")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.success(f"✅ Bestes Feature-Set: **{best_fs}** | Features: {', '.join(feature_cols)}")
    st.dataframe(fs_df[["Feature_Set","Features","CV_MAE_mean","CV_MAE_std"]]
                 .style.format({"CV_MAE_mean": "{:.2f}", "CV_MAE_std": "{:.2f}"}),
                 width='stretch')
    st.caption(
        "Interpretation: `all_features` (Masse + Leistung + Kraftstoff + Getriebe + Karosserie) "
        "erzielt den niedrigsten MAE — jedes Feature trägt zur Vorhersagegüte bei. "
        "Das Weglassen der Karosserie (`no_body`) kostet ~0.6 g/km, "
        "ohne Getriebe (`mass_power_fuel`) bereits ~3 g/km mehr Fehler."
    )

    st.markdown("---")

    # ── 2. Model Performance ─────────────────────────────────────────────────
    st.subheader(f"2️⃣ Modellvergleich: R² und MAE ({best_fs})")
    st.markdown(
        "Fünf Modelle werden auf demselben Train/Test-Split (80/20) verglichen. "
        "**R²** misst den Anteil erklärter Varianz (1.0 = perfekt). "
        "**MAE** ist die durchschnittliche Abweichung in g/km. "
        "Ein großer Gap zwischen Train- und Test-Metriken deutet auf **Overfitting** hin."
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
        "Interpretation: Gradient Boosting und Random Forest erreichen R²≈0.95 bei ~7–8 g/km MAE — "
        "d.h. das Modell erklärt 95% der CO₂-Varianz mit einem durchschnittlichen Fehler von nur 7 g/km. "
        "Lineare Modelle (Ridge, Lasso, Linear Regression) plateauieren bei R²≈0.86, "
        "da sie nichtlineare Beziehungen (z.B. Masse × Leistung) nicht erfassen können."
    )

    st.markdown("---")

    # ── 3. Feature Importance ────────────────────────────────────────────────
    st.subheader(f"3️⃣ Feature Importance – Random Forest ({best_fs})")
    st.markdown(
        "Feature Importance (Mean Decrease Impurity) misst, wie stark jedes Merkmal "
        "zur Reduktion des Vorhersagefehlers beiträgt. "
        "Kategorische Features wurden per One-Hot-Encoding aufgespalten "
        "(z.B. `cat__Fuel_GO`, `cat__Body_MINIBUS`). "
        "Numerische Features tragen direkt bei (`num__` Präfix)."
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
        "Interpretation: **Leergewicht (46.9%)** und **Motorleistung (37.2%)** dominieren gemeinsam ~84% "
        "der erklärten Varianz — das sind die primären physikalischen Treiber des CO₂-Ausstoßes. "
        "Ganganzahl (4.6%) und Kraftstoffart (je ~2.6%) liefern zusätzliche Information. "
        "Karosserie und Getriebetyp spielen eine untergeordnete Rolle (<1% je Feature)."
    )

    st.markdown("---")

    # ── 4. Partial Dependence Plots ──────────────────────────────────────────
    st.subheader("4️⃣ Partial Dependence Plots – Random Forest")
    st.markdown(
        "PDPs zeigen den **marginalen Effekt** eines einzelnen Features auf den "
        "vorhergesagten CO₂-Wert — alle anderen Features werden dabei auf ihren "
        "Durchschnitt fixiert (ähnlich wie Ceteris-Paribus). "
        "So lässt sich der isolierte, nichtlineare Einfluss jedes Merkmals ablesen."
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
            "Interpretation: Der CO₂-Anstieg mit Masse und Leistung ist nichtlinear — "
            "bei niedrigen Werten ist der Effekt stärker als bei hohen (abnehmende Grenzwirkung). "
            "Mehr Gänge korrelieren leicht negativ mit CO₂ (effizientere Getriebeabstufung). "
            "Automatik zeigt marginal höheren CO₂ als Manual — nach Kontrolle aller anderen Features."
        )
    except Exception as e:
        st.warning(f"PDP nicht verfügbar: {e}")



# ═══════════════════════ TAB 6 – CO₂-RECHNER ═════════════════════════════════
with tabs[6]:
    st.header("🎯 CO₂-Rechner für Kaufentscheidungen")
    st.markdown(
        "Wähle dein Wunschfahrzeug nach **Alltagsmerkmalen** — "
        "die App zeigt den **echten CO₂-Median** aus vergleichbaren Fahrzeugen "
        "im ADEME-Datensatz und welche Marke in deinem Segment am sparsamsten ist."
    )

    # ── Mappings: Konsumentensprache → Datensatz-Werte ────────────────────────
    SEGMENT_MAP = {
        "Kleinwagen":          {"body": ["BERLINE"],                              "mass": (900,  1300), "kw": (40,  75)},
        "Kompaktklasse":       {"body": ["BERLINE", "COMBISPACE"],                "mass": (1100, 1500), "kw": (56,  110)},
        "Mittelklasse":        {"body": ["BERLINE", "BREAK"],                     "mass": (1300, 1700), "kw": (90,  160)},
        "Kombi":               {"body": ["BREAK"],                                "mass": (1300, 1900), "kw": (80,  160)},
        "SUV / Geländewagen":  {"body": ["TS TERRAINS/CHEMINS", "BREAK"],         "mass": (1600, 2500), "kw": (100, 220)},
        "Van / Großraumvan":   {"body": ["MINIBUS", "MONOSPACE", "COMBISPACE"],   "mass": (1500, 2500), "kw": (80,  180)},
        "Cabrio / Coupe":      {"body": ["CABRIOLET", "COUPE"],                   "mass": (1100, 1800), "kw": (100, 300)},
    }

    MOTOR_MAP = {
        "Schwach  (bis 75 PS / ~55 kW)":      (40,  56),
        "Mittel   (76–130 PS / 56–96 kW)":    (56,  96),
        "Stark    (131–200 PS / 97–147 kW)":  (96,  147),
        "Sehr stark (über 200 PS / >147 kW)": (147, 600),
    }

    FUEL_MAP = {"Benzin": "ES", "Diesel": "GO"}
    GEAR_MAP = {"Schaltgetriebe": "M", "Automatik": "A"}

    # ── Formular ──────────────────────────────────────────────────────────────
    with st.form("co2_consumer_form"):
        st.subheader("Dein Wunschfahrzeug")
        col1, col2 = st.columns(2)
        with col1:
            segment      = st.selectbox("Fahrzeugklasse", list(SEGMENT_MAP.keys()), index=1)
            antrieb      = st.radio("Antrieb", ["Benzin", "Diesel"], horizontal=True)
            getriebe     = st.radio("Getriebe", ["Schaltgetriebe", "Automatik"], horizontal=True)
        with col2:
            motorisierung = st.select_slider(
                "Motorisierung",
                options=list(MOTOR_MAP.keys()),
                value="Mittel   (76–130 PS / 56–96 kW)"
            )
            st.info(
                "Basiert auf echten Fahrzeugdaten aus dem ADEME-Datensatz "
                "(Frankreich 2013, 5.700 unique Konfigurationen nach Deduplication)."
            )
        submitted = st.form_submit_button(
            "CO₂ berechnen & Marken vergleichen 🚀", use_container_width=True
        )

    if submitted:
        seg_cfg     = SEGMENT_MAP[segment]
        kw_range    = MOTOR_MAP[motorisierung]
        fuel_val    = FUEL_MAP[antrieb]
        gear_prefix = GEAR_MAP[getriebe]
        ps_lo       = round(kw_range[0] * 1.36)
        ps_hi       = round(kw_range[1] * 1.36)

        # ── Filter: exakt wie Markenvergleich ────────────────────────────────
        mask = (
            df_unique["Fuel"].eq(fuel_val) &
            df_unique["Body"].isin(seg_cfg["body"]) &
            df_unique["Maximum Power (kW)"].between(kw_range[0], kw_range[1]) &
            df_unique["Gearbox"].astype(str).str.startswith(gear_prefix)
        )
        df_match = df_unique[mask].copy()

        # Fallback 1: ohne Getriebe
        used_gear_filter = True
        if len(df_match) < 5:
            mask2 = (
                df_unique["Fuel"].eq(fuel_val) &
                df_unique["Body"].isin(seg_cfg["body"]) &
                df_unique["Maximum Power (kW)"].between(kw_range[0], kw_range[1])
            )
            df_match = df_unique[mask2].copy()
            used_gear_filter = False

        # Fallback 2: ohne Motorisierungsfilter
        used_power_filter = True
        if len(df_match) < 3:
            mask3 = (
                df_unique["Fuel"].eq(fuel_val) &
                df_unique["Body"].isin(seg_cfg["body"])
            )
            df_match = df_unique[mask3].copy()
            used_gear_filter = False
            used_power_filter = False

        co2_vals = df_match["CO2 (g/km)"].dropna()

        if len(co2_vals) == 0:
            st.error("Keine Fahrzeuge gefunden. Bitte andere Konfiguration wählen.")
            st.stop()

        # ── Kennzahlen aus echten Daten ──────────────────────────────────────
        co2_median = co2_vals.median()
        co2_mean   = co2_vals.mean()
        co2_p25    = co2_vals.quantile(0.25)
        co2_p75    = co2_vals.quantile(0.75)
        co2_min    = co2_vals.min()
        co2_max    = co2_vals.max()
        n_match    = len(df_match)
        n_brands   = df_match["Brand"].nunique()

        # Gesamtflotten-Median als Referenz
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
        st.subheader("Ergebnis")

        # Filterhinweis
        filter_info = f"{segment} · {antrieb} · "
        if used_gear_filter:
            filter_info += f"{getriebe} · "
        if used_power_filter:
            filter_info += f"{ps_lo}–{ps_hi} PS"
        else:
            filter_info += "alle PS-Stufen"
        if not used_gear_filter and len(df_match) > 0:
            st.caption(f"Hinweis: Getriebe-Filter wurde erweitert (zu wenig Treffer).")
        if not used_power_filter:
            st.caption(f"Hinweis: PS-Filter wurde erweitert (zu wenig Treffer).")

        # ── Ergebnis-Karte ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='background:{color}22;border-left:6px solid {color};"
            f"padding:20px;border-radius:8px;margin:8px 0;'>"
            f"<h2 style='color:{color};margin:0'>"
            f"🚗 {co2_median:.0f} g CO₂/km <span style='font-size:16px;font-weight:normal'>"
            f"(Median aus {n_match} echten Fahrzeugen)</span></h2>"
            f"<p style='font-size:16px;margin:6px 0'>"
            f"EU-Effizienzklasse: <strong>{euro}</strong>"
            f"&nbsp;·&nbsp; Jahres-CO₂: ca. <strong>{jahres_co2:.0f} kg</strong> "
            f"(bei 15.000 km/Jahr)</p>"
            f"<p style='color:gray;font-size:12px;margin:0'>"
            f"Daten: {filter_info} · {n_brands} Marken · "
            f"Spanne: {co2_min:.0f}–{co2_max:.0f} g/km"
            f"</p></div>",
            unsafe_allow_html=True
        )

        # ── 4 Kennzahlen ─────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median CO₂", f"{co2_median:.0f} g/km")
        c2.metric("Spanne (25–75%)", f"{co2_p25:.0f}–{co2_p75:.0f} g/km")
        c3.metric("vs. Gesamtflotte", f"{delta_fleet:+.0f} g/km",
                  delta_color="inverse")
        c4.metric("Besser als", f"{pct_better:.0f}% aller Fahrzeuge")

        # ── Verteilung im Segment ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(df_unique["CO2 (g/km)"].dropna(), bins=60,
                color="lightgrey", alpha=0.7, label="Gesamtflotte")
        ax.hist(co2_vals, bins=30, color=color, alpha=0.75,
                label=f"{segment} · {antrieb} ({n_match} Fzg.)")
        ax.axvline(co2_median, color=color, lw=2.5, linestyle="--",
                   label=f"Segment-Median: {co2_median:.0f} g/km")
        ax.axvline(fleet_median, color="gray", lw=1.5, linestyle=":",
                   label=f"Flotten-Median: {fleet_median:.0f} g/km")
        ax.set_xlabel("CO₂ (g/km)"); ax.set_ylabel("Häufigkeit")
        ax.set_title("Dein Segment vs. Gesamtflotte")
        ax.legend(fontsize=9)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Markenvergleich ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏷️ Welche Marke ist in diesem Segment am sparsamsten?")
        st.caption(f"{n_match} Fahrzeuge von {n_brands} Marken · {filter_info}")

        brand_summary = (
            df_match.groupby("Brand")["CO2 (g/km)"]
            .agg(Modelle="count", CO2_Min="min",
                 CO2_Median="median", CO2_Mittel="mean", CO2_Max="max")
            .round(1)
            .sort_values("CO2_Median")
            .reset_index()
        )
        # Nur Marken mit ≥2 Modellen (außer wenn zu wenige)
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
                   label=f"Segment-Median: {co2_median:.0f} g/km")

        ax.set_xlabel("Median CO₂ (g/km)")
        ax.set_title(
            f"Markenvergleich: {segment} · {antrieb} · {getriebe} · {ps_lo}–{ps_hi} PS",
            fontsize=12, pad=12
        )
        ax.set_xlim(0, plot_b["CO2_Median"].max() * 1.28)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#2ecc71", label="Top 3 sparsamste Marken"),
            Patch(facecolor=BLUE,      label="Mittelfeld"),
            Patch(facecolor="#e74c3c", label="Top 3 höchster CO₂"),
            plt.Line2D([0],[0], color="black", lw=1.5, linestyle="--",
                       label=f"Segment-Median: {co2_median:.0f} g/km"),
        ]
        ax.legend(handles=legend_els, loc="lower right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Top 3 Empfehlungskarten ──────────────────────────────────────────
        st.subheader("🏆 Top 3 Empfehlungen")
        top3  = brand_summary.head(3)
        cols3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for idx, (col_ui, (_, rb)) in enumerate(zip(cols3, top3.iterrows())):
            saving    = co2_median - rb["CO2_Median"]
            saving_kg = saving * 15000 / 1000
            saving_str = (f"↓ {saving_kg:.0f} kg CO₂/Jahr gespart"
                          if saving > 1 else "Segment-Median")
            col_ui.markdown(
                f"<div style='background:#f0fdf4;border:2px solid #2ecc71;"
                f"padding:16px;border-radius:10px;text-align:center;'>"
                f"<div style='font-size:28px'>{medals[idx]}</div>"
                f"<div style='font-size:17px;font-weight:bold'>{rb['Brand']}</div>"
                f"<div style='font-size:24px;color:#2ecc71;font-weight:bold'>"
                f"{rb['CO2_Median']:.0f} g/km</div>"
                f"<div style='font-size:11px;color:gray;margin-top:4px'>"
                f"Min {rb['CO2_Min']:.0f} · Max {rb['CO2_Max']:.0f} g/km<br>"
                f"{int(rb['Modelle'])} Modelle im Segment<br>"
                f"<strong>{saving_str}</strong>"
                f"</div></div>",
                unsafe_allow_html=True
            )

        # ── Detailtabelle ────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋 Alle gefundenen Fahrzeuge anzeigen"):
            show_cols = [c for c in
                         ["Brand", "Folder Model", "Fuel", "Body", "Gearbox",
                          "Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
                          "CO2 (g/km)", "Combined Consumption (l/100km)"]
                         if c in df_match.columns]
            disp = df_match[show_cols].copy()
            if "Maximum Power (kW)" in disp.columns:
                disp.insert(disp.columns.get_loc("Maximum Power (kW)")+1,
                            "PS", (disp["Maximum Power (kW)"] * 1.36).round(0).astype("Int64"))
            st.dataframe(
                disp.sort_values("CO2 (g/km)").reset_index(drop=True),
                width="stretch"
            )
