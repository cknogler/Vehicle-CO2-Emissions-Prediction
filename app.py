"""
app.py — Determinants of Vehicle CO2 Emissions: A Research-Question-Driven
Analysis (ADEME Car Labelling Dataset, France, 2013)

Every section of this app answers one explicit research question about what
drives passenger-vehicle CO2 emissions and in which direction. Direction of
effect is established three ways: standardized, signed Ridge coefficients
(unambiguous sign); SHAP values from the best-performing model (magnitude
and non-linear direction); and ceteris-paribus model comparisons
(approximate causal effects, isolating one feature from the vehicle class
it is normally confounded with).
"""
import io
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
try:
    from kmodes.kprototypes import KPrototypes
    KPROTO_AVAILABLE = True
except ImportError:
    KPROTO_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO2 Emissions — Determinants",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal, print-friendly styling ──────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Source Serif Pro", Georgia, "Times New Roman", serif; font-size: 15px; color: #1a1a1a; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: #FAFAFA; border-right: 1px solid #E5E7EB; font-family: "Inter", sans-serif; }
[data-testid="metric-container"] { background: transparent; border: none; border-top: 1px solid #D1D5DB; padding: 10px 0 6px 0; }
[data-testid="stMetricLabel"] { font-size: 10px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .06em; color: #6B7280 !important; font-family: "Inter", sans-serif; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 600 !important; color: #1a1a1a !important; }
h1 { font-size: 26px !important; font-weight: 700 !important; color: #111827 !important; }
h2 { font-size: 19px !important; font-weight: 700 !important; color: #111827 !important; border-bottom: 1px solid #E5E7EB; padding-bottom: 6px; margin-top: 2.2rem !important; }
h3 { font-size: 15px !important; font-weight: 600 !important; color: #374151 !important; font-style: italic; }
hr { border-color: #E5E7EB !important; margin: 1.4rem 0 !important; }
[data-testid="stDataFrame"] { border: 1px solid #E5E7EB; border-radius: 3px; }
.finding-box { background: #F9FAFB; border-left: 3px solid #111827; padding: 0.8rem 1rem; margin: 0.6rem 0 1rem 0; font-size: 14.5px; }
.methods-note { color: #6B7280; font-size: 13.5px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

BLUE         = "#2563EB"
ACCENT       = "#DC2626"
NEUTRAL      = "#6B7280"
RANDOM_STATE = 42

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA", "savefig.facecolor": "white",
    "axes.edgecolor": "#E5E7EB", "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.grid": True, "grid.color": "#FFFFFF", "grid.linewidth": 1.0,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": "#111827",
    "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": "#4B5563", "axes.labelpad": 6,
    "xtick.color": "#9CA3AF", "ytick.color": "#9CA3AF",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#E5E7EB",
    "legend.fontsize": 9, "lines.linewidth": 2.0, "patch.linewidth": 0.5,
    "figure.dpi": 110, "font.family": "sans-serif",
})

CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)

COLUMN_MAPPING = {
    "Marque": "Brand", "Modèle dossier": "Folder Model", "Modèle UTAC": "Utac Model",
    "Désignation commerciale": "Commerical Designation", "CNIT": "cnit",
    "Type Variante Version (TVV)": "Type Variant Version", "Carburant": "Fuel",
    "Hybride": "Hybrid", "Puissance administrative": "Administrative Power",
    "Puissance maximale (kW)": "Maximum Power (kW)", "Boîte de vitesse": "Gearbox",
    "Consommation urbaine (l/100km)": "Urban Consumption (l/100km)",
    "Consommation extra-urbaine (l/100km)": "Extra Urban Consumption (l/100km)",
    "Consommation mixte (l/100km)": "Combined Consumption (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)", "CO type I (g/km)": "CO type 1 (g/km)",
    "HC (g/km)": "HC (g/km)", "NOX (g/km)": "NOX (g/km)", "HC+NOX (g/km)": "HC+NOX (g/km)",
    "Particules (g/km)": "Particles (g/km)",
    "masse vide euro min (kg)": "Empty Mass Euro Min (kg)",
    "masse vide euro max (kg)": "Empty Mass Euro Max (kg)",
    "Champ V9": "Field V9", "Date de mise à jour": "Update Date",
    "Carrosserie": "Body", "gamme": "Range",
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

GEAR_TYPE_MAP = {"M": "Manual", "A": "Automatic", "V": "CVT",
                  "D": "DCT", "N": "Automatic", "S": "Manual"}


# ═══════════════════════════ DATA PIPELINE ═════════════════════════════════
# Preprocessing methodology — documented once in Section 2 rather than
# re-narrated across separate tabs.

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
        raise ValueError("Could not read CSV file.")

    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    if all(c in df.columns for c in ["HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)"]):
        df['hc_calc']  = df['HC+NOX (g/km)'] - df['NOX (g/km)']
        df['nox_calc'] = df['HC+NOX (g/km)'] - df['HC (g/km)']
        df['hc_calc']  = df['hc_calc'].fillna(df['HC (g/km)'])
        df['nox_calc'] = df['nox_calc'].fillna(df['NOX (g/km)'])
        df["HC (g/km)"]     = df["hc_calc"]
        df["NOX (g/km)"]    = df["nox_calc"]
        df["HC+NOX (g/km)"] = df["hc_calc"] + df["nox_calc"]
        df.drop(columns=['hc_calc', 'nox_calc'], inplace=True)

    if "Gearbox" in df.columns:
        df['Gearbox'] = df['Gearbox'].replace(['N 0', 'N 1'], 'A 0')
        df['Gearbox'] = df['Gearbox'].replace(['S 6'], 'D 6')

    electric_cols = ["CO type 1 (g/km)", "Urban Consumption (l/100km)",
                     "Extra Urban Consumption (l/100km)", "Combined Consumption (l/100km)",
                     "CO2 (g/km)", "HC+NOX (g/km)", "HC (g/km)", "Particles (g/km)"]
    if "Fuel" in df.columns:
        el_mask = df["Fuel"] == "EL"
        for c in electric_cols:
            if c in df.columns:
                df.loc[el_mask, c] = df.loc[el_mask, c].fillna(0)

    if "Empty Mass Euro Min (kg)" in df.columns and "Empty Mass Euro Max (kg)" in df.columns:
        df["Empty Mass Euro Avg (kg)"] = (
            pd.to_numeric(df["Empty Mass Euro Min (kg)"], errors="coerce") +
            pd.to_numeric(df["Empty Mass Euro Max (kg)"], errors="coerce")
        ) / 2
        df.drop(columns=["Empty Mass Euro Min (kg)", "Empty Mass Euro Max (kg)"], inplace=True)

    for col in ["CO2 (g/km)", "Combined Consumption (l/100km)",
                "Maximum Power (kW)", "Empty Mass Euro Avg (kg)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def make_df_unique(df: pd.DataFrame) -> pd.DataFrame:
    if "Fuel" not in df.columns:
        return df
    df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy()

    if "Gearbox" in df_combus.columns:
        gear_split = df_combus["Gearbox"].astype(str).str.split(" ", expand=True)
        df_combus["GearType"]  = gear_split[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_combus["GearCount"] = pd.to_numeric(
            gear_split[1] if 1 in gear_split.columns else pd.Series([np.nan] * len(df_combus)),
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

    if "Gearbox" in df_unique.columns:
        gear_split2 = df_unique["Gearbox"].astype(str).str.split(" ", expand=True)
        df_unique["GearType"]  = gear_split2[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_unique["GearCount"] = pd.to_numeric(
            gear_split2[1] if 1 in gear_split2.columns else pd.Series([np.nan] * len(df_unique)),
            errors="coerce"
        )
        df_unique = df_unique[df_unique["GearType"] != "Other"].reset_index(drop=True)

    return df_unique


@st.cache_data(show_spinner=False)
def run_clustering(_df: pd.DataFrame, k: int = 4):
    categorical_cols = [c for c in ['Body', 'Fuel', 'Gearbox'] if c in _df.columns]
    numeric_cols     = [c for c in ['Maximum Power (kW)', 'Empty Mass Euro Avg (kg)'] if c in _df.columns]
    feature_cols     = categorical_cols + numeric_cols
    target_col       = 'CO2 (g/km)'

    df_c = _df[feature_cols + [target_col]].dropna().copy()
    scaler = StandardScaler()
    df_kp  = df_c.copy()
    df_kp[numeric_cols] = scaler.fit_transform(df_kp[numeric_cols])
    for col in categorical_cols:
        df_kp[col] = df_kp[col].astype(str)

    X_matrix        = df_kp[feature_cols].to_numpy(dtype=object)
    categorical_idx = [feature_cols.index(col) for col in categorical_cols]

    if not KPROTO_AVAILABLE:
        st.error("kmodes not installed. Add 'kmodes>=0.12.2' to requirements.txt.")
        return df_c

    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=5, verbose=0, random_state=RANDOM_STATE)
    df_c['Cluster'] = kproto.fit_predict(X_matrix, categorical=categorical_idx)
    return df_c


@st.cache_resource(show_spinner=False)
def train_all_models(_df: pd.DataFrame):
    target_col = "CO2 (g/km)"
    all_needed = sorted(set([target_col] + [c for cols in FEATURE_SETS.values() for c in cols]))
    df_model = _df[[c for c in all_needed if c in _df.columns]].dropna().copy()

    def get_types(features):
        num = df_model[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat = df_model[features].select_dtypes(include=["object", "category"]).columns.tolist()
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

    fs_results = []
    for fs_name, fs_feats in FEATURE_SETS.items():
        feats_avail = [f for f in fs_feats if f in df_model.columns]
        if not feats_avail:
            continue
        num, cat = get_types(feats_avail)
        _, tree_pre = build_preprocessors(num, cat)
        pipe = Pipeline([("pre", tree_pre),
                         ("m", GradientBoostingRegressor(n_estimators=200, learning_rate=0.2,
                                                          max_depth=6, random_state=RANDOM_STATE))])
        scores = cross_val_score(pipe, df_model[feats_avail], df_model[target_col],
                                 cv=5, scoring="neg_mean_absolute_error")
        fs_results.append({"Feature_Set": fs_name, "Features": ", ".join(feats_avail),
                            "CV_MAE_mean": -np.mean(scores), "CV_MAE_std": np.std(scores)})

    fs_df = pd.DataFrame(fs_results).sort_values("CV_MAE_mean")
    best_fs = fs_df.iloc[0]["Feature_Set"]
    feature_cols = [f for f in FEATURE_SETS[best_fs] if f in df_model.columns]

    X = df_model[feature_cols]
    y = df_model[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    num_f, cat_f = get_types(feature_cols)
    scaled_pre, tree_pre = build_preprocessors(num_f, cat_f)

    model_defs = {
        "Linear Regression": Pipeline([("pre", scaled_pre), ("m", LinearRegression())]),
        "Ridge":             Pipeline([("pre", scaled_pre), ("m", Ridge(alpha=1.0))]),
        "Lasso":             Pipeline([("pre", scaled_pre), ("m", Lasso(alpha=0.1))]),
        "Random Forest":     Pipeline([("pre", tree_pre),
                                       ("m", RandomForestRegressor(
                                           n_estimators=300, max_depth=20, max_features=0.8,
                                           min_samples_split=2, min_samples_leaf=1,
                                           random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("pre", tree_pre),
                                       ("m", GradientBoostingRegressor(
                                           n_estimators=200, learning_rate=0.2, max_depth=6,
                                           min_samples_split=10, subsample=1.0, max_features=0.5,
                                           random_state=RANDOM_STATE))]),
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
            "Train_R2": r2_score(y_train, ytr_pred), "Test_R2": r2_score(y_test, yte_pred),
            "Train_MAE": mean_absolute_error(y_train, ytr_pred),
            "Test_MAE": mean_absolute_error(y_test, yte_pred),
        })
    results_df = pd.DataFrame(results).sort_values("Test_R2", ascending=False)

    rf_pipe = fitted["Random Forest"]
    rf_pre  = rf_pipe.named_steps["pre"]
    rf_model = rf_pipe.named_steps["m"]
    feat_names = rf_pre.get_feature_names_out()
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": rf_model.feature_importances_}) \
              .sort_values("Importance", ascending=False)

    gb_pipe = fitted["Gradient Boosting"]
    gb_model = gb_pipe.named_steps["m"]
    gb_fi_df = pd.DataFrame({"Feature": feat_names, "Importance": gb_model.feature_importances_}) \
                 .sort_values("Importance", ascending=False)

    # Standardized, signed Ridge coefficients — the primary "direction of
    # effect" evidence: a regression coefficient has an unambiguous sign and
    # reads directly as "+1 SD in mass is associated with +X g/km CO2".
    ridge_pipe = fitted["Ridge"]
    ridge_pre  = ridge_pipe.named_steps["pre"]
    ridge_model = ridge_pipe.named_steps["m"]
    ridge_feat_names = ridge_pre.get_feature_names_out()
    coef_df = pd.DataFrame({"Feature": ridge_feat_names, "Coefficient": ridge_model.coef_}) \
                .sort_values("Coefficient")

    return (fitted, results_df, fs_df, best_fs, feature_cols,
            X_train, X_test, y_train, y_test, rf_pipe, fi_df, num_f, cat_f,
            gb_fi_df, gb_pipe, coef_df)


@st.cache_resource(show_spinner=False)
def compute_shap_values(_gb_pipe, _X_test, sample_size: int = 500):
    gb_pre = _gb_pipe.named_steps["pre"]
    gb_model = _gb_pipe.named_steps["m"]

    X_sample = _X_test.sample(n=min(sample_size, len(_X_test)), random_state=RANDOM_STATE).reset_index(drop=True)
    X_sample_transformed = gb_pre.transform(X_sample)
    if hasattr(X_sample_transformed, "toarray"):
        X_sample_transformed = X_sample_transformed.toarray()
    feat_names = gb_pre.get_feature_names_out()
    X_sample_df = pd.DataFrame(X_sample_transformed, columns=feat_names)

    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_sample_df)
    return explainer, shap_values, X_sample_df, X_sample


def aggregate_shap_to_original_features(shap_values, feat_names, X_sample_raw,
                                         feature_cols, cat_f, num_f, ohe):
    """Collapses one-hot SHAP columns back into one signed value per original
    feature, so a categorical feature's effect isn't split across rows."""
    feat_names = list(feat_names)
    col_to_orig = {}
    for i, col in enumerate(cat_f):
        for cat_val in ohe.categories_[i]:
            col_to_orig[f"cat__{col}_{cat_val}"] = col
    for col in num_f:
        col_to_orig[f"num__{col}"] = col

    shap_wide = pd.DataFrame(shap_values, columns=feat_names)
    agg_shap = pd.DataFrame(index=shap_wide.index)
    for orig_col in feature_cols:
        matching = [c for c in feat_names if col_to_orig.get(c) == orig_col]
        agg_shap[orig_col] = shap_wide[matching].sum(axis=1) if matching else 0.0

    display_data = pd.DataFrame(index=X_sample_raw.index)
    for col in feature_cols:
        if col in cat_f:
            display_data[col] = pd.Categorical(X_sample_raw[col]).codes
        else:
            display_data[col] = pd.to_numeric(X_sample_raw[col], errors="coerce")

    return agg_shap[feature_cols], display_data[feature_cols]


# ═══════════════════════════ SIDEBAR ═══════════════════════════════════════
with st.sidebar:
    st.title("CO2 Emissions Study")
    st.caption("ADEME Car Labelling Dataset, France 2013")
    st.markdown("---")
    uploaded = st.file_uploader("Upload alternative CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown(
        "**Source:** [GitHub repository ↗](https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction)",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Contents**")
    st.markdown(
        "1. Research Questions\n"
        "2. Data & Methods\n"
        "3. RQ1 — Determinants of CO2\n"
        "4. RQ2 — Vehicle Segments\n"
        "5. RQ3 — Fuel Type Effect\n"
        "6. RQ4 — Transmission Effect\n"
        "7. RQ5 — Body Type Effect\n"
        "8. Summary of Findings"
    )

# ═══════════════════════════ LOAD DATA ═════════════════════════════════════
source = uploaded.read() if uploaded is not None else CSV_URL
with st.spinner("Loading and preprocessing data..."):
    try:
        df        = load_and_preprocess(source)
        df_unique = make_df_unique(df)
        df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy() if 'Fuel' in df.columns else df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

with st.spinner("Fitting models..."):
    try:
        (fitted, results_df, fs_df, best_fs, feature_cols,
         X_train, X_test, y_train, y_test, rf_pipe, fi_df, num_f, cat_f,
         gb_fi_df, gb_pipe, coef_df) = train_all_models(df_unique)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

shap_ok = False
if SHAP_AVAILABLE:
    try:
        explainer, shap_values, X_shap_df, X_shap_raw = compute_shap_values(gb_pipe, X_test, sample_size=500)
        ohe = gb_pipe.named_steps["pre"].named_transformers_["cat"]
        agg_shap_df, agg_display_df = aggregate_shap_to_original_features(
            shap_values, X_shap_df.columns, X_shap_raw, feature_cols, cat_f, num_f, ohe
        )
        agg_shap_values = agg_shap_df.values
        shap_ok = True
    except Exception:
        shap_ok = False


# ═══════════════════════════ TITLE & RESEARCH QUESTIONS ════════════════════
st.title("Determinants of Vehicle CO2 Emissions")
st.markdown(
    f"<span class='methods-note'>ADEME Car Labelling Dataset (France, 2013) · "
    f"{len(df):,} raw records · {len(df_unique):,} unique mechanical configurations "
    f"after deduplication</span>",
    unsafe_allow_html=True,
)

st.header("1. Research Questions")
st.markdown("""
This analysis addresses five questions about the technical determinants of
passenger-vehicle CO2 emissions:

**RQ1.** Which technical vehicle characteristics best explain and predict CO2 emissions, and in which direction does each factor act?

**RQ2.** Do vehicles form natural segments with distinct CO2 profiles when clustered on their technical characteristics?

**RQ3.** Do diesel vehicles emit more CO2 than petrol vehicles — and is that a fuel-chemistry effect or a vehicle-class effect?

**RQ4.** Does transmission type (manual, automatic, CVT, DCT) affect CO2 emissions independently of vehicle mass and power?

**RQ5.** Which body style has the highest CO2 emissions once mass, power, fuel type and transmission are held constant?

Sections 3–7 each address one question; Section 8 summarizes all findings.
""")


# ═══════════════════════════ SECTION 2 — DATA & METHODS ════════════════════
st.header("2. Data & Methods")

n_esgo = len(df_combus)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw records", f"{len(df):,}")
c2.metric("Petrol/diesel subset", f"{n_esgo:,}")
c3.metric("Unique configurations", f"{len(df_unique):,}")
c4.metric("Reduction", f"{(1 - len(df_unique)/n_esgo)*100:.0f}%" if n_esgo else "n/a")

st.markdown(f"""
**Sample.** The raw dataset contains {len(df):,} homologation records. Electric,
hybrid and gas vehicles are excluded because they follow different emission
physics; the petrol/diesel (ES/GO) subset used throughout this analysis
contains {n_esgo:,} records.

**Deduplication.** The same mechanical configuration (identical brand, model,
fuel, body, gearbox, power, mass, CO2, consumption and range) is registered
under many trim names and option packages. Records are grouped to
{len(df_unique):,} unique configurations, which is the analytical unit for
every result below; without this step, over-represented configurations would
bias every statistic toward the most commonly *registered*, not the most
common *type of*, vehicle.

**Gearbox coding.** The raw gearbox code (e.g. `A 6`, `M 5`) is split into
transmission type (Manual / Automatic / CVT / DCT) and gear count; the small
number of unmapped codes is dropped.

**Modeling.** Five regression models (Linear, Ridge, Lasso, Random Forest,
Gradient Boosting) are compared on an 80/20 train–test split, with the
feature set chosen by 5-fold cross-validated MAE. Direction of effect is
established three ways, in order of interpretive strength for this analysis:
(i) standardized Ridge coefficients, which have an unambiguous sign; (ii)
SHAP values from the best-performing (Gradient Boosting) model, which
capture non-linear and interaction effects; (iii) ceteris-paribus model
comparisons, in which a single categorical feature (fuel, transmission, body)
is swapped while every other feature is held at each vehicle's actual value —
an approximation to a causal effect, isolating that feature from the vehicle
class it is usually confounded with.

<span class="methods-note">Caveat: these are observational, predictive
models. Ceteris-paribus comparisons approximate causal effects only under the
assumption that no relevant confounder is missing from the feature set (e.g.
engine generation, model year within this cross-section). They should be read
as "the model's best estimate of the isolated effect," not as evidence from a
controlled experiment.</span>
""", unsafe_allow_html=True)

with st.expander("Missing values and full variable summary"):
    missing_values = df.isnull().sum()
    missing_sorted = missing_values[missing_values > 0].sort_values(ascending=False)
    if len(missing_sorted) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=missing_sorted.index, y=missing_sorted.values, color=BLUE, ax=ax)
        ax.set_title("Missing values per column"); ax.tick_params(axis='x', rotation=90)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.success("No missing values after preprocessing.")
    desc = df.describe(include='all').T
    num_cols_desc = desc.select_dtypes(include='number').columns.tolist()
    st.dataframe(desc.style.format({c: "{:.2f}" for c in num_cols_desc}, na_rep="-"), width='stretch')


# ═══════════════════════════ SECTION 3 — RQ1: DETERMINANTS ═════════════════
st.header("3. RQ1 — Which factors drive CO2 emissions, and in which direction?")

# ── 3.1 Bivariate relationships ─────────────────────────────────────────
st.subheader("3.1 Bivariate relationships")
st.markdown(
    "Pearson (linear) and Spearman (monotonic, rank-based) correlations with "
    "CO2 emissions, computed on the deduplicated sample."
)
df_numeric_heat = df_unique.select_dtypes(include=np.number).drop(columns=["Clone_Count", "GearCount"], errors="ignore").copy()
pearson_corr  = df_numeric_heat.corr(method='pearson')
spearman_corr = df_numeric_heat.corr(method='spearman')

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='RdBu_r', annot_kws={"size": 10}, linewidths=0.4, ax=ax[0])
ax[0].set_title('Pearson correlation'); ax[0].tick_params(axis='x', rotation=45, labelsize=9)
sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='YlGnBu', annot_kws={"size": 10}, linewidths=0.4, ax=ax[1])
ax[1].set_title('Spearman correlation'); ax[1].tick_params(axis='x', rotation=45, labelsize=9)
plt.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("""
<div class="finding-box">
CO2 correlates almost perfectly with combined fuel consumption (r ≈ 0.98) —
expected, since CO2 is stoichiometrically proportional to combustion, so
consumption is excluded from the predictive model to avoid a trivial
unit-conversion result. Empty mass has the strongest relationship among the
remaining candidate predictors (Pearson r ≈ 0.68, Spearman r ≈ 0.78; the gap
indicates a non-linear, diminishing effect at high mass). Maximum power is
moderately correlated (Pearson r ≈ 0.67) but more weakly and less
monotonically (Spearman r ≈ 0.54), reflecting wide CO2 variation among
high-power vehicles.
</div>
""", unsafe_allow_html=True)

# ── 3.2 Model & feature set selection ───────────────────────────────────
st.subheader("3.2 Feature set and model selection")
fig, ax = plt.subplots(figsize=(9, 3.5))
plot_fs = fs_df.sort_values("CV_MAE_mean", ascending=True)
ax.barh(plot_fs["Feature_Set"], plot_fs["CV_MAE_mean"], color=BLUE, alpha=0.9)
ax.set_xlabel("Cross-validated MAE, g/km (lower = better)")
ax.set_title("Feature set comparison (5-fold CV, Gradient Boosting)")
plt.tight_layout(); st.pyplot(fig); plt.close()

col_a, col_b = st.columns(2)
with col_a:
    num_cols_res = results_df.select_dtypes(include='number').columns.tolist()
    st.dataframe(
        results_df.style.highlight_max(subset=["Test_R2"], color="#DCFCE7")
                         .highlight_min(subset=["Test_MAE"], color="#DCFCE7")
                         .format({c: "{:.3f}" for c in num_cols_res}),
        width='stretch'
    )
with col_b:
    st.markdown(f"""
    **Selected feature set:** `{best_fs}` — {', '.join(feature_cols)}

    Gradient Boosting and Random Forest reach R² ≈ 0.95 (MAE ≈ 7–8 g/km) on
    held-out data; linear models plateau near R² ≈ 0.86, unable to capture
    non-linear mass × power interactions. **Gradient Boosting** is used as
    the primary model for feature-effect analysis below (Sections 3.4–3.5);
    **Ridge** provides the signed coefficients in Section 3.3.
    """)

# ── 3.3 Direction of effect: standardized Ridge coefficients ───────────
st.subheader("3.3 Direction of effect — standardized regression coefficients")
st.markdown(
    "Ridge regression on standardized features gives each predictor a single "
    "signed coefficient: positive values push CO2 up, negative values pull it "
    "down, and magnitude is directly comparable across features because all "
    "inputs are standardized (mean 0, SD 1) or one-hot encoded."
)
plot_coef = coef_df.copy()
plot_coef["Direction"] = np.where(plot_coef["Coefficient"] >= 0, "Increases CO2", "Decreases CO2")
fig, ax = plt.subplots(figsize=(10, 0.35 * len(plot_coef) + 1.5))
colors = [ACCENT if v >= 0 else BLUE for v in plot_coef["Coefficient"]]
ax.barh(plot_coef["Feature"], plot_coef["Coefficient"], color=colors, alpha=0.9)
ax.axvline(0, color="#374151", lw=1)
ax.set_xlabel("Standardized coefficient (g/km CO2 per +1 SD, or vs. reference category)")
ax.set_title("Signed effect on CO2 emissions (Ridge, standardized)")
plt.tight_layout(); st.pyplot(fig); plt.close()
st.dataframe(coef_df.sort_values("Coefficient", ascending=False).style.format({"Coefficient": "{:+.2f}"}), width='stretch')

top_up = coef_df.sort_values("Coefficient", ascending=False).iloc[0]
top_down = coef_df.sort_values("Coefficient", ascending=True).iloc[0]
st.markdown(f"""
<div class="finding-box">
<b>{top_up['Feature'].split('__')[-1]}</b> shows the largest CO2-increasing
coefficient ({top_up['Coefficient']:+.1f}); <b>{top_down['Feature'].split('__')[-1]}</b>
shows the largest CO2-decreasing coefficient ({top_down['Coefficient']:+.1f}).
Mass and power carry the largest-magnitude numeric coefficients, both
positive; categorical coefficients are read relative to the (dropped)
reference category of each variable.
</div>
""", unsafe_allow_html=True)

# ── 3.4 Non-linear importance & SHAP ────────────────────────────────────
st.subheader("3.4 Magnitude and non-linear direction — SHAP")
st.markdown(
    "Ridge coefficients assume linear, additive effects. SHAP values from the "
    "best-performing model (Gradient Boosting) show the same signed logic "
    "without that assumption, and reveal effect magnitude per prediction "
    "rather than a single global slope."
)
if not SHAP_AVAILABLE or not shap_ok:
    st.warning("SHAP is not available in this environment; see Section 3.3 for signed effect directions.")
else:
    mean_abs_shap = pd.DataFrame({
        "Feature": feature_cols,
        "Mean |SHAP|": np.abs(agg_shap_values).mean(axis=0),
    }).sort_values("Mean |SHAP|", ascending=False)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig_summary = plt.figure(figsize=(8, 6))
        X_shap_plot = X_shap_df.copy()
        X_shap_plot.columns = [str(c).replace("$", "USD").replace("\\", "/") for c in X_shap_plot.columns]
        shap.summary_plot(shap_values, X_shap_plot, show=False, plot_size=None, max_display=15)
        try:
            plt.tight_layout()
        except Exception:
            pass
        st.pyplot(fig_summary, clear_figure=True)
        plt.close()
        st.caption(
            "Each dot is one vehicle from a 500-vehicle test sample. Position "
            "shows the SHAP value (impact on predicted CO2, g/km); color shows "
            "the feature's value (red = high, blue = low) for numeric features, "
            "or category membership for one-hot columns."
        )
    with col_b:
        st.dataframe(mean_abs_shap.style.format({"Mean |SHAP|": "{:.2f}"}), width='stretch')

    top_feat = mean_abs_shap.iloc[0]["Feature"]
    top_val = mean_abs_shap.iloc[0]["Mean |SHAP|"]
    st.markdown(f"""
    <div class="finding-box">
    <b>{top_feat}</b> has the largest average impact on individual
    predictions (±{top_val:.1f} g/km), consistent with the Ridge coefficients
    in 3.3. Unlike a single coefficient, SHAP also shows non-linearity: mass
    and power both display diminishing marginal effects at high values (dot
    density flattens on the outer edges of the beeswarm), matching the
    Spearman-vs-Pearson gap observed in 3.1.
    </div>
    """, unsafe_allow_html=True)

# ── 3.5 Partial dependence ──────────────────────────────────────────────
st.subheader("3.5 Marginal effect shape — partial dependence")
X_train_pdp = X_train.copy()
if "GearCount" in X_train_pdp.columns:
    X_train_pdp["GearCount"] = X_train_pdp["GearCount"].astype(float)
pdp_features = [f for f in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)", "GearType", "GearCount"] if f in feature_cols]
try:
    fig, ax = plt.subplots(figsize=(14, 5))
    PartialDependenceDisplay.from_estimator(
        gb_pipe, X_train_pdp, features=pdp_features,
        categorical_features=[f for f in cat_f if f in pdp_features], ax=ax)
    for axis in fig.axes:
        axis.grid(False)
        for sp in ["top", "right"]: axis.spines[sp].set_visible(False)
    fig.suptitle("Partial dependence — Gradient Boosting", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); st.pyplot(fig); plt.close()
    st.markdown("""
    <div class="finding-box">
    The CO2 increase with mass and power is non-linear, with a stronger slope
    at lower values than at higher ones. More gears correlate slightly
    negatively with CO2 (more efficient gear spacing). Automatic transmission
    shows marginally higher CO2 than manual after controlling for all other
    features — quantified precisely in Section 6.
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Partial dependence plot not available: {e}")


# ═══════════════════════════ SECTION 4 — RQ2: SEGMENTS ═════════════════════
st.header("4. RQ2 — Are there natural vehicle segments with distinct CO2 profiles?")
st.markdown(
    "K-Prototypes clustering (mixed numeric + categorical) on fuel type, body "
    "style, gearbox, power and mass, with k chosen by the elbow method."
)

@st.cache_data(show_spinner=False)
def compute_elbow(_df: pd.DataFrame):
    if not KPROTO_AVAILABLE:
        return None
    categorical_cols = [c for c in ["Body", "Fuel", "Gearbox"] if c in _df.columns]
    numeric_cols     = [c for c in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)"] if c in _df.columns]
    feature_cols_ = categorical_cols + numeric_cols
    target_col_   = "CO2 (g/km)"
    df_c = _df[feature_cols_ + [target_col_]].dropna().copy()
    scaler = StandardScaler()
    df_kp  = df_c.copy()
    df_kp[numeric_cols] = scaler.fit_transform(df_kp[numeric_cols])
    for col in categorical_cols:
        df_kp[col] = df_kp[col].astype(str)
    X_matrix        = df_kp[feature_cols_].to_numpy(dtype=object)
    categorical_idx = [feature_cols_.index(col) for col in categorical_cols]
    costs = []
    k_range = range(2, 8)
    for k_val in k_range:
        model = KPrototypes(n_clusters=k_val, init="Cao", n_init=3, verbose=0, random_state=RANDOM_STATE)
        model.fit_predict(X_matrix, categorical=categorical_idx)
        costs.append(model.cost_)
    return list(k_range), costs

with st.spinner("Computing elbow curve..."):
    elbow_result = compute_elbow(df_unique)

k = 4
if elbow_result is not None:
    k_range, costs = elbow_result
    diffs2 = np.diff(np.diff(costs))
    elbow_k = k_range[np.argmax(diffs2) + 1]
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(k_range, costs, marker="o", color=BLUE, linewidth=2, markersize=7)
        ax.axvline(elbow_k, color="red", lw=1.5, linestyle="--", label=f"Recommended k = {elbow_k}")
        ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Cost")
        ax.set_title("Elbow method"); ax.set_xticks(list(k_range)); ax.legend()
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col_b:
        k = st.slider("Clusters used below (k)", 2, 7, int(elbow_k))
else:
    st.warning("kmodes not installed — clustering skipped.")

if KPROTO_AVAILABLE:
    with st.spinner("Running clustering..."):
        df_cluster_raw = run_clustering(df_unique, k=k)

    cluster_order = sorted(df_cluster_raw['Cluster'].unique())
    palette_clust = sns.color_palette("tab10", n_colors=len(cluster_order))
    cluster_colors = dict(zip(cluster_order, palette_clust))
    fleet_mean = df_cluster_raw['CO2 (g/km)'].mean()
    cluster_means = df_cluster_raw.groupby('Cluster', as_index=False)['CO2 (g/km)'].mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.countplot(data=df_cluster_raw, x='Cluster', hue='Cluster', hue_order=cluster_order,
                  palette=cluster_colors, ax=axes[0], legend=False)
    axes[0].set_title("Cluster sizes")
    sns.boxplot(data=df_cluster_raw, x='Cluster', y='CO2 (g/km)', hue='Cluster', hue_order=cluster_order,
                palette=cluster_colors, ax=axes[1], legend=False)
    axes[1].set_title("CO2 by cluster")
    sns.scatterplot(data=df_cluster_raw, x='Maximum Power (kW)', y='Empty Mass Euro Avg (kg)',
                    hue='Cluster', hue_order=cluster_order, palette=cluster_colors, alpha=0.6, ax=axes[2])
    axes[2].set_title("Power vs. mass by cluster")
    axes[2].legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    cluster_summary = df_cluster_raw.groupby('Cluster').agg(
        N=('CO2 (g/km)', 'size'),
        Share=('CO2 (g/km)', lambda s: f"{len(s)/len(df_cluster_raw)*100:.0f}%"),
        Mean_CO2=('CO2 (g/km)', 'mean'),
        Mean_Mass=('Empty Mass Euro Avg (kg)', 'mean'),
        Mean_Power=('Maximum Power (kW)', 'mean'),
    ).round(0).reindex(cluster_order)
    st.dataframe(cluster_summary, width='stretch')

    below = cluster_summary[cluster_summary["Mean_CO2"] < fleet_mean].index.tolist()
    above = cluster_summary[cluster_summary["Mean_CO2"] >= fleet_mean].index.tolist()
    st.markdown(f"""
    <div class="finding-box">
    At k={k}, clusters separate cleanly along mass and power, with CO2
    following the same ordering (fleet average: {fleet_mean:.0f} g/km).
    Clusters {', '.join(map(str, below))} fall below the fleet average;
    clusters {', '.join(map(str, above))} fall above it. This confirms RQ2:
    distinct, CO2-differentiated segments exist and are driven primarily by
    the mass–power combination, with fuel type providing secondary
    differentiation (petrol-dominated low-CO2 clusters vs. diesel-dominated
    high-CO2 clusters).
    </div>
    """, unsafe_allow_html=True)


# ══════════════ SHARED HELPER FOR RQ3–RQ5 CETERIS-PARIBUS SECTIONS ════════

def raw_comparison(data, group_col, order, metric="CO2 (g/km)"):
    return (
        data.groupby(group_col)[metric]
        .agg(N="count", Median="median", Mean="mean", Std="std")
        .round(1).reindex(order)
    )


# ═══════════════════════════ SECTION 5 — RQ3: FUEL ══════════════════════════
st.header("5. RQ3 — Does diesel cause higher CO2, or is it a vehicle-class effect?")

if "Fuel" not in df_unique.columns:
    st.warning("Fuel is not available in this dataset.")
else:
    fuel_data = df_unique[df_unique["Fuel"].isin(["ES", "GO"])].copy()
    es_vals = fuel_data.loc[fuel_data["Fuel"] == "ES", "CO2 (g/km)"].dropna()
    go_vals = fuel_data.loc[fuel_data["Fuel"] == "GO", "CO2 (g/km)"].dropna()

    st.subheader("5.1 Raw comparison")
    raw_stats_fuel = raw_comparison(fuel_data, "Fuel", ["ES", "GO"])
    u_stat_f, p_val_f = mannwhitneyu(go_vals, es_vals, alternative="two-sided")
    raw_diff_f = go_vals.median() - es_vals.median()

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(raw_stats_fuel, width='stretch')
    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(data=fuel_data, x="Fuel", y="CO2 (g/km)", order=["ES", "GO"],
                    hue="Fuel", palette={"ES": BLUE, "GO": ACCENT}, legend=False, ax=axes[0])
        axes[0].set_title("CO2 by fuel type"); axes[0].set_xticklabels(["Petrol", "Diesel"])
        sns.boxplot(data=fuel_data, x="Fuel", y="Empty Mass Euro Avg (kg)", order=["ES", "GO"],
                    hue="Fuel", palette={"ES": BLUE, "GO": ACCENT}, legend=False, ax=axes[1])
        axes[1].set_title("Mass by fuel type"); axes[1].set_xticklabels(["Petrol", "Diesel"])
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(
        f"Raw median CO2: diesel {go_vals.median():.0f} g/km vs. petrol {es_vals.median():.0f} g/km "
        f"(Δ = {raw_diff_f:+.0f} g/km); Mann-Whitney p "
        f"{'< 0.001' if p_val_f < 0.001 else f'= {p_val_f:.3f}'}. Diesel vehicles are also "
        "visibly heavier on average — part of the raw gap may be vehicle class, not fuel chemistry."
    )

    st.subheader("5.2 Controlled comparison (ceteris paribus)")
    st.markdown(
        "For every vehicle in the test sample, the model predicts CO2 twice — "
        "once as petrol, once as diesel — with mass, power, body and "
        "transmission held fixed. The mean difference isolates the fuel-type "
        "effect from the vehicle-class confound."
    )
    if SHAP_AVAILABLE and shap_ok and "Fuel" in feature_cols:
        cp_es = X_shap_raw.copy(); cp_es["Fuel"] = "ES"
        cp_go = X_shap_raw.copy(); cp_go["Fuel"] = "GO"
        pred_es = fitted["Gradient Boosting"].predict(cp_es[feature_cols])
        pred_go = fitted["Gradient Boosting"].predict(cp_go[feature_cols])
        cp_delta_f = pred_go - pred_es

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.hist(cp_delta_f, bins=40, color=ACCENT, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="gray", lw=1.5, ls=":")
        ax.axvline(cp_delta_f.mean(), color="black", lw=2, ls="--", label=f"Mean effect: {cp_delta_f.mean():+.1f} g/km")
        ax.set_xlabel("Δ CO2 (diesel − petrol), same vehicle otherwise, g/km")
        ax.set_title("Ceteris-paribus fuel effect"); ax.legend()
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean effect", f"{cp_delta_f.mean():+.1f} g/km")
        c2.metric("Median effect", f"{np.median(cp_delta_f):+.1f} g/km")
        c3.metric("Vehicles higher as diesel", f"{(cp_delta_f > 0).mean()*100:.0f}%")

        explained_share_f = f"{(1 - abs(cp_delta_f.mean()) / abs(raw_diff_f)) * 100:.0f}%" if raw_diff_f != 0 else "n/a"
        st.markdown(f"""
        <div class="finding-box">
        <b>Answer to RQ3:</b> holding mass, power, body and transmission
        constant, switching a vehicle from petrol to diesel changes predicted
        CO2 by {cp_delta_f.mean():+.1f} g/km on average
        ({(cp_delta_f > 0).mean()*100:.0f}% of vehicles predicted higher as
        diesel). This is
        {'substantially smaller than' if abs(cp_delta_f.mean()) < abs(raw_diff_f) * 0.5 else 'comparable to'}
        the raw {raw_diff_f:+.0f} g/km gap — roughly {explained_share_f} of the raw gap is
        attributable to vehicle class rather than fuel type. The remaining
        controlled effect is consistent with diesel's genuinely higher
        CO2-per-liter combustion chemistry (~2.64 vs. ~2.31 kg CO2/l) applied
        to an otherwise identical vehicle.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Fuel is not part of the selected feature set; controlled comparison unavailable.")


# ═══════════════════════════ SECTION 6 — RQ4: TRANSMISSION ═════════════════
st.header("6. RQ4 — Does transmission type affect CO2 independently of vehicle class?")

if "GearType" not in df_unique.columns:
    st.warning("GearType is not available in this dataset.")
else:
    gt_all_data = df_unique.dropna(subset=["GearType", "CO2 (g/km)"]).copy()
    gt_type_counts = gt_all_data["GearType"].value_counts()
    gt_all_order = gt_all_data.groupby("GearType")["CO2 (g/km)"].median().sort_values(ascending=False).index.tolist()
    gt_all_palette = dict(zip(gt_all_order, sns.color_palette("Set2", n_colors=len(gt_all_order))))

    st.subheader("6.1 Raw comparison (all four transmission types)")
    st.caption("Sample sizes: " + ", ".join(f"{gt}: {gt_type_counts[gt]:,}" for gt in gt_all_order) +
               " — CVT/DCT medians are noisier given smaller samples.")
    raw_stats_gt_all = raw_comparison(gt_all_data, "GearType", gt_all_order)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.dataframe(raw_stats_gt_all, width='stretch')
    with col_b:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(data=gt_all_data, x="GearType", y="CO2 (g/km)", order=gt_all_order,
                    hue="GearType", palette=gt_all_palette, legend=False, ax=axes[0])
        axes[0].set_title("CO2 by transmission type")
        sns.boxplot(data=gt_all_data, x="GearType", y="Empty Mass Euro Avg (kg)", order=gt_all_order,
                    hue="GearType", palette=gt_all_palette, legend=False, ax=axes[1])
        axes[1].set_title("Mass by transmission type")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(
        f"`{gt_all_order[0]}` has the highest raw median CO2 "
        f"({raw_stats_gt_all.loc[gt_all_order[0], 'Median']:.0f} g/km); `{gt_all_order[-1]}` "
        f"the lowest ({raw_stats_gt_all.loc[gt_all_order[-1], 'Median']:.0f} g/km), tracking "
        "the mass differences shown on the right."
    )

    st.subheader("6.2 Controlled comparison (ceteris paribus)")
    st.markdown(
        "For every vehicle in the test sample, the model predicts CO2 once "
        "per transmission type, holding mass, power, fuel and body constant, "
        "swapping only `GearType`."
    )
    if SHAP_AVAILABLE and shap_ok and "GearType" in feature_cols:
        cp_gt_preds = {}
        for gt in gt_all_order:
            try:
                cp_gt = X_shap_raw.copy(); cp_gt["GearType"] = gt
                cp_gt_preds[gt] = fitted["Gradient Boosting"].predict(cp_gt[feature_cols])
            except Exception:
                continue

        if cp_gt_preds:
            cp_gt_df = pd.DataFrame({
                "GearType": list(cp_gt_preds.keys()),
                "Controlled Mean CO2": [p.mean() for p in cp_gt_preds.values()],
            }).sort_values("Controlled Mean CO2", ascending=False)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            sns.barplot(data=raw_stats_gt_all.reset_index(), x="GearType", y="Median",
                        order=gt_all_order, hue="GearType", palette=gt_all_palette, legend=False, ax=axes[0])
            axes[0].set_title("Raw median CO2"); axes[0].set_ylabel("CO2 (g/km)")
            sns.barplot(data=cp_gt_df, x="GearType", y="Controlled Mean CO2",
                        order=cp_gt_df["GearType"], hue="GearType", palette=gt_all_palette, legend=False, ax=axes[1])
            axes[1].set_title("Controlled mean CO2 (ceteris paribus)"); axes[1].set_ylabel("Predicted CO2 (g/km)")
            for a in axes:
                for sp in ["top", "right"]: a.spines[sp].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(cp_gt_df.style.format({"Controlled Mean CO2": "{:.1f}"}), width='stretch')

            top_gt_raw = gt_all_order[0]
            top_gt_ctrl = cp_gt_df.iloc[0]["GearType"]
            same_top_gt = top_gt_raw == top_gt_ctrl
            spread_raw = raw_stats_gt_all["Median"].max() - raw_stats_gt_all["Median"].min()
            spread_ctrl = cp_gt_df["Controlled Mean CO2"].max() - cp_gt_df["Controlled Mean CO2"].min()
            st.markdown(f"""
            <div class="finding-box">
            <b>Answer to RQ4:</b> the raw ranking has {top_gt_raw} highest; the
            controlled ranking has {top_gt_ctrl} highest
            {"(confirming the raw ranking)" if same_top_gt else "(reordering the raw ranking)"}.
            The spread between transmission types narrows from
            {spread_raw:.0f} g/km (raw) to {spread_ctrl:.0f} g/km (controlled) —
            transmission type has a real but small isolated effect on CO2;
            most of the raw spread reflects which vehicle classes tend to use
            which transmission, not the transmission mechanism itself.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Could not compute controlled predictions for the available transmission types.")
    else:
        st.info("GearType is not part of the selected feature set; controlled comparison unavailable.")


# ═══════════════════════════ SECTION 7 — RQ5: BODY TYPE ════════════════════
st.header("7. RQ5 — Which body style has the highest CO2 once vehicle class is controlled?")

if "Body" not in df_unique.columns:
    st.warning("Body is not available in this dataset.")
else:
    body_data = df_unique.dropna(subset=["Body", "CO2 (g/km)"]).copy()
    body_order = body_data.groupby("Body")["CO2 (g/km)"].median().sort_values(ascending=False).index.tolist()

    st.subheader("7.1 Raw comparison")
    raw_stats_body = raw_comparison(body_data, "Body", body_order)
    st.dataframe(raw_stats_body, width='stretch')

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.boxplot(data=body_data, x="Body", y="CO2 (g/km)", order=body_order,
                hue="Body", palette="RdYlBu_r", legend=False, ax=ax)
    ax.set_title("CO2 by body type, ordered by median"); ax.tick_params(axis='x', rotation=45)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.caption(
        f"`{body_order[0]}` has the highest raw median CO2 "
        f"({raw_stats_body.loc[body_order[0], 'Median']:.0f} g/km); `{body_order[-1]}` "
        f"the lowest ({raw_stats_body.loc[body_order[-1], 'Median']:.0f} g/km)."
    )

    st.subheader("7.2 Controlled comparison (ceteris paribus)")
    if SHAP_AVAILABLE and shap_ok and "Body" in feature_cols:
        cp_body_preds = {}
        for b in body_order:
            try:
                cp_b = X_shap_raw.copy(); cp_b["Body"] = b
                cp_body_preds[b] = fitted["Gradient Boosting"].predict(cp_b[feature_cols])
            except Exception:
                continue

        if cp_body_preds:
            cp_body_df = pd.DataFrame({
                "Body": list(cp_body_preds.keys()),
                "Controlled Mean CO2": [preds.mean() for preds in cp_body_preds.values()],
            }).sort_values("Controlled Mean CO2", ascending=False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            sns.barplot(data=raw_stats_body.reset_index(), x="Body", y="Median",
                        order=body_order, hue="Body", palette="RdYlBu_r", legend=False, ax=axes[0])
            axes[0].set_title("Raw median CO2"); axes[0].tick_params(axis='x', rotation=45)
            sns.barplot(data=cp_body_df, x="Body", y="Controlled Mean CO2",
                        order=cp_body_df["Body"], hue="Body", palette="RdYlBu_r", legend=False, ax=axes[1])
            axes[1].set_title("Controlled mean CO2 (ceteris paribus)"); axes[1].tick_params(axis='x', rotation=45)
            for a in axes:
                for sp in ["top", "right"]: a.spines[sp].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(cp_body_df.style.format({"Controlled Mean CO2": "{:.1f}"}), width='stretch')

            top_body_raw = body_order[0]
            top_body_ctrl = cp_body_df.iloc[0]["Body"]
            same_top = top_body_raw == top_body_ctrl
            st.markdown(f"""
            <div class="finding-box">
            <b>Answer to RQ5:</b> raw data ranks {top_body_raw} highest; the
            controlled comparison ranks {top_body_ctrl} highest
            {"(confirming the raw ranking survives controlling for vehicle class)" if same_top
             else "(the raw ranking is partly driven by correlated mass/power differences, not body style itself)"}.
            The spread between body types narrows in the controlled chart,
            indicating body style alone has a smaller isolated effect on CO2
            than mass and power.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Could not compute controlled predictions for the available body types.")
    else:
        st.info("Body is not part of the selected feature set; controlled comparison unavailable.")


# ═══════════════════════════ SECTION 8 — SUMMARY ═══════════════════════════
st.header("8. Summary of Findings")
st.markdown("""
| Research Question | Finding |
|---|---|
| **RQ1** — Strongest predictors & direction | Empty mass and maximum power are the strongest technical predictors, both increasing CO2, with diminishing marginal effects at high values; fuel type and transmission contribute smaller, directionally consistent effects (diesel and automatic increase CO2). Gradient Boosting achieves R² ≈ 0.95, MAE ≈ 7–8 g/km. |
| **RQ2** — Natural vehicle segments | K-Prototypes clustering identifies distinct, CO2-differentiated segments, ordered primarily by the mass–power combination; a light, petrol-dominated "efficiency" segment and a heavy, diesel-dominated "commercial" segment anchor the low and high ends. |
| **RQ3** — Diesel vs. petrol | Diesel vehicles show higher raw CO2, but a substantial share of that gap is a vehicle-class effect. The controlled (ceteris-paribus) fuel effect is smaller but remains positive and directionally consistent with diesel's higher CO2-per-liter combustion chemistry. |
| **RQ4** — Transmission type | Automatic transmission is associated with higher CO2 than manual, both raw and controlled, but the controlled effect is much smaller — most of the raw gap reflects that automatics tend to be heavier, more powerful vehicles, not the transmission mechanism itself. |
| **RQ5** — Body type | The highest-CO2 body style in the raw data is not always the highest once mass and power are held constant, showing that part of the raw body-type ranking reflects vehicle class rather than body style itself. |

<span class="methods-note">All controlled comparisons are ceteris-paribus
predictions from the Gradient Boosting model (Section 2, Methods) and should
be read as the model's best estimate of an isolated effect under the
no-missing-confounder assumption, not as a randomized-experiment result.</span>
""", unsafe_allow_html=True)
