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

# ── Design System ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --mint:#00C8A0;--mint-lo:#00C8A015;--red:#E84855;--red-lo:#E8485515;
  --amber:#F5A623;--amber-lo:#F5A62315;--bg:#0F1117;--card:#1A1D27;
  --card2:#20233000;--border:#2A2D3A;--text:#E8EAF0;--muted:#7B8094;
  --font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace
}
html,body,.stApp{background:var(--bg)!important;color:var(--text)!important;font-family:var(--font)!important}
[data-testid="stSidebar"]{background:#12141E!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
[data-testid="stSidebar"] a{color:var(--mint)!important;text-decoration:none;font-weight:500}
[data-testid="stSidebar"] a:hover{text-decoration:underline}
[data-testid="stSidebar"] hr{border-top:1px solid var(--border)!important}
h1{font-size:2rem!important;font-weight:900!important;letter-spacing:-0.03em!important;color:var(--text)!important}
h2{font-size:1.3rem!important;font-weight:700!important;letter-spacing:-0.02em!important;border-bottom:1px solid var(--border)!important;padding-bottom:.35rem!important;margin-top:1.6rem!important;color:var(--text)!important}
h3{font-size:1.05rem!important;font-weight:600!important;color:var(--text)!important}
h4{font-size:.8rem!important;font-weight:600!important;color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.09em!important}
p,li{color:var(--text)!important;line-height:1.65!important}
[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:1rem 1.25rem!important;border-top:3px solid var(--mint)!important;transition:border-color .2s}
[data-testid="stMetric"]:hover{border-color:var(--mint)!important}
[data-testid="stMetricLabel"] p{font-size:.7rem!important;font-weight:600!important;color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.09em!important;margin:0!important}
[data-testid="stMetricValue"]{font-size:1.65rem!important;font-weight:700!important;color:var(--text)!important;font-family:var(--mono)!important;font-feature-settings:'tnum' 1!important}
[data-testid="stMetricDelta"] div{font-size:.78rem!important}
[data-testid="stTabPanel"]{padding-top:1.5rem!important}
button[data-baseweb="tab"]{background:transparent!important;border:none!important;border-bottom:2px solid transparent!important;color:var(--muted)!important;font-size:.82rem!important;font-weight:500!important;padding:.55rem .9rem!important;transition:all .15s!important;font-family:var(--font)!important}
button[data-baseweb="tab"]:hover{color:var(--text)!important}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--mint)!important;border-bottom:2px solid var(--mint)!important;font-weight:600!important}
[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden!important}
[data-testid="stExpander"] summary{font-weight:500!important;font-size:.88rem!important;padding:.7rem 1rem!important;color:var(--text)!important}
[data-testid="stExpander"] summary:hover{color:var(--mint)!important}
.stButton>button{background:var(--mint)!important;color:#0F1117!important;border:none!important;border-radius:6px!important;font-weight:700!important;font-size:.84rem!important;letter-spacing:.04em!important;padding:.5rem 1.4rem!important;transition:opacity .15s!important}
.stButton>button:hover{opacity:.85!important}
[data-testid="stFormSubmitButton"]>button{background:linear-gradient(135deg,#00C8A0,#00A882)!important;color:#0F1117!important;font-weight:700!important;border-radius:8px!important;border:none!important;padding:.7rem 2rem!important;letter-spacing:.04em!important;font-size:.9rem!important}
[data-testid="stFormSubmitButton"]>button:hover{opacity:.88!important}
[data-testid="stSelectbox"] label,[data-testid="stSlider"] label,
[data-testid="stRadio"] label,[data-testid="stSelectSlider"] label{color:var(--muted)!important;font-size:.78rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important}
[data-testid="stSelectbox"]>div>div{background:var(--card)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:6px!important}
[data-testid="stRadio"] div[role="radiogroup"] label{color:var(--text)!important;font-size:.88rem!important;text-transform:none!important;letter-spacing:normal!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:8px!important;overflow:hidden!important}
[data-testid="stDataFrame"] th{background:var(--card)!important;color:var(--muted)!important;font-size:.72rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important;border-bottom:1px solid var(--border)!important}
[data-testid="stDataFrame"] td{font-family:var(--mono)!important;font-size:.83rem!important;color:var(--text)!important}
[data-testid="stCaptionContainer"] p,.stCaption{color:var(--muted)!important;font-size:.77rem!important;line-height:1.55!important}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.5rem 0!important}
[data-testid="stAlert"]{border-radius:8px!important;border:none!important}
[data-testid="stFileUploader"]{background:var(--card)!important;border:1px dashed var(--border)!important;border-radius:8px!important}
[data-testid="stFileUploader"] *{color:var(--text)!important}
[data-testid="stSpinner"]{color:var(--mint)!important}
.stSuccess{background:var(--mint-lo)!important;border-left:3px solid var(--mint)!important;color:var(--text)!important}
.stInfo{background:var(--mint-lo)!important;border-left:3px solid var(--mint)!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}
table{border-collapse:collapse!important;width:100%}
table th{background:var(--card)!important;color:var(--muted)!important;font-size:.72rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.07em!important;border:1px solid var(--border)!important;padding:.5rem .75rem!important}
table td{background:var(--bg)!important;color:var(--text)!important;border:1px solid var(--border)!important;padding:.45rem .75rem!important;font-size:.85rem!important}
</style>
""", unsafe_allow_html=True)

# ── Color tokens (Python side) ────────────────────────────────────────────────
MINT    = "#00C8A0"
RED     = "#E84855"
AMBER   = "#F5A623"
BG      = "#1A1D27"
BORDER  = "#2A2D3A"
MUTED   = "#7B8094"
TEXT    = "#E8EAF0"
BLUE    = MINT          # legacy alias so existing code still works
RANDOM_STATE = 42

# ── Matplotlib theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     BG,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    MUTED,
    "axes.titlecolor":    TEXT,
    "axes.titlesize":     13,
    "axes.titleweight":   "600",
    "axes.labelsize":     10,
    "axes.grid":          True,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "grid.color":         BORDER,
    "grid.linewidth":     0.5,
    "text.color":         TEXT,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "figure.titlesize":   15,
    "figure.titleweight": "700",
    "legend.facecolor":   BG,
    "legend.edgecolor":   BORDER,
    "legend.fontsize":    9,
    "font.family":        "DejaVu Sans",
    "image.cmap":         "viridis",
    "savefig.facecolor":  BG,
    "savefig.edgecolor":  BG,
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

    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    if all(c in df.columns for c in ["HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)"]):
        df['hc_calc']  = df['HC+NOX (g/km)'] - df['NOX (g/km)']
        df['nox_calc'] = df['HC+NOX (g/km)'] - df['HC (g/km)']
        df['hc_calc']  = df['hc_calc'].fillna(df['HC (g/km)'])
        df['nox_calc'] = df['nox_calc'].fillna(df['NOX (g/km)'])
        df["HC (g/km)"]    = df["hc_calc"]
        df["NOX (g/km)"]   = df["nox_calc"]
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


GEAR_TYPE_MAP = {"M": "Manual", "A": "Automatic", "V": "CVT",
                  "D": "DCT", "N": "Automatic", "S": "Manual"}

@st.cache_data(show_spinner=False)
def make_df_unique(df: pd.DataFrame) -> pd.DataFrame:
    if "Fuel" not in df.columns:
        return df
    df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy()

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

    results_df = pd.DataFrame(results).sort_values("Test_MAE", ascending=True)

    rf_pipe  = fitted["Random Forest"]
    rf_pre   = rf_pipe.named_steps["pre"]
    rf_model = rf_pipe.named_steps["m"]
    feat_names = rf_pre.get_feature_names_out()
    fi_df = pd.DataFrame({"Feature": feat_names,
                           "Importance": rf_model.feature_importances_})\
              .sort_values("Importance", ascending=False)

    return (fitted, results_df, fs_df, best_fs, feature_cols,
            X_train, X_test, y_train, y_test, rf_pipe, fi_df, num_f, cat_f)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1rem'>
      <div style='font-size:1.5rem;font-weight:900;letter-spacing:-0.03em;color:#E8EAF0'>
        🚗 CO₂ Dashboard
      </div>
      <div style='font-size:.72rem;font-weight:600;color:#7B8094;text-transform:uppercase;
                  letter-spacing:.1em;margin-top:.2rem'>
        ADEME · France · 2013
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Dataset loads automatically from the repository.")
    uploaded = st.file_uploader("Upload custom CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown(
        "**Source:** [GitHub ↗](https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction)",
        unsafe_allow_html=True
    )

# ── Load data ────────────────────────────────────────────────────────────────
source = uploaded.read() if uploaded is not None else CSV_URL

with st.spinner("Loading and preprocessing data …"):
    try:
        df        = load_and_preprocess(source)
        df_unique = make_df_unique(df)
        df_combus = df[df['Fuel'].isin(['ES', 'GO'])].copy() if 'Fuel' in df.columns else df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

with st.sidebar:
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:.72rem;color:#7B8094;line-height:1.8'>
      <span style='color:#00C8A0;font-weight:600'>{len(df):,}</span> raw rows<br>
      <span style='color:#00C8A0;font-weight:600'>{len(df_unique):,}</span> unique ES/GO configs
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Preprocessing",
    "📊 EDA",
    "🔗 Correlations",
    "📉 Deduplication",
    "🔵 Clustering",
    "🤖 Prediction",
    "🎯 CO₂ Calculator",
])

# ═══════════════════════ TAB 0 – PREPROCESSING ═══════════════════════════════
with tabs[0]:
    st.header("Preprocessing & Dataset Overview")

    st.markdown(
        "Every transformation applied to the raw ADEME dataset before analysis — "
        "raw data is rarely analysis-ready and hidden issues can silently bias every "
        "downstream result."
    )

    st.subheader("Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",            f"{len(df):,}")
    c2.metric("Columns",               len(df.columns))
    c3.metric("ES+GO Vehicles",        f"{len(df_combus):,}")
    c4.metric("Unique Configurations", f"{len(df_unique):,}")

    st.markdown(
        "**Source:** ADEME Car Labelling Dataset — official French vehicle emission registry (2013). "
        "Contains all vehicles type-approved for sale in France. "
        "All column names were translated from French to English."
    )

    st.markdown("---")
    st.subheader("Preprocessing Pipeline")

    steps = [
        ("🏷️ Column Renaming",
         "All 25 French column names were mapped to English equivalents "
         "(e.g. `Marque` → `Brand`, `gamme` → `Range`, `Carrosserie` → `Body`). "
         "This ensures consistency across the entire analysis."),
        ("⚕️ HC/NOX Imputation",
         "34,447 values for `HC (g/km)` were missing. These were recovered "
         "algebraically: `HC = HC+NOX − NOX` and `NOX = HC+NOX − HC`. "
         "This reduced missing pollutant data from 34,447 to 303 rows."),
        ("🔧 Gearbox Code Fix",
         "Two erroneous gearbox codes were corrected: `N 0` and `N 1` → `A 0` "
         "(Automatic), `S 6` → `D 6` (Dual-Clutch). These were data entry errors "
         "in the original registry."),
        ("⚡ Electric Vehicle Treatment",
         "39 electric vehicles (Fuel = `EL`) had NaN values for all emission and "
         "consumption columns. These were set to 0 — EVs have no tailpipe emissions."),
        ("⚖️ Average Kerb Weight",
         "The dataset provides minimum and maximum kerb weight separately. "
         "Merged into `Empty Mass Euro Avg (kg)` = (min + max) / 2, "
         "original columns dropped."),
        ("⚙️ Gearbox Split",
         "The `Gearbox` column contains codes like `A 6` (Automatic, 6 gears). "
         "Split into `GearType` (Manual / Automatic / CVT / DCT) "
         "and `GearCount` (4–8, numerical)."),
        ("🔢 Numeric Type Coercion",
         "Four key columns cast to `float64` using `pd.to_numeric(..., errors='coerce')` "
         "to handle any non-numeric entries."),
    ]

    for icon_title, explanation in steps:
        with st.expander(icon_title):
            st.markdown(explanation)

    st.markdown("---")
    st.subheader("Missing Values Analysis")
    st.markdown(
        "The heatmap shows the **pattern** of missing values (highlighted = missing). "
        "The bar chart shows the **count** per column."
    )

    missing_values = df.isnull().sum()
    missing_sorted = missing_values[missing_values > 0].sort_values(ascending=False)

    if len(missing_sorted) > 0:
        cols_with_na = missing_sorted.index.tolist()
        from matplotlib.colors import ListedColormap
        cmap_mv = ListedColormap([BORDER, MINT])

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.patch.set_facecolor(BG)
        sns.heatmap(df[cols_with_na].isna(), cmap=cmap_mv, cbar=False,
                    yticklabels=False, ax=axes[0])
        axes[0].set_title("Missing Values Pattern")
        axes[0].tick_params(axis="x", rotation=90, colors=MUTED)

        bars = axes[1].bar(range(len(missing_sorted)), missing_sorted.values,
                           color=MINT, alpha=0.85)
        axes[1].set_xticks(range(len(missing_sorted)))
        axes[1].set_xticklabels(missing_sorted.index, rotation=90)
        axes[1].set_title("Missing Values per Column")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(
            f"Remaining missing values after imputation: "
            f"{missing_sorted.sum():,} across {len(missing_sorted)} columns."
        )
    else:
        st.success("No missing values after preprocessing!")

    st.markdown("---")
    st.subheader("Column Overview")
    col_info = []
    for col in df.columns:
        dtype  = str(df[col].dtype)
        n_null = df[col].isnull().sum()
        n_uniq = df[col].nunique()
        if df[col].dtype in [np.float64, np.int64]:
            summary = f"min={df[col].min():.1f} / mean={df[col].mean():.1f} / max={df[col].max():.1f}"
        else:
            top = df[col].value_counts().index[0] if n_uniq > 0 else "—"
            summary = f"Top: {top} ({df[col].value_counts().iloc[0]:,}x)"
        col_info.append({
            "Column": col, "Type": dtype,
            "Missing": n_null, "Unique": n_uniq, "Summary": summary
        })
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    desc = df.describe(include="all").T
    num_cols_desc = desc.select_dtypes(include="number").columns.tolist()
    fmt = {c: "{:.2f}" for c in num_cols_desc}
    st.dataframe(desc.style.format(fmt, na_rep="-"), use_container_width=True)

    st.markdown("---")
    st.subheader("Sample Data — First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)


# ═══════════════════════ TAB 1 – EDA ═════════════════════════════════════════
with tabs[1]:
    st.header("Fleet-wide Distribution & Frequency Analysis")

    st.subheader("CO₂ Emissions — Target Variable (before Deduplication)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle('CO₂ Emissions Analysis — Target Variable', fontsize=15, fontweight='700', color=TEXT)

    co2_data = df['CO2 (g/km)'].dropna()

    axes[0,0].hist(co2_data, bins=50, alpha=0.85, color=MINT, edgecolor=BG, linewidth=0.4)
    axes[0,0].axvline(co2_data.mean(),   color=RED,   linestyle='--', lw=1.5,
                      label=f'Mean: {co2_data.mean():.1f}')
    axes[0,0].axvline(co2_data.median(), color=AMBER, linestyle='--', lw=1.5,
                      label=f'Median: {co2_data.median():.1f}')
    axes[0,0].set_title('Distribution of CO₂ Emissions')
    axes[0,0].set_xlabel('CO₂ (g/km)'); axes[0,0].set_ylabel('Frequency')
    axes[0,0].legend()

    bp = axes[0,1].boxplot(co2_data, patch_artist=True,
                            medianprops=dict(color=MINT, lw=2),
                            whiskerprops=dict(color=MUTED),
                            capprops=dict(color=MUTED),
                            flierprops=dict(marker='o', color=AMBER, alpha=0.4, markersize=3))
    for patch in bp['boxes']:
        patch.set_facecolor(f"{MINT}30")
        patch.set_edgecolor(MINT)
    axes[0,1].set_title('CO₂ Emissions Box Plot')
    axes[0,1].set_ylabel('CO₂ (g/km)')

    stats.probplot(co2_data, dist="norm", plot=axes[1,0])
    axes[1,0].get_lines()[0].set(color=MINT, alpha=0.5, markersize=2)
    axes[1,0].get_lines()[1].set(color=RED, lw=2)
    axes[1,0].set_title('Q-Q Plot (Normality Check)')

    axes[1,1].axis('off')
    txt = (f"Mean:    {co2_data.mean():.1f} g/km\n"
           f"Median:  {co2_data.median():.1f} g/km\n"
           f"Std Dev: {co2_data.std():.1f} g/km\n"
           f"Min:     {co2_data.min():.0f} g/km\n"
           f"Max:     {co2_data.max():.0f} g/km\n"
           f"N:       {len(co2_data):,}")
    axes[1,1].text(0.5, 0.5, txt, fontsize=13, ha='center', va='center',
                   color=TEXT, fontfamily='monospace',
                   bbox=dict(facecolor=f"{MINT}15", edgecolor=MINT,
                             alpha=1, boxstyle='round,pad=0.8', lw=1))
    axes[1,1].set_title('Summary Statistics')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Fleet-wide Distributions")
    fig, axes = plt.subplots(5, 2, figsize=(20, 30))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Fleet-wide Distribution and Frequency Analysis', fontsize=18, y=1.01, color=TEXT, fontweight='700')

    def _hist(ax, data, title):
        ax.hist(data.dropna(), bins=80, alpha=0.85, color=MINT, edgecolor=BG, lw=0.3)
        ax.set_title(title)

    _hist(axes[0,0], df['Empty Mass Euro Avg (kg)'], 'Vehicle Mass Distribution (kg)')
    _hist(axes[0,1], df['Maximum Power (kW)'],       'Maximum Power Distribution (kW)')
    _hist(axes[1,0], df['Combined Consumption (l/100km)'], 'Combined Consumption (l/100km)')
    _hist(axes[1,1], df['CO2 (g/km)'],               'CO₂ Emissions Distribution (g/km)')

    def _countplot(ax, col, title, rotate=0):
        if col not in df.columns: return
        order = df[col].value_counts().index
        counts = df[col].value_counts()
        ax.bar(range(len(order)), counts.values, color=MINT, alpha=0.85, edgecolor=BG, lw=0.3)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=rotate, ha='right' if rotate else 'center')
        ax.set_title(title)

    _countplot(axes[2,0], 'Fuel',    'Fuel Type Frequency')
    _countplot(axes[2,1], 'Body',    'Body Type Frequency', 45)
    _countplot(axes[3,0], 'Gearbox', 'Gearbox Frequency')
    _countplot(axes[3,1], 'Range',   'Vehicle Range Frequency', 30)

    top_brands = df['Brand'].value_counts().nlargest(25)
    axes[4,0].barh(top_brands.index, top_brands.values, color=MINT, alpha=0.85, edgecolor=BG, lw=0.3)
    axes[4,0].set_title('Top 25 Brands by Frequency')

    if 'Commerical Designation' in df.columns:
        top_models = df['Commerical Designation'].value_counts().nlargest(15)
        axes[4,1].barh(top_models.index, top_models.values, color=MINT, alpha=0.85, edgecolor=BG, lw=0.3)
        axes[4,1].set_title('Top 15 Vehicle Models by Frequency')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Primary Drivers of CO₂ Emissions")
    fig, axes = plt.subplots(4, 2, figsize=(20, 28))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Primary Drivers of CO₂ Emissions', fontsize=18, y=1.01, color=TEXT, fontweight='700')

    _hist(axes[0,0], df['CO2 (g/km)'], 'Distribution of CO₂ Emissions')

    for ax, xcol, title in [
        (axes[0,1], 'Empty Mass Euro Avg (kg)', 'Vehicle Mass vs CO₂'),
        (axes[1,0], 'Maximum Power (kW)',        'Maximum Power vs CO₂'),
        (axes[1,1], 'Combined Consumption (l/100km)', 'Combined Consumption vs CO₂'),
    ]:
        ax.scatter(df[xcol], df['CO2 (g/km)'], alpha=0.25, s=6, color=MINT, rasterized=True)
        ax.set_xlabel(xcol); ax.set_ylabel('CO₂ (g/km)'); ax.set_title(title)

    for ax, col, title, rot in [
        (axes[2,0], 'Fuel',    'CO₂ by Fuel Type',    0),
        (axes[2,1], 'Body',    'CO₂ by Body Type',   45),
        (axes[3,0], 'Gearbox', 'CO₂ by Gearbox Type',45),
    ]:
        cats = sorted(df[col].dropna().unique())
        data_bp = [df[df[col]==c]['CO2 (g/km)'].dropna().values for c in cats]
        bp2 = ax.boxplot(data_bp, patch_artist=True,
                          medianprops=dict(color=MINT, lw=1.5),
                          whiskerprops=dict(color=MUTED, lw=0.8),
                          capprops=dict(color=MUTED),
                          flierprops=dict(marker='o', color=AMBER, alpha=0.3, markersize=2))
        colors_cycle = [f"{MINT}30", f"{AMBER}30", f"{RED}30", "#8888AA30"]
        for i, patch in enumerate(bp2['boxes']):
            patch.set_facecolor(colors_cycle[i % len(colors_cycle)])
            patch.set_edgecolor(MINT)
        ax.set_xticklabels(cats, rotation=rot, ha='right' if rot else 'center')
        ax.set_title(title); ax.set_ylabel('CO₂ (g/km)')

    axes[3,1].axis('off')
    summary_text = ("Key Findings\n\n"
                    "① Consumption has the highest correlation with CO₂\n"
                    "② Mass & Power are significant secondary drivers\n"
                    "③ Fuel type drives substantial variance between segments\n"
                    "④ Gearbox & Body show measurable distributional impact")
    axes[3,1].text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center',
                   color=TEXT, fontfamily='monospace',
                   bbox=dict(facecolor=f"{MINT}15", edgecolor=MINT,
                             boxstyle='round,pad=0.9', lw=1))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════ TAB 2 – KORRELATIONEN ═══════════════════════════════
with tabs[2]:
    st.header("Correlation & Statistical Analysis")

    st.subheader("Pearson vs. Spearman Correlation Heatmap")
    st.markdown(
        "**Pearson** measures linear relationships (assumes normality). "
        "**Spearman** measures monotonic relationships (rank-based, robust to outliers). "
        "Large differences between both reveal non-linear relationships."
    )

    df_numeric = df.select_dtypes(include=np.number).copy()
    pearson_corr  = df_numeric.corr(method='pearson')
    spearman_corr = df_numeric.corr(method='spearman')

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    fig.patch.set_facecolor(BG)
    sns.heatmap(pearson_corr,  annot=True, fmt='.2f', cmap='RdYlGn', ax=ax[0],
                annot_kws={"size": 8}, linewidths=0.3, linecolor=BORDER)
    ax[0].set_title('Pearson Correlation')
    sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax[1],
                annot_kws={"size": 8}, linewidths=0.3, linecolor=BORDER)
    ax[1].set_title('Spearman Correlation')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        "CO₂ correlates most strongly with Combined Consumption (Pearson r=0.96, Spearman r=0.98). "
        "Empty mass shows strong correlation (r=0.69). "
        "Maximum power has moderate Pearson (r=0.36) but weaker Spearman (r=0.18) — non-linear relationship."
    )

    st.markdown("---")

    SCATTER_INTERP = {
        "Empty Mass": (
            "**Strong positive correlation** (Pearson r=0.68, Spearman r=0.78, R²=0.46). "
            "46% of CO₂ variance explained by empty mass alone. "
            "Spearman exceeds Pearson — slightly non-linear: for very heavy vehicles (>2,500 kg) "
            "the CO₂ increase per kg diminishes."
        ),
        "Combined Consumption": (
            "**Near-perfect linear correlation** (Pearson r=0.98, R²=0.96). "
            "96% of CO₂ variance is explained by fuel consumption — physically expected, "
            "as CO₂ is directly proportional to combustion (petrol ≈ 2.31 kg/l, diesel ≈ 2.64 kg/l). "
            "Note: Combined Consumption is excluded from the prediction model — "
            "including it would reduce the model to a trivial conversion factor."
        ),
        "Maximum Power": (
            "**Moderate correlation** (Pearson r=0.67, Spearman r=0.54, R²=0.45). "
            "Pearson notably higher than Spearman — predominantly linear with high scatter. "
            "High-performance vehicles (>300 kW) span 200–550 g/km — "
            "power alone is less precise than mass because power correlates strongly with mass."
        ),
    }

    for var_name, var_col in [
        ("Empty Mass", "Empty Mass Euro Avg (kg)"),
        ("Combined Consumption", "Combined Consumption (l/100km)"),
        ("Maximum Power", "Maximum Power (kW)"),
    ]:
        if var_col not in df_unique.columns:
            continue
        st.subheader(f"{var_name} vs CO₂ (Deduplicated Dataset)")
        d = df_unique[[var_col, 'CO2 (g/km)']].dropna()
        pc, _ = pearsonr(d[var_col], d['CO2 (g/km)'])
        sc, _ = spearmanr(d[var_col], d['CO2 (g/km)'])
        r2 = pc**2

        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        fig.patch.set_facecolor(BG)

        axes[0,0].scatter(d[var_col], d['CO2 (g/km)'],
                          alpha=0.35, s=8, color=MINT, rasterized=True)
        m, b = np.polyfit(d[var_col], d['CO2 (g/km)'], 1)
        x_line = np.linspace(d[var_col].min(), d[var_col].max(), 200)
        axes[0,0].plot(x_line, m*x_line+b, color=RED, lw=2, alpha=0.9)
        axes[0,0].set_title(f'r = {pc:.4f} · R² = {r2:.4f}')
        axes[0,0].set_xlabel(var_col); axes[0,0].set_ylabel('CO₂ (g/km)')

        hb = axes[0,1].hexbin(d[var_col], d['CO2 (g/km)'],
                               gridsize=30, cmap='YlGn', mincnt=1)
        axes[0,1].set_title('Density (Hexbin)')
        axes[0,1].set_xlabel(var_col)
        plt.colorbar(hb, ax=axes[0,1])

        axes[1,0].hist(d[var_col], bins=50, alpha=0.85, color=MINT, edgecolor=BG, lw=0.3)
        axes[1,0].set_xlabel(var_col); axes[1,0].set_title(f'Distribution of {var_col}')

        axes[1,1].hist(d['CO2 (g/km)'], bins=50, alpha=0.85, color=AMBER, edgecolor=BG, lw=0.3)
        axes[1,1].set_xlabel('CO₂ (g/km)'); axes[1,1].set_title('Distribution of CO₂')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson r", f"{pc:.4f}")
        c2.metric("Spearman r", f"{sc:.4f}")
        c3.metric("R²", f"{r2:.4f}")

        if var_name in SCATTER_INTERP:
            st.caption(SCATTER_INTERP[var_name])

        st.markdown("---")


# ═══════════════════════ TAB 3 – DEDUPLICATION ═══════════════════════════════
with tabs[3]:
    st.header("Data Deduplication — Unique Mechanical Configurations")

    total_obs          = len(df)
    filtered_obs       = len(df_combus)
    unique_designs     = len(df_unique)
    duplicates_removed = filtered_obs - unique_designs
    redundancy_pct     = (duplicates_removed / filtered_obs * 100) if filtered_obs > 0 else 0
    top_clone          = int(df_unique['Clone_Count'].iloc[0]) if len(df_unique) > 0 else 0

    st.markdown(f"""
    The raw ADEME dataset contains **{total_obs:,} records** — but most are not unique vehicles.
    The same mechanical configuration appears hundreds of times under different trim names or option packages.
    Including duplicates would **bias every statistical analysis and ML model** towards the most common
    configurations (Mercedes-Benz Minibuses dominate ~86% of raw records).
    """)

    with st.expander("How deduplication works — three steps"):
        st.markdown("""
        **Step 1 — Fuel filter:** Only petrol (`ES`) and diesel (`GO`) vehicles are kept.
        This reduces the dataset from **44,850 → 43,935 records**.

        **Step 2 — Define a unique mechanical configuration:** A vehicle is unique if it has a
        distinct combination of: `Brand · Folder Model · Fuel · Body · Gearbox · Maximum Power (kW) ·
        Empty Mass Euro Avg (kg) · CO₂ (g/km) · Combined Consumption (l/100km) · Range`

        **Step 3 — Group and count:** For each unique configuration, the number of duplicate rows
        (`Clone_Count`) is recorded. Result: **5,700 unique mechanical configurations**.
        """)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{total_obs:,}")
    c2.metric("ES+GO Filter",    f"{filtered_obs:,}")
    c3.metric("Unique Designs",  f"{unique_designs:,}")
    c4.metric("Redundancy Rate", f"{redundancy_pct:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    categories = ['Total Records', 'Unique Designs']
    values = [total_obs, unique_designs]
    bar_colors = [f"{MUTED}80", MINT]
    bars = ax.bar(categories, values, color=bar_colors, edgecolor=BG, lw=0)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, v + total_obs*0.01,
                f'{v:,}', ha='center', va='bottom', fontweight='700',
                fontsize=14, color=TEXT)
    stats_txt = f"Redundancy: {redundancy_pct:.1f}%\nUnique: {unique_designs:,}\nTotal: {total_obs:,}"
    ax.text(1.35, total_obs*0.60, stats_txt, fontsize=11, color=TEXT,
            bbox=dict(facecolor=f"{MINT}20", edgecolor=MINT, boxstyle='round,pad=0.5', lw=1))
    ax.set_title('Engineering Fleet Diversity', fontsize=14)
    ax.set_ylabel('Number of Observations')
    ax.set_ylim(0, total_obs * 1.18)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("CO₂ Analysis after Deduplication")
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle('CO₂ Emissions (Deduplicated Data)', fontsize=14, fontweight='700', color=TEXT)

    co2_u = df_unique['CO2 (g/km)'].dropna()
    axes[0,0].hist(co2_u, bins=50, alpha=0.85, color=MINT, edgecolor=BG, lw=0.3)
    axes[0,0].axvline(co2_u.mean(),   color=RED,   linestyle='--', lw=1.5,
                      label=f'Mean: {co2_u.mean():.1f}')
    axes[0,0].axvline(co2_u.median(), color=AMBER, linestyle='--', lw=1.5,
                      label=f'Median: {co2_u.median():.1f}')
    axes[0,0].set_title('Distribution (Deduplicated)')
    axes[0,0].legend()

    bp3 = axes[0,1].boxplot(co2_u, patch_artist=True,
                              medianprops=dict(color=MINT, lw=2),
                              whiskerprops=dict(color=MUTED),
                              capprops=dict(color=MUTED),
                              flierprops=dict(marker='o', color=AMBER, alpha=0.4, markersize=3))
    for patch in bp3['boxes']:
        patch.set_facecolor(f"{MINT}30"); patch.set_edgecolor(MINT)
    axes[0,1].set_title('Box Plot (Deduplicated)')

    stats.probplot(co2_u, dist="norm", plot=axes[1,0])
    axes[1,0].get_lines()[0].set(color=MINT, alpha=0.5, markersize=2)
    axes[1,0].get_lines()[1].set(color=RED, lw=2)
    axes[1,0].set_title('Q-Q Plot (Normality Check)')

    axes[1,1].axis('off')
    txt2 = (f"Mean:    {co2_u.mean():.1f} g/km\n"
            f"Median:  {co2_u.median():.1f} g/km\n"
            f"Std Dev: {co2_u.std():.1f} g/km\n"
            f"N unique:{len(co2_u):,}")
    axes[1,1].text(0.5, 0.5, txt2, fontsize=13, ha='center', va='center',
                   color=TEXT, fontfamily='monospace',
                   bbox=dict(facecolor=f"{MINT}15", edgecolor=MINT,
                             boxstyle='round,pad=0.8', lw=1))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Outlier Analysis (IQR Method)")
    st.markdown(
        "IQR method flags values outside **Q1 − 1.5 × IQR** and **Q3 + 1.5 × IQR**. "
        "A statistical outlier ≠ data error. All flagged values are real, homologated vehicles — "
        "documented here for transparency but **not removed**."
    )

    OUTLIER_CONTEXT = {
        "CO2 (g/km)": (
            "**128 statistical outliers** — all upper outliers (>306 g/km). "
            "Examples: Lamborghini Aventador (398 g/km), Bugatti Veyron (596 g/km), "
            "Rolls-Royce Phantom (385 g/km). Real homologated vehicles correctly included."
        ),
        "Maximum Power (kW)": (
            "**477 statistical outliers** — the lower bound of **-5.0 kW is a methodological artefact**. "
            "IQR is symmetric; for right-skewed distributions like power it produces physically impossible "
            "negative bounds. Upper outliers (>243 kW = ~330 HP) are genuine performance vehicles."
        ),
        "Empty Mass Euro Avg (kg)": (
            "**Only 1 statistical outlier** above 2,943 kg — a heavy commercial van. "
            "Mass is the most normally distributed variable, so IQR works well here."
        ),
    }

    for col_name in ['CO2 (g/km)', 'Maximum Power (kW)', 'Empty Mass Euro Avg (kg)']:
        if col_name not in df_unique.columns:
            continue
        Q1 = df_unique[col_name].quantile(0.25)
        Q3 = df_unique[col_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_unique[(df_unique[col_name] < Q1 - 1.5*IQR) |
                              (df_unique[col_name] > Q3 + 1.5*IQR)]
        with st.expander(
            f"**{col_name}**: {len(outliers)} outliers "
            f"(bounds: <{Q1 - 1.5*IQR:.1f} or >{Q3 + 1.5*IQR:.1f})"
        ):
            st.markdown(OUTLIER_CONTEXT.get(col_name, ""))
            st.dataframe(
                outliers[["Brand","Folder Model","Fuel","Body",
                           "Maximum Power (kW)","Empty Mass Euro Avg (kg)","CO2 (g/km)"]
                         ].sort_values(col_name, ascending=False).head(10),
                use_container_width=True
            )

    st.subheader("Top 5 Most Redundant Configurations")
    st.dataframe(df_unique.head(5), use_container_width=True)


# ═══════════════════════ TAB 4 – CLUSTERING ══════════════════════════════════
with tabs[4]:
    st.header("K-Prototypes Clustering")

    st.markdown("""
    > **Research Question:** Which natural vehicle segments can be identified based on
    > technical characteristics (fuel type, body style, gearbox, power, mass)
    > in the French vehicle market 2013, and how do these segments differ in CO₂ emissions?
    """)

    st.subheader("Elbow Method — Optimal Number of Clusters")
    st.markdown(
        "The Elbow Method computes the **total cost** (intra-cluster distance) for k=2–9. "
        "The 'elbow' in the cost curve indicates the optimal k."
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
            model = KPrototypes(n_clusters=k_val, init="Cao", n_init=3,
                                verbose=0, random_state=RANDOM_STATE)
            model.fit_predict(X_matrix, categorical=categorical_idx)
            costs.append(model.cost_)
        return list(k_range), costs

    with st.spinner("Computing Elbow Method (k=2–9) …"):
        elbow_result = compute_elbow(df_unique)

    if elbow_result is not None:
        k_range, costs = elbow_result
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(BG)
        ax.plot(k_range, costs, marker='o', color=MINT, lw=2, markersize=8,
                markerfacecolor=BG, markeredgecolor=MINT, markeredgewidth=2)
        ax.fill_between(k_range, costs, alpha=0.08, color=MINT)

        diffs2 = np.diff(np.diff(costs))
        elbow_k = k_range[np.argmax(diffs2) + 1]
        ax.axvline(elbow_k, color=RED, lw=1.5, linestyle='--',
                   label=f'Recommended k = {elbow_k}')
        ax.scatter([elbow_k], [costs[k_range.index(elbow_k)]],
                   color=RED, zorder=5, s=100)

        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Cost (intra-cluster distance)")
        ax.set_title("Elbow Method for K-Prototypes")
        ax.set_xticks(list(k_range))
        ax.legend()
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
    CLUSTER_PALETTE = [MINT, AMBER, RED, "#A855F7", "#3B82F6", "#F97316", "#10B981", "#EC4899"]
    cluster_colors = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]
                      for i, c in enumerate(cluster_order)}

    st.subheader("Cluster Overview")
    fleet_mean   = df_cluster_raw['CO2 (g/km)'].mean()
    cluster_means = df_cluster_raw.groupby('Cluster', as_index=False)['CO2 (g/km)'].mean()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    ax1, ax2, ax3, ax4 = axes.flatten()

    cnt_data = df_cluster_raw['Cluster'].value_counts().sort_index()
    ax1.bar([str(c) for c in cnt_data.index], cnt_data.values,
            color=[cluster_colors[c] for c in cnt_data.index],
            edgecolor=BG, lw=0, alpha=0.9)
    ax1.set_title("Observations per Cluster"); ax1.set_xlabel("Cluster"); ax1.set_ylabel("Count")

    data_box = [df_cluster_raw[df_cluster_raw['Cluster']==c]['CO2 (g/km)'].dropna().values
                for c in cluster_order]
    bp4 = ax2.boxplot(data_box, patch_artist=True,
                       medianprops=dict(color=TEXT, lw=2),
                       whiskerprops=dict(color=MUTED, lw=0.8),
                       capprops=dict(color=MUTED),
                       flierprops=dict(marker='o', alpha=0.3, markersize=2))
    for i, patch in enumerate(bp4['boxes']):
        c = cluster_order[i]
        patch.set_facecolor(f"{cluster_colors[c]}40"); patch.set_edgecolor(cluster_colors[c])
    ax2.set_xticklabels([f"Cluster {c}" for c in cluster_order])
    ax2.set_title("CO₂ Distribution per Cluster"); ax2.set_ylabel("CO₂ (g/km)")

    for c in cluster_order:
        mask_c = df_cluster_raw['Cluster'] == c
        ax3.scatter(df_cluster_raw[mask_c]['Maximum Power (kW)'],
                    df_cluster_raw[mask_c]['Empty Mass Euro Avg (kg)'],
                    color=cluster_colors[c], alpha=0.5, s=12,
                    label=f"Cluster {c}", rasterized=True)
    ax3.set_title("Power vs Mass"); ax3.set_xlabel("Max Power (kW)"); ax3.set_ylabel("Mass (kg)")
    ax3.legend(markerscale=2, framealpha=0.7)

    ax4.bar([str(c) for c in cluster_means['Cluster']], cluster_means['CO2 (g/km)'],
            color=[cluster_colors[c] for c in cluster_means['Cluster']],
            edgecolor=BG, lw=0, alpha=0.9)
    ax4.axhline(fleet_mean, color=RED, linestyle='--', lw=1.5,
                label=f'Fleet Avg {fleet_mean:.1f} g/km')
    ax4.set_title("Mean CO₂ per Cluster"); ax4.legend()

    plt.subplots_adjust(hspace=0.45, wspace=0.30)
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Cluster Interpretation")
    st.markdown("""
    | Cluster | Size | Avg CO₂ | Profile |
    |---------|------|---------|--------|
    | **0** | ~2,400 (42%) | ~148 g/km | Light mid-range — low mass & power, below fleet average |
    | **1** | ~1,430 (25%) | ~210 g/km | Heavy commercial — high mass, mostly diesel |
    | **2** | ~1,130 (20%) | ~126 g/km | **Efficiency cluster** — lowest CO₂, light petrol vehicles |
    | **3** | ~740 (13%)   | ~243 g/km | High-performance — highest power & mass |

    **Fleet average: 171.3 g/km** — Clusters 0 and 2 are clearly below, Clusters 1 and 3 above.
    """)

    st.markdown("---")
    st.subheader("Categorical Distribution per Cluster")
    cat_cols_clust = [c for c in ['Body', 'Fuel', 'Gearbox'] if c in df_cluster_raw.columns]
    fig, axes = plt.subplots(1, len(cat_cols_clust), figsize=(18, 6))
    fig.patch.set_facecolor(BG)
    if len(cat_cols_clust) == 1: axes = [axes]
    for i, feature in enumerate(cat_cols_clust):
        counts = pd.crosstab(df_cluster_raw['Cluster'], df_cluster_raw[feature])
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        pct.plot(kind='bar', stacked=True, colormap='viridis',
                 ax=axes[i], edgecolor=BG, lw=0.3)
        axes[i].set_title(f'{feature} Distribution')
        axes[i].set_xlabel('Cluster'); axes[i].set_ylabel('Percentage (%)')
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].legend(title=feature, bbox_to_anchor=(1.02, 1),
                       loc='upper left', fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    st.pyplot(fig)
    plt.close()

    st.caption(
        "Cluster 2 is almost exclusively petrol (ES) — light petrol saloons with manual gearbox. "
        "Cluster 1 is nearly entirely diesel (GO) — heavy minibuses and vans dominate this segment."
    )

    st.markdown("---")
    st.subheader("Cluster Profiles — Radar & Heatmap")
    profile_features = ['Maximum Power (kW)', 'Empty Mass Euro Avg (kg)', 'CO2 (g/km)']
    profile_labels   = ['Power', 'Mass', 'CO₂']

    df_profile = df_cluster_raw.copy()
    for f in profile_features:
        mn, mx = df_profile[f].min(), df_profile[f].max()
        df_profile[f] = (df_profile[f] - mn) / (mx - mn + 1e-9)

    cluster_profiles = df_profile.groupby('Cluster')[profile_features].mean()
    cluster_profiles.columns = profile_labels
    cluster_profiles = cluster_profiles.reindex(cluster_order)

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])

    ax_radar = fig.add_subplot(gs[0], projection='polar')
    ax_radar.set_facecolor(BG)
    labels = profile_labels
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    for cid in cluster_profiles.index:
        vals = cluster_profiles.loc[cid].tolist() + [cluster_profiles.loc[cid].tolist()[0]]
        col  = cluster_colors.get(cid, MUTED)
        ax_radar.plot(angles, vals, color=col, lw=2, label=f'Cluster {cid}')
        ax_radar.fill(angles, vals, color=col, alpha=0.10)
    ax_radar.set_theta_offset(np.pi/2); ax_radar.set_theta_direction(-1)
    ax_radar.set_xticks(angles[:-1]); ax_radar.set_xticklabels(labels, fontsize=11, color=TEXT)
    ax_radar.set_ylim(0, 1)
    ax_radar.grid(color=BORDER, lw=0.6)
    ax_radar.set_title("Cluster Profiles (Radar)", fontsize=13, pad=30, color=TEXT)
    ax_radar.legend(loc='upper left', bbox_to_anchor=(1.10, 1.02),
                    frameon=False, fontsize=10)

    ax_heat = fig.add_subplot(gs[1])
    ax_heat.set_facecolor(BG)
    hm_data = cluster_profiles.values
    nr, nc = hm_data.shape
    for i, cid in enumerate(cluster_profiles.index):
        col = cluster_colors.get(cid, MUTED)
        for j in range(nc):
            ax_heat.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                             color=col, alpha=0.3))
            ax_heat.text(j, i, f"{hm_data[i,j]:.2f}",
                         ha='center', va='center', fontsize=11,
                         color=TEXT, fontweight='600')
    ax_heat.set_xticks(np.arange(nc)); ax_heat.set_xticklabels(profile_labels, color=TEXT)
    ax_heat.set_yticks(np.arange(nr))
    ax_heat.set_yticklabels([f'Cluster {c}' for c in cluster_profiles.index], color=TEXT)
    ax_heat.set_xlim(-0.5, nc-0.5); ax_heat.set_ylim(-0.5, nr-0.5)
    ax_heat.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax_heat.grid(which='minor', color=BORDER, lw=1)
    ax_heat.tick_params(which='minor', bottom=False, left=False)
    ax_heat.set_title("Cluster Profiles (Heatmap)", fontsize=13, color=TEXT)
    for sp in ['top','right','left','bottom']: ax_heat.spines[sp].set_color(BORDER)

    fig.suptitle("Vehicle Cluster Profiles Dashboard", fontsize=15, color=TEXT)
    st.pyplot(fig)
    plt.close()

    st.caption(
        "Normalised values (0 = minimum, 1 = maximum of dataset). "
        "Cluster 3: highest across all three dimensions — high-performance segment. "
        "Cluster 2: lowest values — efficiency cluster. "
        "Cluster 1: unusually high Mass at moderate power — heavy commercial/diesel segment."
    )


# ═══════════════════════ TAB 5 – PREDICTION ══════════════════════════════════
with tabs[5]:
    st.header("Predictive Modeling")

    st.markdown("""
    > **Research Question:** What is the relative contribution of vehicle mass, engine power,
    > fuel type, body style and gearbox type in explaining CO₂ emissions, and which minimal
    > feature set achieves the best predictive performance?
    """)

    with st.spinner("Training models (Feature Sets + CV + 5 algorithms) …"):
        try:
            (fitted, results_df, fs_df, best_fs, feature_cols,
             X_train, X_test, y_train, y_test,
             rf_pipe, fi_df, num_f, cat_f) = train_all_models(df_unique)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    best_model_name = "Random Forest"

    # ── 1. Feature Set Comparison ────────────────────────────────────────────
    st.subheader("① Feature Set Comparison — 5-Fold CV, Random Forest")
    st.markdown(
        "Four feature combinations compared using **5-fold cross-validation** with a Random Forest. "
        "MAE = average deviation in g/km (lower = better)."
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    plot_fs = fs_df.sort_values("CV_MAE_mean", ascending=True)
    colors_fs = [MINT if i==0 else f"{MINT}50" for i in range(len(plot_fs))]
    ax.barh(plot_fs["Feature_Set"], plot_fs["CV_MAE_mean"],
            color=colors_fs, edgecolor=BG, lw=0)
    ax.set_xlabel("CV MAE (lower = better)")
    ax.set_title("Feature Set Comparison (5-Fold CV, Random Forest)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ Best Feature Set: **{best_fs}** — Features: {', '.join(feature_cols)}")
    num_cols_fs = fs_df.select_dtypes(include='number').columns.tolist()
    st.dataframe(
        fs_df[["Feature_Set","Features","CV_MAE_mean","CV_MAE_std"]]
        .style.format({"CV_MAE_mean": "{:.2f}", "CV_MAE_std": "{:.2f}"}),
        use_container_width=True
    )

    st.markdown("---")

    # ── 2. Model Comparison ──────────────────────────────────────────────────
    st.subheader(f"② Model Comparison — R² & MAE ({best_fs})")
    st.markdown(
        "Five models evaluated on 80/20 train/test split. "
        "**R²** = proportion of explained variance (1.0 = perfect). "
        "**MAE** = average deviation in g/km. "
        "Large gap between train and test = overfitting."
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    plot_r = results_df.sort_values("Test_R2", ascending=True)

    def _hbar(ax, labels, values, title, xlabel):
        bar_c = [MINT if v == max(values) else f"{MINT}55" for v in values]
        ax.barh(labels, values, color=bar_c, edgecolor=BG, lw=0)
        ax.set_title(title); ax.set_xlabel(xlabel)

    _hbar(axes[0], plot_r["Model"], plot_r["Test_R2"],
          f"Test R² — {best_fs}", "Test R²")
    _hbar(axes[1], plot_r["Model"], plot_r["Test_MAE"],
          f"Test MAE — {best_fs}", "Test MAE (g/km)")

    fig.suptitle(f"Regression Model Performance — {best_fs}", fontsize=14, color=TEXT)
    fig.tight_layout(rect=[0,0,1,0.95])
    st.pyplot(fig)
    plt.close()

    num_cols_res = results_df.select_dtypes(include='number').columns.tolist()
    st.dataframe(
        results_df.style
        .highlight_max(subset=["Test_R2"],  color="#00C8A025")
        .highlight_min(subset=["Test_MAE"], color="#00C8A025")
        .format({c: "{:.4f}" for c in num_cols_res}),
        use_container_width=True
    )
    st.caption(
        "Gradient Boosting and Random Forest achieve R²≈0.95 at ~7–8 g/km MAE. "
        "Linear models plateau at R²≈0.86 — they cannot capture non-linear relationships."
    )

    st.markdown("---")

    # ── 3. Feature Importance ────────────────────────────────────────────────
    st.subheader(f"③ Feature Importance — Random Forest ({best_fs})")
    top15 = fi_df.head(15).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    bar_clr = [MINT if i >= len(top15)-3 else f"{MINT}60" for i in range(len(top15))]
    ax.barh(top15["Feature"], top15["Importance"], color=bar_clr, edgecolor=BG, lw=0)
    ax.set_xlabel("Importance (Mean Decrease Impurity)")
    ax.set_title(f"Top 15 Feature Importances — Random Forest ({best_fs})")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.dataframe(fi_df.head(15), use_container_width=True)
    st.caption(
        "**Empty mass (46.9%)** and **engine power (37.2%)** together account for ~84% of explained variance. "
        "Gear count (4.6%) and fuel type (~2.6% each) provide additional signal. "
        "Body style and gearbox type play a minor role (<1% per feature)."
    )

    st.markdown("---")

    # ── 4. Partial Dependence Plots ──────────────────────────────────────────
    st.subheader("④ Partial Dependence Plots — Random Forest")
    st.markdown(
        "PDPs show the **marginal effect** of a single feature on predicted CO₂ — "
        "all other features held at their mean (ceteris paribus)."
    )
    X_train_pdp = X_train.copy()
    if "GearCount" in X_train_pdp.columns:
        X_train_pdp["GearCount"] = X_train_pdp["GearCount"].astype(float)

    pdp_features = [f for f in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
                                  "GearType", "GearCount"] if f in feature_cols]
    try:
        fig, ax = plt.subplots(figsize=(16, 6))
        fig.patch.set_facecolor(BG)
        PartialDependenceDisplay.from_estimator(
            rf_pipe, X_train_pdp, features=pdp_features,
            categorical_features=[f for f in cat_f if f in pdp_features], ax=ax)
        for axis in fig.axes:
            axis.set_facecolor(BG)
            for line in axis.get_lines():
                line.set_color(MINT)
        fig.suptitle(f"Partial Dependence Plots — Random Forest ({best_fs})", fontsize=14, color=TEXT)
        fig.tight_layout(rect=[0,0,1,0.95])
        st.pyplot(fig)
        plt.close()
        st.caption(
            "**Empty Mass:** diminishing returns above ~1,600 kg. "
            "**Maximum Power:** increasing returns above 150 kW. "
            "**GearCount:** slight negative effect — more gears → marginally lower CO₂."
        )
    except Exception as e:
        st.warning(f"PDP not available: {e}")

    # ── 5. What-If Simulator ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⑤ What-If Simulator — Live Model Interaction")
    st.markdown(
        "Adjust vehicle features using the sliders. "
        "All other features are held at the **dataset median / mode** "
        "(the 'typical' vehicle in this dataset)."
    )

    base_mass  = float(df_unique["Empty Mass Euro Avg (kg)"].median())
    base_power = float(df_unique["Maximum Power (kW)"].median())
    base_gears = float(df_unique["GearCount"].median()) if "GearCount" in df_unique.columns else 6.0
    base_fuel  = str(df_unique["Fuel"].mode().iloc[0]) if "Fuel" in df_unique.columns else "GO"
    base_body  = str(df_unique["Body"].mode().iloc[0]) if "Body" in df_unique.columns else "BERLINE"
    base_gtype = str(df_unique["GearType"].mode().iloc[0]) if "GearType" in df_unique.columns else "Manual"

    base_row = {}
    for f in feature_cols:
        if f == "Empty Mass Euro Avg (kg)": base_row[f] = base_mass
        elif f == "Maximum Power (kW)":     base_row[f] = base_power
        elif f == "GearCount":              base_row[f] = base_gears
        elif f == "Fuel":                   base_row[f] = base_fuel
        elif f == "Body":                   base_row[f] = base_body
        elif f == "GearType":               base_row[f] = base_gtype
        else:                               base_row[f] = 0

    base_pred = float(fitted[best_model_name].predict(pd.DataFrame([base_row]))[0])

    st.markdown(
        f"<div style='background:#1A1D27;border:1px solid #2A2D3A;border-left:4px solid #00C8A0;"
        f"border-radius:8px;padding:.75rem 1rem;font-size:.84rem;color:#7B8094;margin-bottom:1rem'>"
        f"Base vehicle: <span style='color:#E8EAF0'>{base_mass:.0f} kg · {base_power:.0f} kW "
        f"({base_power*1.36:.0f} HP) · {base_fuel} · {base_body} · {base_gtype} · {base_gears:.0f} gears</span>"
        f" → Base CO₂: <span style='color:#00C8A0;font-weight:700'>{base_pred:.1f} g/km</span></div>",
        unsafe_allow_html=True
    )

    col_sliders, col_result = st.columns([2, 1])

    with col_sliders:
        sim_mass = st.slider(
            "Empty Mass (kg)", min_value=800, max_value=3200,
            value=int(base_mass), step=50, help=f"Dataset median: {base_mass:.0f} kg"
        )
        sim_power = st.slider(
            "Maximum Power (kW)", min_value=40, max_value=560,
            value=int(base_power), step=5, help=f"Dataset median: {base_power:.0f} kW"
        )
        if "GearCount" in feature_cols:
            sim_gears = st.slider("Number of Gears", min_value=4, max_value=8,
                                   value=int(base_gears), step=1)
        else:
            sim_gears = base_gears

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            sim_fuel = st.selectbox("Fuel", ["GO", "ES"],
                                     index=0 if base_fuel == "GO" else 1)
        with col_b:
            sim_gtype = st.selectbox(
                "Gearbox Type", ["Manual", "Automatic", "CVT", "DCT"],
                index=["Manual","Automatic","CVT","DCT"].index(base_gtype)
                      if base_gtype in ["Manual","Automatic","CVT","DCT"] else 0
            )
        with col_c:
            body_opts = sorted(df_unique["Body"].dropna().unique().tolist()) \
                        if "Body" in df_unique.columns else ["BERLINE"]
            sim_body = st.selectbox("Body Style", body_opts,
                                     index=body_opts.index(base_body) if base_body in body_opts else 0)

    sim_row = {}
    for f in feature_cols:
        if f == "Empty Mass Euro Avg (kg)": sim_row[f] = float(sim_mass)
        elif f == "Maximum Power (kW)":     sim_row[f] = float(sim_power)
        elif f == "GearCount":              sim_row[f] = float(sim_gears)
        elif f == "Fuel":                   sim_row[f] = sim_fuel
        elif f == "Body":                   sim_row[f] = sim_body
        elif f == "GearType":               sim_row[f] = sim_gtype
        else:                               sim_row[f] = 0

    sim_pred  = float(fitted[best_model_name].predict(pd.DataFrame([sim_row]))[0])
    delta_co2 = sim_pred - base_pred
    delta_mass  = sim_mass  - base_mass
    delta_power = sim_power - base_power

    euro_sim = ("A (≤100)"   if sim_pred <= 100 else
                "B (101–120)" if sim_pred <= 120 else
                "C (121–140)" if sim_pred <= 140 else
                "D (141–160)" if sim_pred <= 160 else
                "E (161–200)" if sim_pred <= 200 else "F/G (>200)")
    box_color   = MINT if sim_pred <= 120 else AMBER if sim_pred <= 160 else RED
    delta_col   = RED  if delta_co2 > 0 else MINT if delta_co2 < 0 else MUTED

    with col_result:
        st.markdown(
            f"<div style='background:#1A1D27;border:1px solid #2A2D3A;"
            f"border-top:4px solid {box_color};border-radius:10px;"
            f"padding:1.5rem;text-align:center;'>"
            f"<div style='font-size:.7rem;font-weight:600;color:#7B8094;"
            f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem'>Predicted CO₂</div>"
            f"<div style='font-size:2.6rem;font-weight:900;color:{box_color};"
            f"font-family:monospace;letter-spacing:-0.03em'>{sim_pred:.1f}</div>"
            f"<div style='font-size:.8rem;color:#7B8094;margin-bottom:.75rem'>g/km · Class {euro_sim}</div>"
            f"<div style='font-size:1.1rem;font-weight:700;color:{delta_col}'>"
            f"{'▲' if delta_co2 > 0 else '▼' if delta_co2 < 0 else '='} "
            f"{delta_co2:+.1f} g/km vs base</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown(" ")
        st.metric("Mass change",  f"{delta_mass:+.0f} kg")
        st.metric("Power change", f"{delta_power:+.0f} kW ({delta_power*1.36:+.0f} HP)")
        st.metric("CO₂ change",   f"{delta_co2:+.1f} g/km", delta_color="inverse")

    st.markdown("---")
    st.markdown("##### Sensitivity: CO₂ as each feature varies (all others fixed)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor(BG)

    mass_range = np.arange(800, 3300, 100)
    mass_preds = []
    for m in mass_range:
        r = sim_row.copy(); r["Empty Mass Euro Avg (kg)"] = float(m)
        mass_preds.append(float(fitted[best_model_name].predict(pd.DataFrame([r]))[0]))

    axes[0].plot(mass_range, mass_preds, color=MINT, lw=2)
    axes[0].fill_between(mass_range, mass_preds, alpha=0.06, color=MINT)
    axes[0].axvline(sim_mass, color=RED, lw=1.5, linestyle='--',
                    label=f"Current: {sim_mass} kg → {sim_pred:.0f} g/km")
    axes[0].axvline(base_mass, color=MUTED, lw=1, linestyle=':',
                    label=f"Base: {base_mass:.0f} kg")
    axes[0].set_xlabel("Empty Mass (kg)"); axes[0].set_ylabel("Predicted CO₂ (g/km)")
    axes[0].set_title("CO₂ vs. Mass")
    axes[0].legend(fontsize=8)

    power_range = np.arange(40, 570, 10)
    power_preds = []
    for p in power_range:
        r = sim_row.copy(); r["Maximum Power (kW)"] = float(p)
        power_preds.append(float(fitted[best_model_name].predict(pd.DataFrame([r]))[0]))

    axes[1].plot(power_range, power_preds, color=AMBER, lw=2)
    axes[1].fill_between(power_range, power_preds, alpha=0.06, color=AMBER)
    axes[1].axvline(sim_power, color=RED, lw=1.5, linestyle='--',
                    label=f"Current: {sim_power} kW → {sim_pred:.0f} g/km")
    axes[1].axvline(base_power, color=MUTED, lw=1, linestyle=':',
                    label=f"Base: {base_power:.0f} kW")
    axes[1].set_xlabel("Maximum Power (kW)"); axes[1].set_ylabel("Predicted CO₂ (g/km)")
    axes[1].set_title("CO₂ vs. Power")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        f"Model: Random Forest · Feature set: {best_fs} · "
        f"Test MAE ≈ {results_df[results_df['Model']=='Random Forest']['Test_MAE'].iloc[0]:.1f} g/km"
    )

    # ── 6. Brand Comparison ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⑥ Brand Comparison — Real Data vs. Simulator Settings")
    st.markdown(
        "Select up to 5 brands. For each brand, the real vehicles matching the current "
        "**Fuel · Body · Power range** filter are pulled from the dataset and compared — "
        "actual measured CO₂ median alongside the model prediction for each brand's "
        "typical vehicle in this segment."
    )

    # ── Derive power bracket from sim_power ──────────────────────────────────
    if   sim_power <= 55:  kw_lo, kw_hi = 0,   55
    elif sim_power <= 96:  kw_lo, kw_hi = 56,  96
    elif sim_power <= 147: kw_lo, kw_hi = 97,  147
    else:                  kw_lo, kw_hi = 148, 600

    mask_seg = (
        df_unique["Fuel"].eq(sim_fuel) &
        df_unique["Body"].eq(sim_body) &
        df_unique["Maximum Power (kW)"].between(kw_lo, kw_hi)
    )
    df_seg = df_unique[mask_seg].copy()

    # Fallback: relax power bracket if too few results
    if len(df_seg) < 10:
        df_seg = df_unique[df_unique["Fuel"].eq(sim_fuel) & df_unique["Body"].eq(sim_body)].copy()

    available_brands = sorted(
        df_seg.groupby("Brand")["CO2 (g/km)"].count()
        .where(lambda x: x >= 2).dropna().index.tolist()
    )

    if not available_brands:
        st.info("No brands with ≥2 models found for the current filter. Adjust Fuel or Body Style.")
    else:
        # Default: top 4 brands by model count in segment
        default_brands = (
            df_seg.groupby("Brand")["CO2 (g/km)"].count()
            .sort_values(ascending=False)
            .head(4).index.tolist()
        )
        default_sel = [b for b in default_brands if b in available_brands]

        selected_brands = st.multiselect(
            "Select brands to compare",
            options=available_brands,
            default=default_sel,
            max_selections=5,
            help="Max 5 brands. Only brands with ≥2 models in the current segment are shown."
        )

        if selected_brands:
            BRAND_PALETTE = [MINT, AMBER, RED, "#A855F7", "#3B82F6"]
            brand_colors  = {b: BRAND_PALETTE[i] for i, b in enumerate(selected_brands)}

            # ── Per-brand stats from real data ────────────────────────────────
            brand_rows = []
            for brand in selected_brands:
                df_b = df_seg[df_seg["Brand"] == brand]["CO2 (g/km)"].dropna()
                if len(df_b) == 0:
                    continue
                # Typical vehicle for this brand in segment (median mass/power/gears)
                df_bv = df_seg[df_seg["Brand"] == brand].copy()
                b_mass  = float(df_bv["Empty Mass Euro Avg (kg)"].median())
                b_power = float(df_bv["Maximum Power (kW)"].median())
                b_gears = float(df_bv["GearCount"].median()) if "GearCount" in df_bv.columns else sim_gears
                b_gtype = str(df_bv["GearType"].mode().iloc[0]) if "GearType" in df_bv.columns and len(df_bv) > 0 else sim_gtype

                # Model prediction for this brand's typical vehicle
                b_row = sim_row.copy()
                b_row["Empty Mass Euro Avg (kg)"] = b_mass
                b_row["Maximum Power (kW)"]       = b_power
                if "GearCount" in b_row: b_row["GearCount"] = b_gears
                if "GearType"  in b_row: b_row["GearType"]  = b_gtype
                b_pred = float(fitted[best_model_name].predict(pd.DataFrame([b_row]))[0])

                brand_rows.append({
                    "Brand":        brand,
                    "N_models":     len(df_b),
                    "Real_Median":  df_b.median(),
                    "Real_P25":     df_b.quantile(0.25),
                    "Real_P75":     df_b.quantile(0.75),
                    "Real_Min":     df_b.min(),
                    "Real_Max":     df_b.max(),
                    "Model_Pred":   b_pred,
                    "Typical_Mass": b_mass,
                    "Typical_kW":   b_power,
                })

            if not brand_rows:
                st.warning("No data available for the selected brands in this segment.")
            else:
                brand_df = pd.DataFrame(brand_rows).sort_values("Real_Median")

                # ── Metric cards ──────────────────────────────────────────────
                n_cols = len(brand_df)
                metric_cols = st.columns(n_cols)
                best_brand = brand_df.iloc[0]["Brand"]
                for col_ui, (_, row) in zip(metric_cols, brand_df.iterrows()):
                    col_ui.markdown(
                        f"<div style='background:#1A1D27;border:1px solid #2A2D3A;"
                        f"border-top:4px solid {brand_colors[row['Brand']]};"
                        f"border-radius:10px;padding:1rem;text-align:center;'>"
                        f"<div style='font-size:.68rem;font-weight:600;color:#7B8094;"
                        f"text-transform:uppercase;letter-spacing:.09em;margin-bottom:.3rem'>"
                        f"{row['Brand']}</div>"
                        f"<div style='font-size:1.8rem;font-weight:900;"
                        f"color:{brand_colors[row['Brand']]};font-family:monospace'>"
                        f"{row['Real_Median']:.0f}</div>"
                        f"<div style='font-size:.72rem;color:#7B8094'>g/km median · {int(row['N_models'])} models</div>"
                        f"<div style='font-size:.75rem;color:#7B8094;margin-top:.35rem'>"
                        f"Predicted: <span style='color:#E8EAF0;font-weight:600'>"
                        f"{row['Model_Pred']:.0f} g/km</span></div>"
                        f"{'<div style=\"font-size:.68rem;color:#00C8A0;font-weight:600;margin-top:.2rem\">★ Most efficient</div>' if row['Brand'] == best_brand else ''}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown(" ")

                # ── Chart 1: Real median + IQR bars with model prediction dots ─
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                fig.patch.set_facecolor(BG)

                x_pos   = np.arange(len(brand_df))
                brands  = brand_df["Brand"].tolist()
                clr_map = [brand_colors[b] for b in brands]

                # Bar = real median, error bars = IQR
                lower_err = brand_df["Real_Median"] - brand_df["Real_P25"]
                upper_err = brand_df["Real_P75"]    - brand_df["Real_Median"]
                axes[0].bar(x_pos, brand_df["Real_Median"], color=clr_map,
                            alpha=0.7, edgecolor=BG, lw=0, label="Real Median CO₂")
                axes[0].errorbar(x_pos, brand_df["Real_Median"],
                                  yerr=[lower_err, upper_err],
                                  fmt='none', color=TEXT, capsize=5, lw=1.5, alpha=0.6)
                axes[0].scatter(x_pos, brand_df["Model_Pred"],
                                color=TEXT, s=60, zorder=5, marker='D',
                                label="Model Prediction")
                axes[0].axhline(sim_pred, color=MUTED, lw=1.2, linestyle='--',
                                 label=f"Sim. setting: {sim_pred:.0f} g/km")
                axes[0].set_xticks(x_pos)
                axes[0].set_xticklabels(brands, rotation=20, ha='right')
                axes[0].set_ylabel("CO₂ (g/km)")
                axes[0].set_title("Real Median CO₂ + IQR  ◆ Model Prediction")
                axes[0].legend(fontsize=8)

                # Annotate bars with value
                for xi, (_, row) in zip(x_pos, brand_df.iterrows()):
                    axes[0].text(xi, row["Real_Median"] + upper_err.iloc[list(brand_df.index).index(_)] + 1,
                                  f"{row['Real_Median']:.0f}", ha='center', va='bottom',
                                  fontsize=9, color=TEXT, fontweight='600')

                # Chart 2: Box plots side by side
                box_data   = [df_seg[df_seg["Brand"] == b]["CO2 (g/km)"].dropna().values
                               for b in brands]
                bp5 = axes[1].boxplot(box_data, patch_artist=True,
                                       medianprops=dict(color=TEXT, lw=2),
                                       whiskerprops=dict(color=MUTED, lw=0.8),
                                       capprops=dict(color=MUTED),
                                       flierprops=dict(marker='o', alpha=0.35, markersize=3))
                for i, patch in enumerate(bp5['boxes']):
                    c = clr_map[i]
                    patch.set_facecolor(f"{c}40"); patch.set_edgecolor(c)
                    for fp in bp5['flierprops'] if isinstance(bp5['flierprops'], list) else []:
                        fp.set_markeredgecolor(c)
                axes[1].set_xticks(np.arange(1, len(brands)+1))
                axes[1].set_xticklabels(brands, rotation=20, ha='right')
                axes[1].axhline(sim_pred, color=MUTED, lw=1.2, linestyle='--',
                                 label=f"Sim. setting: {sim_pred:.0f} g/km")
                axes[1].set_ylabel("CO₂ (g/km)")
                axes[1].set_title("Full CO₂ Distribution per Brand")
                axes[1].legend(fontsize=8)

                filter_lbl = f"{sim_fuel} · {sim_body} · {kw_lo}–{kw_hi} kW"
                fig.suptitle(f"Brand Comparison — {filter_lbl}", fontsize=13, color=TEXT)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                st.pyplot(fig)
                plt.close()

                # ── Power sensitivity overlaid per brand ──────────────────────
                st.markdown("##### Power Sensitivity per Brand")
                st.caption(
                    "How does each brand's predicted CO₂ change as power increases? "
                    "All other features fixed at that brand's typical segment values."
                )
                fig, ax = plt.subplots(figsize=(14, 4))
                fig.patch.set_facecolor(BG)

                pw_range = np.arange(40, 570, 10)
                for _, row in brand_df.iterrows():
                    b = row["Brand"]
                    b_row_base = sim_row.copy()
                    b_row_base["Empty Mass Euro Avg (kg)"] = row["Typical_Mass"]
                    if "GearType" in b_row_base:
                        b_row_base["GearType"] = sim_gtype
                    pw_preds = []
                    for p in pw_range:
                        r2 = b_row_base.copy(); r2["Maximum Power (kW)"] = float(p)
                        pw_preds.append(float(fitted[best_model_name].predict(pd.DataFrame([r2]))[0]))
                    ax.plot(pw_range, pw_preds, color=brand_colors[b], lw=2, label=b)
                    # Mark brand's typical power
                    ax.scatter([row["Typical_kW"]], [row["Model_Pred"]],
                                color=brand_colors[b], s=70, zorder=5)

                ax.set_xlabel("Maximum Power (kW)")
                ax.set_ylabel("Predicted CO₂ (g/km)")
                ax.set_title("CO₂ vs. Power — per Brand (at brand's typical mass)")
                ax.legend(fontsize=9, framealpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # ── Summary table ─────────────────────────────────────────────
                with st.expander("📋 Full comparison table"):
                    disp_df = brand_df[[
                        "Brand", "N_models", "Real_Median", "Real_P25", "Real_P75",
                        "Real_Min", "Real_Max", "Model_Pred", "Typical_Mass", "Typical_kW"
                    ]].copy()
                    disp_df.columns = [
                        "Brand", "Models", "Median CO₂", "P25", "P75",
                        "Min", "Max", "Model Pred.", "Typical Mass (kg)", "Typical kW"
                    ]
                    st.dataframe(
                        disp_df.style.format({
                            c: "{:.1f}" for c in
                            ["Median CO₂","P25","P75","Min","Max","Model Pred.",
                             "Typical Mass (kg)","Typical kW"]
                        }).highlight_min(subset=["Median CO₂"], color="#00C8A025")
                         .highlight_max(subset=["Median CO₂"], color="#E8485515"),
                        use_container_width=True
                    )


# ═══════════════════════ TAB 6 – CO₂-RECHNER ═════════════════════════════════
with tabs[6]:
    st.header("CO₂ Calculator & Brand Comparison")
    st.markdown(
        "Choose your vehicle by **everyday criteria** — the app shows the **real CO₂ median** "
        "from comparable vehicles in the ADEME dataset and which brand is most efficient in your segment."
    )

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
        "Up to 75 HP (≤55 kW)":  (0,   55),
        "76–130 HP (56–96 kW)":   (56,  96),
        "131–200 HP (97–147 kW)": (97,  147),
        "Over 200 HP (>147 kW)":  (148, 600),
    }

    FUEL_MAP = {"Petrol": "ES", "Diesel": "GO"}
    GEAR_MAP = {"Manual": "M", "Automatic": "A"}

    with st.form("co2_consumer_form"):
        st.subheader("Configure Your Vehicle")
        col1, col2 = st.columns(2)
        with col1:
            body_sel = st.selectbox("Body Style", list(BODY_MAP.keys()), index=0)
            antrieb  = st.radio("Fuel Type", ["Petrol", "Diesel"], horizontal=True)
            getriebe = st.radio("Gearbox",   ["Manual", "Automatic"], horizontal=True)
        with col2:
            power_sel = st.select_slider(
                "Engine Power", options=list(POWER_MAP.keys()),
                value="76–130 HP (56–96 kW)"
            )
            st.markdown("""
            <div style='font-size:.7rem;font-weight:600;color:#7B8094;
                        text-transform:uppercase;letter-spacing:.09em;margin-bottom:.4rem'>
            Segment Reference
            </div>
            """, unsafe_allow_html=True)
            for body, (co2_ref, n_ref) in BODY_INFO.items():
                bullet = "▶ " if body == body_sel else "  "
                color  = "#00C8A0" if body == body_sel else "#7B8094"
                st.markdown(
                    f"<div style='font-size:.76rem;color:{color};line-height:1.7'>"
                    f"{bullet}<b>{body}</b>: {co2_ref} median · {n_ref}</div>",
                    unsafe_allow_html=True
                )

        submitted = st.form_submit_button("Calculate CO₂ & Compare Brands →", use_container_width=True)

    if submitted:
        body_val    = BODY_MAP[body_sel]
        kw_range    = POWER_MAP[power_sel]
        fuel_val    = FUEL_MAP[antrieb]
        gear_prefix = GEAR_MAP[getriebe]
        ps_lo       = round(kw_range[0] * 1.36)
        ps_hi       = round(kw_range[1] * 1.36) if kw_range[1] < 600 else None

        mask = (
            df_unique["Body"].eq(body_val) &
            df_unique["Fuel"].eq(fuel_val) &
            df_unique["Maximum Power (kW)"].between(kw_range[0], kw_range[1]) &
            df_unique["Gearbox"].astype(str).str.startswith(gear_prefix)
        )
        df_match = df_unique[mask].copy()

        used_gear_filter = True
        if len(df_match) < 5:
            mask2 = (df_unique["Body"].eq(body_val) &
                     df_unique["Fuel"].eq(fuel_val) &
                     df_unique["Maximum Power (kW)"].between(kw_range[0], kw_range[1]))
            df_match = df_unique[mask2].copy()
            used_gear_filter = False

        used_power_filter = True
        if len(df_match) < 3:
            mask3 = df_unique["Body"].eq(body_val) & df_unique["Fuel"].eq(fuel_val)
            df_match = df_unique[mask3].copy()
            used_gear_filter  = False
            used_power_filter = False

        co2_vals = df_match["CO2 (g/km)"].dropna()
        if len(co2_vals) == 0:
            st.error("No vehicles found. Please try a different configuration.")
            st.stop()

        co2_median   = co2_vals.median()
        co2_p25      = co2_vals.quantile(0.25)
        co2_p75      = co2_vals.quantile(0.75)
        co2_min      = co2_vals.min()
        co2_max      = co2_vals.max()
        n_match      = len(df_match)
        n_brands     = df_match["Brand"].nunique()
        fleet_median = df_unique["CO2 (g/km)"].median()
        pct_better   = (df_unique["CO2 (g/km)"] > co2_median).mean() * 100
        delta_fleet  = co2_median - fleet_median
        jahres_co2   = co2_median * 15000 / 1000

        euro  = ("A (≤100 g/km)" if co2_median <= 100 else
                 "B (101–120)"   if co2_median <= 120 else
                 "C (121–140)"   if co2_median <= 140 else
                 "D (141–160)"   if co2_median <= 160 else
                 "E (161–200)"   if co2_median <= 200 else "F/G (>200)")
        color = MINT if co2_median <= 120 else AMBER if co2_median <= 160 else RED

        st.markdown("---")
        st.subheader("Result")

        ps_str = f"{ps_lo}–{ps_hi} HP" if ps_hi else f"{ps_lo}+ HP"
        filter_info = f"{body_sel} · {antrieb} · {ps_str}"
        if used_gear_filter:     filter_info += f" · {getriebe}"
        else:                    filter_info += " · all gearbox types"
        if not used_power_filter: st.caption(f"Power filter broadened — showing all {body_sel} {antrieb} vehicles.")
        elif not used_gear_filter: st.caption("Gearbox filter broadened (too few matches).")

        st.markdown(
            f"<div style='background:#1A1D27;border:1px solid #2A2D3A;"
            f"border-left:6px solid {color};border-radius:10px;padding:1.5rem 2rem;margin:.5rem 0'>"
            f"<div style='font-size:.7rem;font-weight:600;color:#7B8094;"
            f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem'>{filter_info}</div>"
            f"<div style='display:flex;align-items:baseline;gap:.6rem'>"
            f"<span style='font-size:3rem;font-weight:900;color:{color};"
            f"font-family:monospace;letter-spacing:-0.04em'>{co2_median:.0f}</span>"
            f"<span style='font-size:1.1rem;color:#7B8094;font-weight:500'>g CO₂/km</span>"
            f"<span style='font-size:.85rem;color:{color};font-weight:600;"
            f"background:{color}20;border-radius:4px;padding:.15rem .5rem'>{euro}</span>"
            f"</div>"
            f"<div style='font-size:.84rem;color:#7B8094;margin-top:.4rem'>"
            f"Annual CO₂: <span style='color:#E8EAF0;font-weight:600'>{jahres_co2:.0f} kg</span> "
            f"(at 15,000 km/year) · Median from "
            f"<span style='color:#E8EAF0;font-weight:600'>{n_match}</span> real vehicles · "
            f"<span style='color:#E8EAF0;font-weight:600'>{n_brands}</span> brands · "
            f"Range: {co2_min:.0f}–{co2_max:.0f} g/km"
            f"</div></div>",
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median CO₂",    f"{co2_median:.0f} g/km")
        c2.metric("IQR (25–75%)",  f"{co2_p25:.0f}–{co2_p75:.0f} g/km")
        c3.metric("vs. Fleet",     f"{delta_fleet:+.0f} g/km", delta_color="inverse")
        c4.metric("Better than",   f"{pct_better:.0f}% of fleet")

        fig, ax = plt.subplots(figsize=(10, 3))
        fig.patch.set_facecolor(BG)
        ax.hist(df_unique["CO2 (g/km)"].dropna(), bins=60,
                color=BORDER, alpha=0.8, label="All vehicles")
        ax.hist(co2_vals, bins=30, color=color, alpha=0.8,
                label=f"{body_sel} · {antrieb} ({n_match} vehicles)")
        ax.axvline(co2_median,   color=color, lw=2.5, linestyle='--',
                   label=f"Segment median: {co2_median:.0f} g/km")
        ax.axvline(fleet_median, color=MUTED, lw=1.5, linestyle=':',
                   label=f"Fleet median: {fleet_median:.0f} g/km")
        ax.set_xlabel("CO₂ (g/km)"); ax.set_ylabel("Frequency")
        ax.set_title("Your Segment vs. Full Fleet")
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

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
        if (brand_summary["Modelle"] >= 2).sum() >= 3:
            brand_summary = brand_summary[brand_summary["Modelle"] >= 2]

        top_n  = min(15, len(brand_summary))
        plot_b = brand_summary.head(top_n)

        bar_colors_brand = []
        for rank in range(len(plot_b)):
            if rank < 3:          bar_colors_brand.append(MINT)
            elif rank >= top_n-3: bar_colors_brand.append(RED)
            else:                 bar_colors_brand.append(f"{MUTED}80")

        fig, ax = plt.subplots(figsize=(11, max(5, top_n * 0.55)))
        fig.patch.set_facecolor(BG)
        bars = ax.barh(plot_b["Brand"], plot_b["CO2_Median"],
                       color=bar_colors_brand, alpha=0.9, edgecolor=BG, lw=0)

        for bar, val, n in zip(bars, plot_b["CO2_Median"], plot_b["Modelle"]):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.0f} g/km  ({int(n)} models)",
                    va='center', fontsize=9, color=MUTED)

        ax.axvline(co2_median, color=TEXT, lw=1.5, linestyle='--',
                   label=f"Segment median: {co2_median:.0f} g/km")
        ax.set_xlabel("Median CO₂ (g/km)")
        ax.set_title(f"Brand Comparison: {body_sel} · {antrieb} · {getriebe} · {power_sel}",
                     fontsize=12, color=TEXT)
        ax.set_xlim(0, plot_b["CO2_Median"].max() * 1.28)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor=MINT,  label="Top 3 most efficient"),
            Patch(facecolor=f"{MUTED}80", label="Mid-field"),
            Patch(facecolor=RED,   label="Top 3 highest CO₂"),
            plt.Line2D([0],[0], color=TEXT, lw=1.5, linestyle='--',
                       label=f"Segment median: {co2_median:.0f} g/km"),
        ]
        ax.legend(handles=legend_els, loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("🏆 Top 3 Recommendations")
        top3  = brand_summary.head(3)
        cols3 = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for idx, (col_ui, (_, rb)) in enumerate(zip(cols3, top3.iterrows())):
            saving     = co2_median - rb["CO2_Median"]
            saving_kg  = saving * 15000 / 1000
            saving_str = (f"↓ {saving_kg:.0f} kg CO₂/year less"
                          if saving > 1 else "At segment median")
            col_ui.markdown(
                f"<div style='background:#1A1D27;border:1px solid #2A2D3A;"
                f"border-top:4px solid #00C8A0;border-radius:10px;"
                f"padding:1.25rem;text-align:center;'>"
                f"<div style='font-size:1.8rem'>{medals[idx]}</div>"
                f"<div style='font-size:1rem;font-weight:700;color:#E8EAF0;"
                f"margin:.3rem 0'>{rb['Brand']}</div>"
                f"<div style='font-size:1.6rem;font-weight:900;color:#00C8A0;"
                f"font-family:monospace'>{rb['CO2_Median']:.0f}</div>"
                f"<div style='font-size:.7rem;color:#7B8094'>g/km median</div>"
                f"<div style='font-size:.75rem;color:#7B8094;margin-top:.5rem'>"
                f"Range: {rb['CO2_Min']:.0f}–{rb['CO2_Max']:.0f} g/km · {int(rb['Modelle'])} models<br>"
                f"<span style='color:#00C8A0;font-weight:600'>{saving_str}</span>"
                f"</div></div>",
                unsafe_allow_html=True
            )

        st.markdown("---")
        with st.expander("📋 Show all matching vehicles"):
            show_cols = [c for c in
                         ["Brand","Folder Model","Fuel","Body","Gearbox",
                          "Maximum Power (kW)","Empty Mass Euro Avg (kg)",
                          "CO2 (g/km)","Combined Consumption (l/100km)"]
                         if c in df_match.columns]
            disp = df_match[show_cols].copy()
            if "Maximum Power (kW)" in disp.columns:
                disp.insert(disp.columns.get_loc("Maximum Power (kW)")+1,
                            "HP", (disp["Maximum Power (kW)"] * 1.36).round(0).astype("Int64"))
            st.dataframe(
                disp.sort_values("CO2 (g/km)").reset_index(drop=True),
                use_container_width=True
            )
