"""
app.py – Vehicle CO₂ · Executive Dashboard
Self-contained: pipeline logic + Streamlit UI in one file.
ADEME Car Labelling Dataset (cl_JUIN_2013-complet3.csv)
"""
# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE  (data loading, preprocessing, model training)
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations
import io
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)
RANDOM_STATE = 42
TARGET = "CO2 (g/km)"

COLUMN_MAPPING = {
    "Marque": "Brand", "Modèle dossier": "Folder Model",
    "Désignation commerciale": "Commercial Designation",
    "Carburant": "Fuel", "Puissance maximale (kW)": "Maximum Power (kW)",
    "Boîte de vitesse": "Gearbox",
    "Consommation mixte (l/100km)": "Combined Consumption (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)", "HC (g/km)": "HC (g/km)",
    "NOX (g/km)": "NOX (g/km)", "HC+NOX (g/km)": "HC+NOX (g/km)",
    "Particules (g/km)": "Particles (g/km)",
    "Consommation urbaine (l/100km)": "Urban Consumption (l/100km)",
    "Consommation extra-urbaine (l/100km)": "Extra Urban Consumption (l/100km)",
    "CO type I (g/km)": "CO type 1 (g/km)",
    "masse vide euro min (kg)": "Empty Mass Euro Min (kg)",
    "masse vide euro max (kg)": "Empty Mass Euro Max (kg)",
    "Carrosserie": "Body", "gamme": "Range",
    "Modèle UTAC": "Utac Model", "CNIT": "cnit",
    "Type Variante Version (TVV)": "Type Variant Version",
    "Hybride": "Hybrid", "Puissance administrative": "Administrative Power",
    "Champ V9": "Field V9", "Date de mise à jour": "Update Date",
}

DEDUP_COLS = [
    "Brand", "Folder Model", "Fuel", "Body", "Gearbox",
    "Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
    "CO2 (g/km)", "Combined Consumption (l/100km)", "Range",
]

GEAR_TYPE_MAP = {
    "M": "Manual", "A": "Automatic", "V": "CVT",
    "D": "DCT", "N": "Automatic", "S": "Manual",
}

FEATURE_SETS = {
    "all_features":    ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel", "GearType", "GearCount", "Body"],
    "no_body":         ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel", "GearType", "GearCount"],
    "mass_power_fuel": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel"],
    "mass_power_only": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)"],
}


# ── Data loading & preprocessing ──────────────────────────────────────────────
def _read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("latin1", "utf-8", "cp1252"):
        for sep in (";", ","):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    return df
            except Exception:
                continue
    raise ValueError("CSV could not be parsed.")


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    if all(c in df.columns for c in ("HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)")):
        hc  = (df["HC+NOX (g/km)"] - df["NOX (g/km)"]).fillna(df["HC (g/km)"])
        nox = (df["HC+NOX (g/km)"] - df["HC (g/km)"]).fillna(df["NOX (g/km)"])
        df["HC (g/km)"], df["NOX (g/km)"] = hc, nox
        df["HC+NOX (g/km)"] = hc + nox

    if "Gearbox" in df.columns:
        df["Gearbox"] = df["Gearbox"].replace({"N 0": "A 0", "N 1": "A 0", "S 6": "D 6"})

    ev_cols = ["CO type 1 (g/km)", "Urban Consumption (l/100km)",
               "Extra Urban Consumption (l/100km)", "Combined Consumption (l/100km)",
               "CO2 (g/km)", "HC+NOX (g/km)", "HC (g/km)", "Particles (g/km)"]
    if "Fuel" in df.columns:
        mask_ev = df["Fuel"] == "EL"
        for c in ev_cols:
            if c in df.columns:
                df.loc[mask_ev, c] = df.loc[mask_ev, c].fillna(0)

    if "Empty Mass Euro Min (kg)" in df.columns and "Empty Mass Euro Max (kg)" in df.columns:
        df["Empty Mass Euro Avg (kg)"] = (
            pd.to_numeric(df["Empty Mass Euro Min (kg)"], errors="coerce")
            + pd.to_numeric(df["Empty Mass Euro Max (kg)"], errors="coerce")
        ) / 2
        df.drop(columns=["Empty Mass Euro Min (kg)", "Empty Mass Euro Max (kg)"], inplace=True)

    for col in (TARGET, "Combined Consumption (l/100km)", "Maximum Power (kW)", "Empty Mass Euro Avg (kg)"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _make_unique(df: pd.DataFrame) -> pd.DataFrame:
    df_c = df[df["Fuel"].isin(["ES", "GO"])].copy()
    if "Gearbox" in df_c.columns:
        gs = df_c["Gearbox"].astype(str).str.split(" ", expand=True)
        df_c["GearType"]  = gs[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_c["GearCount"] = pd.to_numeric(
            gs[1] if 1 in gs.columns else pd.Series(dtype=float), errors="coerce"
        )
    cols = [c for c in DEDUP_COLS if c in df_c.columns]
    df_u = (
        df_c.groupby(cols, dropna=False).size()
        .reset_index(name="Clone_Count")
        .sort_values("Clone_Count", ascending=False)
        .reset_index(drop=True)
    )
    if "Gearbox" in df_u.columns:
        gs2 = df_u["Gearbox"].astype(str).str.split(" ", expand=True)
        df_u["GearType"]  = gs2[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_u["GearCount"] = pd.to_numeric(
            gs2[1] if 1 in gs2.columns else pd.Series(dtype=float), errors="coerce"
        )
    return df_u


def load_data(source=CSV_URL):
    if isinstance(source, str):
        with urllib.request.urlopen(source) as r:
            raw = r.read()
    else:
        raw = bytes(source)
    df_raw    = _preprocess(_read_csv(raw))
    df_unique = _make_unique(df_raw)
    return df_raw, df_unique


# ── Model training ─────────────────────────────────────────────────────────────
@dataclass
class ModelBundle:
    fitted:               dict
    results:              pd.DataFrame
    feature_cols:         list
    best_fs:              str
    fs_comparison:        pd.DataFrame
    X_train:              pd.DataFrame
    X_test:               pd.DataFrame
    y_train:              pd.Series
    y_test:               pd.Series
    rf_pipe:              object
    feature_importance:   pd.DataFrame
    numeric_features:     list
    categorical_features: list


def train_models(df_unique: pd.DataFrame) -> ModelBundle:
    all_cols = sorted({TARGET} | {c for cols in FEATURE_SETS.values() for c in cols})
    df_m = df_unique[[c for c in all_cols if c in df_unique.columns]].dropna().copy()

    def split_types(feats):
        num = df_m[feats].select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat = df_m[feats].select_dtypes(include=["object", "category"]).columns.tolist()
        return num, cat

    def make_preprocessors(num, cat):
        scaled = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])
        tree = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ])
        return scaled, tree

    # Feature-set cross-validation
    fs_rows = []
    for name, feats in FEATURE_SETS.items():
        avail = [f for f in feats if f in df_m.columns]
        if not avail:
            continue
        _, tree_pre = make_preprocessors(*split_types(avail))
        pipe = Pipeline([("pre", tree_pre),
                         ("m", RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1))])
        scores = cross_val_score(pipe, df_m[avail], df_m[TARGET],
                                 cv=5, scoring="neg_mean_absolute_error")
        fs_rows.append({
            "Feature_Set": name, "Features": ", ".join(avail),
            "CV_MAE_mean": -scores.mean(), "CV_MAE_std": scores.std(),
        })

    fs_df     = pd.DataFrame(fs_rows).sort_values("CV_MAE_mean")
    best_fs   = fs_df.iloc[0]["Feature_Set"]
    feat_cols = [f for f in FEATURE_SETS[best_fs] if f in df_m.columns]

    X, y = df_m[feat_cols], df_m[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    num_f, cat_f    = split_types(feat_cols)
    scaled_pre, tree_pre = make_preprocessors(num_f, cat_f)

    model_defs = {
        "Linear Regression": Pipeline([("pre", scaled_pre), ("m", LinearRegression())]),
        "Ridge":             Pipeline([("pre", scaled_pre), ("m", Ridge(alpha=1.0))]),
        "Lasso":             Pipeline([("pre", scaled_pre), ("m", Lasso(alpha=0.1))]),
        "Random Forest":     Pipeline([("pre", tree_pre),
                                       ("m", RandomForestRegressor(
                                           n_estimators=300, max_depth=20, max_features=0.8,
                                           random_state=RANDOM_STATE, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("pre", tree_pre),
                                       ("m", GradientBoostingRegressor(
                                           n_estimators=200, learning_rate=0.2, max_depth=6,
                                           subsample=1.0, max_features=0.5, random_state=RANDOM_STATE))]),
    }

    results, fitted = [], {}
    for name, pipe in model_defs.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append({
            "Model":     name,
            "Train_R2":  r2_score(y_train, pipe.predict(X_train)),
            "Test_R2":   r2_score(y_test,  pipe.predict(X_test)),
            "Train_MAE": mean_absolute_error(y_train, pipe.predict(X_train)),
            "Test_MAE":  mean_absolute_error(y_test,  pipe.predict(X_test)),
        })

    results_df = pd.DataFrame(results).sort_values("Test_MAE")
    rf_pipe    = fitted["Random Forest"]
    fi_names   = rf_pipe.named_steps["pre"].get_feature_names_out()
    fi_df      = (
        pd.DataFrame({"Feature": fi_names, "Importance": rf_pipe.named_steps["m"].feature_importances_})
        .sort_values("Importance", ascending=False).reset_index(drop=True)
    )
    return ModelBundle(
        fitted=fitted, results=results_df, feature_cols=feat_cols,
        best_fs=best_fs, fs_comparison=fs_df,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        rf_pipe=rf_pipe, feature_importance=fi_df,
        numeric_features=num_f, categorical_features=cat_f,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────
def segment_filter(df_unique, fuel, body, kw_lo=0, kw_hi=9999, gear_prefix=None):
    mask = (
        df_unique["Fuel"].eq(fuel)
        & df_unique["Body"].eq(body)
        & df_unique["Maximum Power (kW)"].between(kw_lo, kw_hi)
    )
    if gear_prefix:
        mask &= df_unique["Gearbox"].astype(str).str.startswith(gear_prefix)
    return df_unique[mask].copy()


def predict_co2(bundle, row_dict, model_name="Random Forest"):
    return float(bundle.fitted[model_name].predict(pd.DataFrame([row_dict]))[0])


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CO₂ Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
MINT   = "#00C8A0"
AMBER  = "#F5A623"
RED    = "#E84855"
BG     = "#0D0F18"
CARD   = "#13161F"
BORDER = "#1E2130"
TEXT   = "#EDF0F7"
MUTED  = "#6B7280"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --mint:MINT_VAL; --amber:AMBER_VAL; --red:RED_VAL;
  --bg:BG_VAL; --card:CARD_VAL; --border:BORDER_VAL;
  --text:TEXT_VAL; --muted:MUTED_VAL;
  --font: 'Inter', system-ui, sans-serif;
  --mono: 'JetBrains Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--font) !important; }
section[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1400px; }
h1 { font-size: 1.75rem !important; font-weight: 800 !important; letter-spacing: -.04em !important; color: var(--text) !important; margin: 0 !important; }
h2 { font-size: .75rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .12em !important; border: none !important; margin: 0 0 1rem !important; }
h3 { font-size: 1rem !important; font-weight: 600 !important; color: var(--text) !important; }
p, li { color: var(--text) !important; line-height: 1.6 !important; }
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 1.1rem 1.3rem !important; border-top: none !important; }
[data-testid="stMetricLabel"] p { font-size: .65rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .1em !important; margin: 0 !important; }
[data-testid="stMetricValue"] { font-size: 1.55rem !important; font-weight: 700 !important; color: var(--text) !important; font-family: var(--mono) !important; }
[data-testid="stMetricDelta"] div { font-size: .75rem !important; }
[data-testid="stSlider"] label { font-size: .7rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .09em !important; }
[data-testid="stSelectbox"] label, [data-testid="stRadio"] label { font-size: .7rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .09em !important; }
[data-testid="stSelectbox"] > div > div { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 8px !important; }
[data-testid="stRadio"] div[role="radiogroup"] label { color: var(--text) !important; font-size: .88rem !important; text-transform: none !important; letter-spacing: normal !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-weight: 500 !important; font-size: .85rem !important; color: var(--text) !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.75rem 0 !important; }
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: .73rem !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
"""
# Inject token values via simple string replacement (no f-string brace conflicts)
_CSS = (_CSS
    .replace("MINT_VAL",   MINT)
    .replace("AMBER_VAL",  AMBER)
    .replace("RED_VAL",    RED)
    .replace("BG_VAL",     BG)
    .replace("CARD_VAL",   CARD)
    .replace("BORDER_VAL", BORDER)
    .replace("TEXT_VAL",   TEXT)
    .replace("MUTED_VAL",  MUTED)
)
st.markdown(_CSS, unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": CARD,  "axes.facecolor":  CARD,
    "axes.edgecolor":   BORDER,"axes.labelcolor": MUTED,
    "axes.titlecolor":  TEXT,  "axes.titlesize":  11,
    "axes.titleweight": "600", "axes.labelsize":  9,
    "axes.grid":        True,  "grid.color":      BORDER, "grid.linewidth": 0.4,
    "axes.spines.top":  False, "axes.spines.right": False,
    "text.color":       TEXT,  "xtick.color":     MUTED,  "ytick.color": MUTED,
    "xtick.labelsize":  8,     "ytick.labelsize": 8,
    "legend.facecolor": CARD,  "legend.edgecolor": BORDER, "legend.fontsize": 8,
    "savefig.facecolor": CARD, "savefig.edgecolor": CARD,
})


def co2_color(v):
    return MINT if v <= 120 else AMBER if v <= 160 else RED


def euro_class(v):
    if v <= 100: return "A"
    if v <= 120: return "B"
    if v <= 140: return "C"
    if v <= 160: return "D"
    if v <= 200: return "E"
    return "F/G"


# ── Load & train (cached) ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load(source):
    return load_data(source)


@st.cache_resource(show_spinner=False)
def _train(_df):
    return train_models(_df)


with st.spinner("Loading data …"):
    df_raw, df_u = _load(CSV_URL)

with st.spinner("Training model …"):
    mb = _train(df_u)

best_model = "Random Forest"
rf_mae = float(mb.results[mb.results["Model"] == best_model]["Test_MAE"].iloc[0])
rf_r2  = float(mb.results[mb.results["Model"] == best_model]["Test_R2"].iloc[0])


# ── Helper: build a feature row ────────────────────────────────────────────────
def make_row(mass, power, gears, fuel, body, gtype, feature_cols):
    row = {f: 0 for f in feature_cols}
    for k, v in {
        "Empty Mass Euro Avg (kg)": float(mass),
        "Maximum Power (kW)":       float(power),
        "GearCount":                float(gears),
        "Fuel": fuel, "Body": body, "GearType": gtype,
    }.items():
        if k in row:
            row[k] = v
    return row


# ── Base values (dataset medians) ─────────────────────────────────────────────
base = {
    "mass":  float(df_u["Empty Mass Euro Avg (kg)"].median()),
    "power": float(df_u["Maximum Power (kW)"].median()),
    "gears": float(df_u["GearCount"].median()) if "GearCount" in df_u.columns else 6.0,
    "fuel":  str(df_u["Fuel"].mode().iloc[0]),
    "body":  str(df_u["Body"].mode().iloc[0]),
    "gtype": str(df_u["GearType"].mode().iloc[0]) if "GearType" in df_u.columns else "Manual",
}
base_pred = predict_co2(mb, make_row(**base, feature_cols=mb.feature_cols))

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:.25rem'>"
    "<span style='font-size:1.75rem;font-weight:800;letter-spacing:-.04em;color:#EDF0F7'>CO\u2082 Intelligence</span>"
    "<span style='font-size:.72rem;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:.12em'>ADEME \u00b7 France \u00b7 2013</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-size:.83rem;color:#6B7280;margin-bottom:2rem'>"
    "Predictive model for vehicle CO\u2082 emissions \u2014 Random Forest \u00b7 "
    "R\u00b2 <span style='color:#EDF0F7'>{r2:.2f}</span> \u00b7 "
    "MAE <span style='color:#EDF0F7'>{mae:.1f} g/km</span> \u00b7 "
    "{n:,} unique configurations</div>".format(r2=rf_r2, mae=rf_mae, n=len(df_u)),
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
fleet_med  = float(df_u["CO2 (g/km)"].median())
fleet_mean = float(df_u["CO2 (g/km)"].mean())
n_brands   = df_u["Brand"].nunique()
pct_sub130 = (df_u["CO2 (g/km)"] <= 130).mean() * 100

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fleet Median",      "{:.0f} g/km".format(fleet_med))
k2.metric("Fleet Mean",        "{:.0f} g/km".format(fleet_mean))
k3.metric("Brands",            str(n_brands))
k4.metric("Configs \u2264130 g/km", "{:.0f}%".format(pct_sub130))
k5.metric("Model MAE",         "{:.1f} g/km".format(rf_mae), delta="R\u00b2 {:.3f}".format(rf_r2))

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN: Simulator (left) | Brand Comparison (right)
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.1, 1.9], gap="large")

with col_left:
    st.markdown("## Configure Vehicle")

    sim_mass  = st.slider("Kerb Weight (kg)",  800,  3200, int(base["mass"]),  50)
    sim_power = st.slider("Max Power (kW)",     40,   560, int(base["power"]),  5)

    if "GearCount" in mb.feature_cols:
        sim_gears = st.slider("Number of Gears", 4, 8, int(base["gears"]), 1)
    else:
        sim_gears = base["gears"]

    ca, cb = st.columns(2)
    with ca:
        sim_fuel  = st.radio("Fuel",    ["Diesel (GO)", "Petrol (ES)"],
                             index=0 if base["fuel"] == "GO" else 1)
    with cb:
        sim_gtype = st.radio("Gearbox", ["Manual", "Automatic"],
                             index=0 if base["gtype"] == "Manual" else 1)

    body_opts = sorted(df_u["Body"].dropna().unique().tolist())
    sim_body  = st.selectbox(
        "Body Style", body_opts,
        index=body_opts.index(base["body"]) if base["body"] in body_opts else 0,
    )

    fuel_code  = "GO" if "GO" in sim_fuel else "ES"
    gtype_code = sim_gtype  # "Manual" or "Automatic"

    sim_row  = make_row(sim_mass, sim_power, sim_gears, fuel_code, sim_body, gtype_code, mb.feature_cols)
    sim_pred = predict_co2(mb, sim_row)
    delta    = sim_pred - base_pred
    clr      = co2_color(sim_pred)
    ecls     = euro_class(sim_pred)
    dclr     = RED if delta > 0 else MINT if delta < 0 else MUTED

    # Prediction card
    arrow = "\u25b2" if delta > 0 else "\u25bc" if delta < 0 else "\u2014"
    st.markdown(
        "<div style='background:{card};border:1px solid {border};border-radius:14px;"
        "padding:1.5rem 1.75rem;margin-top:1.25rem;border-left:4px solid {clr}'>"
        "<div style='font-size:.65rem;font-weight:600;color:{muted};"
        "text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem'>Predicted CO\u2082</div>"
        "<div style='display:flex;align-items:baseline;gap:.6rem'>"
        "<span style='font-size:3.2rem;font-weight:800;color:{clr};"
        "font-family:var(--mono);letter-spacing:-.04em'>{pred:.0f}</span>"
        "<span style='font-size:1rem;color:{muted}'>g/km</span>"
        "<span style='font-size:.75rem;font-weight:700;color:{clr};"
        "background:{clr}20;border-radius:5px;padding:.2rem .55rem'>Class {ecls}</span>"
        "</div>"
        "<div style='font-size:.85rem;color:{dclr};font-weight:600;margin-top:.4rem'>"
        "{arrow} {delta:+.1f} g/km vs fleet base</div>"
        "<div style='font-size:.72rem;color:{muted};margin-top:.6rem'>"
        "Annual CO\u2082 \u2248 {annual:.0f} kg @ 15,000 km \u00b7 Model MAE \u00b1 {mae:.1f} g/km"
        "</div></div>".format(
            card=CARD, border=BORDER, clr=clr, muted=MUTED, dclr=dclr,
            pred=sim_pred, ecls=ecls, arrow=arrow,
            delta=delta, annual=sim_pred * 15000 / 1000, mae=rf_mae,
        ),
        unsafe_allow_html=True,
    )

    # Sensitivity sparklines (batched)
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    fig.patch.set_facecolor(CARD)
    for ax, key, rng, clr_line, label in [
        (axes[0], "Empty Mass Euro Avg (kg)", np.arange(800,  3300, 80), MINT,  "Mass (kg)"),
        (axes[1], "Maximum Power (kW)",        np.arange(40,   570, 15), AMBER, "Power (kW)"),
    ]:
        batch = pd.DataFrame([dict(sim_row, **{key: float(v)}) for v in rng])
        preds = mb.fitted["Random Forest"].predict(batch)
        ax.plot(rng, preds, color=clr_line, lw=1.8)
        ax.fill_between(rng, preds, alpha=0.07, color=clr_line)
        ax.axvline(sim_row[key], color=RED, lw=1.2, linestyle="--", alpha=0.8)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("CO\u2082 g/km", fontsize=8)
        ax.set_title("CO\u2082 vs. " + label.split(" ")[0], fontsize=9)
    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ── Right column ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown("## Brand Comparison")

    if   sim_power <= 55:  kw_lo, kw_hi = 0,   55
    elif sim_power <= 96:  kw_lo, kw_hi = 56,  96
    elif sim_power <= 147: kw_lo, kw_hi = 97,  147
    else:                  kw_lo, kw_hi = 148, 600

    df_seg = segment_filter(df_u, fuel_code, sim_body, kw_lo, kw_hi)
    if len(df_seg) < 5:
        df_seg = df_u[df_u["Fuel"].eq(fuel_code) & df_u["Body"].eq(sim_body)].copy()

    avail_brands = (
        df_seg.groupby("Brand")["CO2 (g/km)"].count()
        .where(lambda x: x >= 2).dropna()
        .sort_values(ascending=False).index.tolist()
    )

    if not avail_brands:
        st.info("No brands with \u22652 models for this segment. Adjust Body Style or Fuel.")
    else:
        selected = st.multiselect(
            "Brands (max 5)",
            options=avail_brands,
            default=avail_brands[:4],
            max_selections=5,
        )

        if selected:
            PALETTE = [MINT, AMBER, RED, "#A78BFA", "#38BDF8"]

            rows = []
            for brand in selected:
                df_b  = df_seg[df_seg["Brand"] == brand]
                co2_s = df_b["CO2 (g/km)"].dropna()
                if co2_s.empty:
                    continue
                b_mass  = float(df_b["Empty Mass Euro Avg (kg)"].median())
                b_power = float(df_b["Maximum Power (kW)"].median())
                b_gears = float(df_b["GearCount"].median()) if "GearCount" in df_b.columns else sim_gears
                b_gtype = (str(df_b["GearType"].mode().iloc[0])
                           if "GearType" in df_b.columns and len(df_b) > 0 else gtype_code)
                b_row   = make_row(b_mass, b_power, b_gears, fuel_code, sim_body, b_gtype, mb.feature_cols)
                rows.append({
                    "Brand":      brand,
                    "N":          len(co2_s),
                    "Median":     co2_s.median(),
                    "P25":        co2_s.quantile(0.25),
                    "P75":        co2_s.quantile(0.75),
                    "Min":        co2_s.min(),
                    "Max":        co2_s.max(),
                    "Pred":       predict_co2(mb, b_row),
                    "Typical_kW": b_power,
                    "Typical_kg": b_mass,
                })

            bdf  = pd.DataFrame(rows).sort_values("Median").reset_index(drop=True)
            # Colors assigned after sort so rank-1 always gets MINT
            bclr = {b: PALETTE[i] for i, b in enumerate(bdf["Brand"])}

            # Metric cards
            best_median  = bdf["Median"].iloc[0]
            brand_cols   = st.columns(len(bdf))
            for col_ui, (_, r) in zip(brand_cols, bdf.iterrows()):
                diff     = r["Median"] - best_median
                diff_lbl = (
                    "<div style='font-size:.62rem;color:#00C8A0;font-weight:600;margin-top:.3rem'>\u2605 Most efficient</div>"
                    if r.name == 0 else
                    "<div style='font-size:.62rem;color:{muted};margin-top:.3rem'>+{d:.0f} g/km vs best</div>".format(
                        muted=MUTED, d=diff)
                )
                col_ui.markdown(
                    "<div style='background:{card};border:1px solid {border};border-radius:12px;"
                    "border-top:3px solid {clr};padding:.9rem;text-align:center'>"
                    "<div style='font-size:.6rem;font-weight:600;color:{muted};"
                    "text-transform:uppercase;letter-spacing:.1em;margin-bottom:.2rem'>{brand}</div>"
                    "<div style='font-size:1.6rem;font-weight:800;color:{clr};font-family:var(--mono)'>{med:.0f}</div>"
                    "<div style='font-size:.65rem;color:{muted}'>g/km \u00b7 {n} models</div>"
                    "{diff_lbl}</div>".format(
                        card=CARD, border=BORDER, clr=bclr[r["Brand"]], muted=MUTED,
                        brand=r["Brand"], med=r["Median"], n=int(r["N"]), diff_lbl=diff_lbl,
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

            ch1, ch2 = st.columns(2)

            with ch1:
                fig, ax = plt.subplots(figsize=(6, 3.8))
                fig.patch.set_facecolor(CARD)
                x    = np.arange(len(bdf))
                clrs = [bclr[b] for b in bdf["Brand"]]
                ax.bar(x, bdf["Median"], color=clrs, alpha=0.75, edgecolor=CARD, lw=0, width=0.5)
                ax.errorbar(x, bdf["Median"],
                            yerr=[bdf["Median"] - bdf["P25"], bdf["P75"] - bdf["Median"]],
                            fmt="none", color=TEXT, capsize=4, lw=1.2, alpha=0.5)
                ax.scatter(x, bdf["Pred"], color=TEXT, s=40, zorder=5,
                           marker="D", label="Model prediction")
                ax.axhline(sim_pred, color=MUTED, lw=1, linestyle="--", alpha=0.7,
                           label="Simulator: {:.0f}".format(sim_pred))
                ax.set_xticks(x)
                ax.set_xticklabels(bdf["Brand"], rotation=20, ha="right")
                ax.set_ylabel("CO\u2082 (g/km)")
                ax.set_title("Median \u00b1 IQR  \u25c6 Prediction")
                ax.legend(fontsize=7)
                for xi, (_, r) in zip(x, bdf.iterrows()):
                    ax.text(xi, r["P75"] + 1.5, "{:.0f}".format(r["Median"]),
                            ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="600")
                plt.tight_layout(pad=0.6)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with ch2:
                fig, ax = plt.subplots(figsize=(6, 3.8))
                fig.patch.set_facecolor(CARD)
                pw_rng = np.arange(40, 570, 12)
                for _, r in bdf.iterrows():
                    b      = r["Brand"]
                    b_base = make_row(r["Typical_kg"], sim_power, sim_gears,
                                      fuel_code, sim_body, gtype_code, mb.feature_cols)
                    batch  = pd.DataFrame([dict(b_base, **{"Maximum Power (kW)": float(p)}) for p in pw_rng])
                    preds  = mb.fitted["Random Forest"].predict(batch)
                    ax.plot(pw_rng, preds, color=bclr[b], lw=1.8, label=b)
                    ax.scatter([r["Typical_kW"]], [r["Pred"]], color=bclr[b], s=40, zorder=5)
                ax.axvline(sim_power, color=MUTED, lw=1, linestyle="--", alpha=0.7,
                           label="Current: {} kW".format(sim_power))
                ax.set_xlabel("Max Power (kW)")
                ax.set_ylabel("Predicted CO\u2082 (g/km)")
                ax.set_title("CO\u2082 vs. Power by Brand")
                ax.legend(fontsize=7)
                plt.tight_layout(pad=0.6)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with st.expander("Full comparison table"):
                disp = bdf.rename(columns={
                    "N": "Models", "Median": "Median CO\u2082", "Pred": "Model Pred.",
                    "Typical_kW": "Typical kW", "Typical_kg": "Typical kg",
                })
                num_disp_cols = ["Median CO\u2082", "P25", "P75", "Min", "Max",
                                 "Model Pred.", "Typical kW", "Typical kg"]
                st.dataframe(
                    disp.style
                    .format({c: "{:.1f}" for c in num_disp_cols})
                    .highlight_min(subset=["Median CO\u2082"], color="#00C8A018")
                    .highlight_max(subset=["Median CO\u2082"], color="#E8485518"),
                    use_container_width=True,
                    hide_index=True,
                )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='display:flex;justify-content:space-between;align-items:center;"
    "font-size:.7rem;color:{muted}'>"
    "<span>ADEME Car Labelling Dataset 2013 \u00b7 {n:,} unique ES/GO configurations</span>"
    "<span>Random Forest \u00b7 {fs} \u00b7 R\u00b2 {r2:.3f} \u00b7 MAE {mae:.1f} g/km \u00b7 "
    "<a href='https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction' "
    "style='color:{mint};text-decoration:none'>GitHub \u2197</a></span>"
    "</div>".format(
        muted=MUTED, n=len(df_u), fs=mb.best_fs,
        r2=rf_r2, mae=rf_mae, mint=MINT,
    ),
    unsafe_allow_html=True,
)
