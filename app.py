"""
search_app.py – Vehicle CO₂ Search Portal
ADEME Car Labelling Dataset — multi-criteria search, comparison, CO₂ label.
Self-contained, no external imports.
"""
from __future__ import annotations
import io
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)

COLUMN_MAPPING = {
    "Marque": "Brand",
    "Modèle dossier": "Folder Model",
    "Désignation commerciale": "Commercial Designation",
    "Carburant": "Fuel",
    "Puissance maximale (kW)": "Max Power (kW)",
    "Puissance administrative": "Fiscal Power (CV)",
    "Boîte de vitesse": "Gearbox",
    "Consommation urbaine (l/100km)": "Urban Conso (l/100km)",
    "Consommation extra-urbaine (l/100km)": "ExUrban Conso (l/100km)",
    "Consommation mixte (l/100km)": "Mixed Conso (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)",
    "CO type I (g/km)": "CO (g/km)",
    "HC (g/km)": "HC (g/km)",
    "NOX (g/km)": "NOx (g/km)",
    "HC+NOX (g/km)": "HC+NOx (g/km)",
    "Particules (g/km)": "Particles (g/km)",
    "masse vide euro min (kg)": "Mass Min (kg)",
    "masse vide euro max (kg)": "Mass Max (kg)",
    "Carrosserie": "Body",
    "gamme": "Segment",
    "Hybride": "Hybrid",
    "Champ V9": "Euro Norm",
    "Date de mise à jour": "Updated",
}

FUEL_LABELS = {
    "ES": "Petrol",   "GO": "Diesel",  "EL": "Electric",
    "GH": "Hybrid (Diesel)", "EH": "Hybrid (Petrol)",
    "EE": "Plug-in Hybrid (Petrol)", "GL": "Plug-in Hybrid (Diesel)",
    "GP": "LPG", "GN": "CNG", "FE": "E85",
}

GEAR_LABELS = {
    "M": "Manual", "A": "Automatic", "V": "CVT", "D": "DCT",
}

CO2_CLASSES = {
    "A": (0,   100),
    "B": (101, 120),
    "C": (121, 140),
    "D": (141, 160),
    "E": (161, 200),
    "F": (201, 250),
    "G": (251, 9999),
}

CLASS_COLOR = {
    "A": "#1a9641", "B": "#52b747", "C": "#a6d96a",
    "D": "#ffffbf", "E": "#fdae61", "F": "#f46d43", "G": "#d73027",
}
CLASS_TEXT  = {"A":"#fff","B":"#fff","C":"#111","D":"#111","E":"#111","F":"#fff","G":"#fff"}


def co2_class(v):
    if pd.isna(v): return "?"
    for cls, (lo, hi) in CO2_CLASSES.items():
        if lo <= v <= hi:
            return cls
    return "G"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    with urllib.request.urlopen(CSV_URL) as r:
        raw = r.read()

    df = None
    for enc in ("latin1", "utf-8", "cp1252"):
        for sep in (";", ","):
            try:
                tmp = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if tmp.shape[1] > 5:
                    df = tmp; break
            except Exception:
                continue
        if df is not None:
            break
    if df is None:
        raise ValueError("Could not parse CSV.")

    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    # HC/NOx imputation
    if all(c in df.columns for c in ("HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)")):
        hc  = (df["HC+NOx (g/km)"] - df["NOx (g/km)"]).fillna(df["HC (g/km)"])
        nox = (df["HC+NOx (g/km)"] - df["HC (g/km)"]).fillna(df["NOx (g/km)"])
        df["HC (g/km)"], df["NOx (g/km)"] = hc, nox
        df["HC+NOx (g/km)"] = hc + nox

    # Gearbox fix
    if "Gearbox" in df.columns:
        df["Gearbox"] = df["Gearbox"].replace({"N 0":"A 0","N 1":"A 0","S 6":"D 6"})
        gs = df["Gearbox"].astype(str).str.split(" ", expand=True)
        df["Gear Type"]  = gs[0].map(GEAR_LABELS).fillna(gs[0])
        df["Gear Count"] = pd.to_numeric(
            gs[1] if 1 in gs.columns else pd.Series(dtype=float), errors="coerce"
        ).astype("Int64")

    # EV zeros
    ev_cols = ["CO2 (g/km)", "Mixed Conso (l/100km)", "Urban Conso (l/100km)",
               "ExUrban Conso (l/100km)", "HC (g/km)", "NOx (g/km)", "Particles (g/km)"]
    if "Fuel" in df.columns:
        mask_ev = df["Fuel"] == "EL"
        for c in ev_cols:
            if c in df.columns:
                df.loc[mask_ev, c] = df.loc[mask_ev, c].fillna(0)

    # Average mass
    if "Mass Min (kg)" in df.columns and "Mass Max (kg)" in df.columns:
        df["Mass (kg)"] = (
            pd.to_numeric(df["Mass Min (kg)"], errors="coerce")
            + pd.to_numeric(df["Mass Max (kg)"], errors="coerce")
        ) / 2

    # Numeric coercion
    for c in ("CO2 (g/km)", "Mixed Conso (l/100km)", "Max Power (kW)", "Mass (kg)",
              "HC (g/km)", "NOx (g/km)", "Particles (g/km)", "CO (g/km)"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fuel label
    if "Fuel" in df.columns:
        df["Fuel Label"] = df["Fuel"].map(FUEL_LABELS).fillna(df["Fuel"])

    # CO2 class
    if "CO2 (g/km)" in df.columns:
        df["CO2 Class"] = df["CO2 (g/km)"].apply(co2_class)

    # HP
    if "Max Power (kW)" in df.columns:
        df["Max Power (HP)"] = (df["Max Power (kW)"] * 1.36).round(0).astype("Int64")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CO\u2082 Vehicle Search",
    page_icon="\u25c8",
    layout="wide",
    initial_sidebar_state="expanded",
)

MINT   = "#00C8A0"
AMBER  = "#F5A623"
RED    = "#E84855"
BG     = "#0D0F18"
CARD   = "#13161F"
BORDER = "#1E2130"
TEXT   = "#EDF0F7"
MUTED  = "#6B7280"
SIDEBAR= "#0F1219"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --mint:MINT_V; --amber:AMBER_V; --red:RED_V;
  --bg:BG_V; --card:CARD_V; --border:BORDER_V;
  --text:TEXT_V; --muted:MUTED_V; --sidebar:SIDEBAR_V;
  --font:'Inter',system-ui,sans-serif; --mono:'JetBrains Mono',monospace;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--font) !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 100%; }
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { background: var(--sidebar) !important; border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMarkdown a { color: var(--mint) !important; }
h1 { font-size: 1.6rem !important; font-weight: 800 !important; letter-spacing: -.04em !important; color: var(--text) !important; margin: 0 !important; }
h2 { font-size: .7rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .12em !important; border: none !important; margin: 0 0 .75rem !important; }
h3 { font-size: .95rem !important; font-weight: 600 !important; color: var(--text) !important; }
p, li { color: var(--text) !important; line-height: 1.6 !important; }
[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: .9rem 1.1rem !important; }
[data-testid="stMetricLabel"] p { font-size: .62rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .1em !important; margin: 0 !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; font-family: var(--mono) !important; color: var(--text) !important; }
[data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label, [data-testid="stRadio"] label,
[data-testid="stCheckbox"] label { font-size: .68rem !important; font-weight: 600 !important; color: var(--muted) !important; text-transform: uppercase !important; letter-spacing: .09em !important; }
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; border-radius: 7px !important; }
[data-testid="stRadio"] div[role="radiogroup"] label { color: var(--text) !important; font-size: .85rem !important; text-transform: none !important; letter-spacing: normal !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { font-weight: 500 !important; font-size: .85rem !important; color: var(--text) !important; }
.stButton > button { background: var(--mint) !important; color: #0D0F18 !important; border: none !important; border-radius: 7px !important; font-weight: 700 !important; font-size: .82rem !important; letter-spacing: .04em !important; padding: .5rem 1.2rem !important; transition: opacity .15s !important; }
.stButton > button:hover { opacity: .85 !important; }
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.25rem 0 !important; }
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: .72rem !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
"""
_CSS = (_CSS
    .replace("MINT_V",   MINT)
    .replace("AMBER_V",  AMBER)
    .replace("RED_V",    RED)
    .replace("BG_V",     BG)
    .replace("CARD_V",   CARD)
    .replace("BORDER_V", BORDER)
    .replace("TEXT_V",   TEXT)
    .replace("MUTED_V",  MUTED)
    .replace("SIDEBAR_V",SIDEBAR)
)
st.markdown(_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading ADEME dataset \u2026"):
    df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='padding:.25rem 0 1rem'>"
        "<div style='font-size:1.2rem;font-weight:800;letter-spacing:-.03em'>CO\u2082 Search</div>"
        "<div style='font-size:.65rem;font-weight:600;color:{muted};text-transform:uppercase;letter-spacing:.1em'>ADEME \u00b7 France \u00b7 2013</div>"
        "</div>".format(muted=MUTED),
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Brand
    brands = sorted(df["Brand"].dropna().unique())
    sel_brand = st.multiselect("Brand", brands, placeholder="All brands")

    # Model — filtered by brand selection
    if sel_brand:
        models_pool = df[df["Brand"].isin(sel_brand)]["Folder Model"].dropna().unique()
    else:
        models_pool = df["Folder Model"].dropna().unique()
    models = sorted(models_pool)
    sel_model = st.multiselect("Model", models, placeholder="All models")

    st.markdown("---")

    # Body / category
    bodies = sorted(df["Body"].dropna().unique())
    sel_body = st.multiselect("Body Style", bodies, placeholder="All")

    # Segment / range
    if "Segment" in df.columns:
        segments = sorted(df["Segment"].dropna().unique())
        sel_seg = st.multiselect("Segment", segments, placeholder="All")
    else:
        sel_seg = []

    st.markdown("---")

    # Fuel
    fuel_opts = sorted(df["Fuel Label"].dropna().unique()) if "Fuel Label" in df.columns else []
    sel_fuel = st.multiselect("Fuel / Energy", fuel_opts, placeholder="All")

    # Gearbox
    if "Gear Type" in df.columns:
        gear_opts = sorted(df["Gear Type"].dropna().unique())
        sel_gear = st.multiselect("Gearbox", gear_opts, placeholder="All")
    else:
        sel_gear = []

    st.markdown("---")

    # CO2 class
    sel_class = st.multiselect(
        "CO\u2082 Class",
        options=["A","B","C","D","E","F","G"],
        placeholder="All classes",
    )

    # CO2 max slider
    co2_max_val = int(df["CO2 (g/km)"].dropna().max()) if "CO2 (g/km)" in df.columns else 600
    co2_range = st.slider(
        "CO\u2082 Range (g/km)",
        0, co2_max_val, (0, co2_max_val), step=5,
    )

    st.markdown("---")

    # Power
    pw_max = int(df["Max Power (kW)"].dropna().max()) if "Max Power (kW)" in df.columns else 600
    pw_range = st.slider(
        "Max Power (kW)",
        0, pw_max, (0, pw_max), step=5,
    )

    # Mass
    if "Mass (kg)" in df.columns:
        mass_max = int(df["Mass (kg)"].dropna().max())
        mass_range = st.slider("Kerb Mass (kg)", 0, mass_max, (0, mass_max), step=50)
    else:
        mass_range = None

    st.markdown("---")
    st.caption("{:,} vehicles in dataset".format(len(df)))

# ══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
mask = pd.Series(True, index=df.index)

if sel_brand:
    mask &= df["Brand"].isin(sel_brand)
if sel_model:
    mask &= df["Folder Model"].isin(sel_model)
if sel_body:
    mask &= df["Body"].isin(sel_body)
if sel_seg and "Segment" in df.columns:
    mask &= df["Segment"].isin(sel_seg)
if sel_fuel and "Fuel Label" in df.columns:
    mask &= df["Fuel Label"].isin(sel_fuel)
if sel_gear and "Gear Type" in df.columns:
    mask &= df["Gear Type"].isin(sel_gear)
if sel_class and "CO2 Class" in df.columns:
    mask &= df["CO2 Class"].isin(sel_class)
if "CO2 (g/km)" in df.columns:
    mask &= df["CO2 (g/km)"].between(co2_range[0], co2_range[1], inclusive="both") | df["CO2 (g/km)"].isna()
if "Max Power (kW)" in df.columns:
    mask &= df["Max Power (kW)"].between(pw_range[0], pw_range[1], inclusive="both") | df["Max Power (kW)"].isna()
if mass_range is not None and "Mass (kg)" in df.columns:
    mask &= df["Mass (kg)"].between(mass_range[0], mass_range[1], inclusive="both") | df["Mass (kg)"].isna()

results = df[mask].copy()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:.2rem'>"
    "<span style='font-size:1.6rem;font-weight:800;letter-spacing:-.04em'>CO\u2082 Vehicle Search</span>"
    "<span style='font-size:.68rem;font-weight:600;color:{muted};text-transform:uppercase;letter-spacing:.12em'>"
    "ADEME Car Labelling \u00b7 France \u00b7 2013</span>"
    "</div>".format(muted=MUTED),
    unsafe_allow_html=True,
)

# KPI strip
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Results",        "{:,}".format(len(results)))
k2.metric("Brands",         str(results["Brand"].nunique()) if len(results) else "0")
k3.metric("Body Styles",    str(results["Body"].nunique())  if len(results) else "0")

if len(results) and "CO2 (g/km)" in results.columns:
    med_co2 = results["CO2 (g/km)"].median()
    min_co2 = results["CO2 (g/km)"].min()
    k4.metric("Median CO\u2082",  "{:.0f} g/km".format(med_co2))
    k5.metric("Best CO\u2082",    "{:.0f} g/km".format(min_co2))
else:
    k4.metric("Median CO\u2082", "—")
    k5.metric("Best CO\u2082",   "—")

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NO RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if len(results) == 0:
    st.markdown(
        "<div style='background:{card};border:1px solid {border};border-radius:12px;"
        "padding:2rem;text-align:center;color:{muted};margin-top:1rem'>"
        "<div style='font-size:2rem;margin-bottom:.5rem'>\u26aa</div>"
        "<div style='font-weight:600;color:{text}'>No results found</div>"
        "<div style='font-size:.85rem;margin-top:.4rem'>"
        "Try adjusting your filters in the sidebar.</div>"
        "</div>".format(card=CARD, border=BORDER, muted=MUTED, text=TEXT),
        unsafe_allow_html=True,
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SORT + COLUMN SELECTION
# ══════════════════════════════════════════════════════════════════════════════
DISPLAY_COLS = [c for c in [
    "Brand", "Folder Model", "Body", "Segment",
    "Fuel Label", "Gear Type", "Gear Count",
    "Max Power (kW)", "Max Power (HP)",
    "Mixed Conso (l/100km)", "CO2 (g/km)", "CO2 Class",
    "HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)", "Particles (g/km)",
    "Mass (kg)", "Euro Norm",
] if c in results.columns]

ctrl1, ctrl2, _ = st.columns([1.2, 1.2, 3])
with ctrl1:
    sort_opts = [c for c in ["CO2 (g/km)", "Brand", "Max Power (kW)", "Mixed Conso (l/100km)", "Mass (kg)"] if c in results.columns]
    sort_col = st.selectbox("Sort by", sort_opts, index=0, label_visibility="visible")
with ctrl2:
    sort_dir = st.radio("Order", ["Ascending", "Descending"], horizontal=True)

results_sorted = results[DISPLAY_COLS].sort_values(
    sort_col, ascending=(sort_dir == "Ascending"), na_position="last"
)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE  with CO2 Class badge column
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='font-size:.7rem;font-weight:600;color:{muted};"
    "text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem'>"
    "{n:,} result{s} \u00b7 sorted by {col} ({dir})</div>".format(
        muted=MUTED, n=len(results_sorted), col=sort_col,
        dir=sort_dir.lower(), s="s" if len(results_sorted) != 1 else "",
    ),
    unsafe_allow_html=True,
)

# Style numeric columns and highlight CO2 class
num_cols = [c for c in ["CO2 (g/km)", "Max Power (kW)", "Max Power (HP)",
                         "Mixed Conso (l/100km)", "Mass (kg)",
                         "HC (g/km)", "NOx (g/km)", "Particles (g/km)"] if c in results_sorted.columns]

styled = results_sorted.style.format(
    {c: "{:.1f}" for c in num_cols if c != "Max Power (HP)"},
    na_rep="\u2013"
)

def _co2_bg(val):
    cls = co2_class(val)
    bg  = CLASS_COLOR.get(cls, "transparent")
    fg  = CLASS_TEXT.get(cls, "#111")
    return "background-color: {}; color: {}; font-weight: 600; border-radius: 4px".format(bg, fg)

if "CO2 (g/km)" in results_sorted.columns:
    styled = styled.applymap(_co2_bg, subset=["CO2 (g/km)"])

st.dataframe(styled, use_container_width=True, hide_index=True, height=480)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## Compare Vehicles")
st.caption("Select up to 3 vehicles to compare side-by-side. Identified by Brand + Model + Fuel.")

# Build unique identifier label for each row
id_col = results_sorted.apply(
    lambda r: "{} {} \u2014 {}".format(r["Brand"], r.get("Folder Model", ""), r.get("Fuel Label", "")),
    axis=1,
)
results_sorted = results_sorted.copy()
results_sorted["_id"] = id_col.values

options = results_sorted["_id"].unique().tolist()
compare_sel = st.multiselect(
    "Select vehicles",
    options=options,
    default=[],
    max_selections=3,
    placeholder="Choose up to 3 vehicles\u2026",
)

if compare_sel:
    compare_rows = results_sorted[results_sorted["_id"].isin(compare_sel)].drop(columns=["_id"])
    compare_rows = compare_rows.drop_duplicates(subset=["Brand", "Folder Model", "Fuel Label"] if "Fuel Label" in compare_rows.columns else ["Brand", "Folder Model"])

    # --- CO₂ Label cards ---
    cols_cmp = st.columns(len(compare_rows))
    for col_ui, (_, row) in zip(cols_cmp, compare_rows.iterrows()):
        co2_val  = row.get("CO2 (g/km)", float("nan"))
        cls      = co2_class(co2_val)
        bg       = CLASS_COLOR.get(cls, CARD)
        fg       = CLASS_TEXT.get(cls,  TEXT)
        conso    = row.get("Mixed Conso (l/100km)", float("nan"))
        power_kw = row.get("Max Power (kW)", float("nan"))
        power_hp = row.get("Max Power (HP)", float("nan"))
        mass     = row.get("Mass (kg)", float("nan"))
        norm     = row.get("Euro Norm", "\u2013")

        def _fmt(v, dec=1, unit=""):
            return "{:.{}f}{}".format(v, dec, unit) if pd.notna(v) else "\u2013"

        col_ui.markdown(
            # header: brand + model
            "<div style='background:{card};border:1px solid {border};border-radius:14px;overflow:hidden'>"
            # CO2 label banner
            "<div style='background:{bg};padding:1.25rem 1rem;text-align:center'>"
            "<div style='font-size:.6rem;font-weight:700;color:{fg};text-transform:uppercase;"
            "letter-spacing:.14em;opacity:.8'>CO\u2082 Emission Class</div>"
            "<div style='font-size:4rem;font-weight:900;color:{fg};line-height:1;margin:.2rem 0'>{cls}</div>"
            "<div style='font-size:1.4rem;font-weight:700;color:{fg}'>{co2} g/km</div>"
            "</div>"
            # specs
            "<div style='padding:1rem 1.1rem'>"
            "<div style='font-size:.75rem;font-weight:700;color:{text};margin-bottom:.6rem'>"
            "{brand}<br><span style='font-weight:500;color:{muted}'>{model}</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:.78rem'>"
            "<tr><td style='color:{muted};padding:.2rem 0'>Fuel</td>"
            "<td style='color:{text};font-weight:500;text-align:right'>{fuel}</td></tr>"
            "<tr><td style='color:{muted};padding:.2rem 0'>Consumption</td>"
            "<td style='color:{text};font-weight:500;text-align:right'>{conso}</td></tr>"
            "<tr><td style='color:{muted};padding:.2rem 0'>Power</td>"
            "<td style='color:{text};font-weight:500;text-align:right'>{power}</td></tr>"
            "<tr><td style='color:{muted};padding:.2rem 0'>Mass</td>"
            "<td style='color:{text};font-weight:500;text-align:right'>{mass}</td></tr>"
            "<tr><td style='color:{muted};padding:.2rem 0'>Gearbox</td>"
            "<td style='color:{text};font-weight:500;text-align:right'>{gear}</td></tr>"
            "<tr><td style='color:{muted};padding:.2rem 0'>Euro Norm</td>"
            "<td style='color:{text};font-weight:500;text-align:right'>{norm}</td></tr>"
            "</table></div></div>".format(
                card=CARD, border=BORDER, bg=bg, fg=fg, text=TEXT, muted=MUTED,
                cls=cls,
                co2=_fmt(co2_val),
                brand=row.get("Brand", "\u2013"),
                model=row.get("Folder Model", "\u2013"),
                fuel=row.get("Fuel Label", "\u2013"),
                conso=_fmt(conso, 1, " l/100km") if pd.notna(conso) else "\u2013",
                power="{} kW / {} HP".format(_fmt(power_kw, 0), _fmt(power_hp, 0)) if pd.notna(power_kw) else "\u2013",
                mass=_fmt(mass, 0, " kg"),
                gear="{} {}".format(row.get("Gear Type", ""), row.get("Gear Count", "")).strip() if "Gear Type" in row else "\u2013",
                norm=str(norm),
            ),
            unsafe_allow_html=True,
        )

    # --- Pollutants comparison bar chart ---
    poll_cols = [c for c in ("HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)", "Particles (g/km)") if c in compare_rows.columns]
    if poll_cols:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        with st.expander("Pollutant details"):
            poll_data = compare_rows[["Brand", "Folder Model"] + poll_cols].copy()
            poll_data["Vehicle"] = poll_data["Brand"] + " " + poll_data["Folder Model"]
            poll_data = poll_data.drop(columns=["Brand","Folder Model"]).set_index("Vehicle")
            st.dataframe(
                poll_data.style.format("{:.4f}", na_rep="\u2013")
                .background_gradient(cmap="RdYlGn_r", axis=None),
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# CO₂ DISTRIBUTION CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("## CO\u2082 Distribution in Results")

if "CO2 (g/km)" in results.columns:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "figure.facecolor": CARD, "axes.facecolor": CARD,
        "axes.edgecolor": BORDER, "text.color": TEXT,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": BORDER, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "savefig.facecolor": CARD,
    })

    co2_vals = results["CO2 (g/km)"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    fig.patch.set_facecolor(CARD)

    # Histogram coloured by CO2 class
    bins = np.arange(0, co2_vals.max() + 15, 10)
    for cls, (lo, hi) in CO2_CLASSES.items():
        band = co2_vals[(co2_vals >= lo) & (co2_vals <= hi)]
        if len(band):
            axes[0].hist(band, bins=bins, color=CLASS_COLOR[cls], alpha=0.9, label=cls, edgecolor=CARD, lw=0.3)
    axes[0].axvline(co2_vals.median(), color=TEXT, lw=1.5, linestyle="--",
                    label="Median {:.0f}".format(co2_vals.median()))
    axes[0].set_xlabel("CO\u2082 (g/km)", fontsize=9)
    axes[0].set_ylabel("Count", fontsize=9)
    axes[0].set_title("CO\u2082 Distribution by Efficiency Class", fontsize=10, fontweight="600")
    axes[0].legend(fontsize=7, ncol=4, framealpha=0.3)

    # Class breakdown bar
    class_counts = results["CO2 Class"].value_counts().reindex(["A","B","C","D","E","F","G"]).fillna(0)
    bars = axes[1].bar(class_counts.index, class_counts.values,
                       color=[CLASS_COLOR[c] for c in class_counts.index],
                       edgecolor=CARD, lw=0, width=0.65)
    for bar, (cls, cnt) in zip(bars, class_counts.items()):
        if cnt > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, cnt + 0.5,
                         str(int(cnt)), ha="center", va="bottom",
                         fontsize=8, color=TEXT, fontweight="600")
    axes[1].set_xlabel("CO\u2082 Class", fontsize=9)
    axes[1].set_ylabel("Count", fontsize=9)
    axes[1].set_title("Breakdown by Efficiency Class", fontsize=10, fontweight="600")

    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:.68rem;color:{muted};display:flex;justify-content:space-between'>"
    "<span>ADEME Car Labelling Dataset 2013 \u00b7 {:,} total vehicles</span>"
    "<span><a href='https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction' "
    "style='color:{mint};text-decoration:none'>GitHub \u2197</a></span>"
    "</div>".format(len(df), muted=MUTED, mint=MINT),
    unsafe_allow_html=True,
)
