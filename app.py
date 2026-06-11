"""
search_app.py  –  CO₂ Vehicle Search
Replicates the ADEME Car Labelling portal (carlabelling.ademe.fr).
Self-contained – no external module imports.
"""
from __future__ import annotations
import io, urllib.request, warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)

COLUMN_MAPPING = {
    "Marque": "Brand",
    "Modèle dossier": "Model",
    "Désignation commerciale": "Commercial Name",
    "Carburant": "_fuel_code",
    "Puissance maximale (kW)": "Max Power (kW)",
    "Puissance administrative": "Fiscal Power",
    "Boîte de vitesse": "_gearbox_raw",
    "Consommation urbaine (l/100km)": "Urban Conso",
    "Consommation extra-urbaine (l/100km)": "Extra-Urban Conso",
    "Consommation mixte (l/100km)": "Conso Min",   # dataset has one value; we use as min
    "CO2 (g/km)": "CO2",
    "CO type I (g/km)": "CO (g/km)",
    "HC (g/km)": "HC (g/km)",
    "NOX (g/km)": "NOx (g/km)",
    "HC+NOX (g/km)": "HC+NOx (g/km)",
    "Particules (g/km)": "Particles (g/km)",
    "masse vide euro min (kg)": "_mass_min",
    "masse vide euro max (kg)": "_mass_max",
    "Carrosserie": "Body",
    "gamme": "Size",
    "Hybride": "_hybrid",
    "Champ V9": "Euro Norm",
    "Date de mise à jour": "Updated",
}

FUEL_MAP = {
    "ES":"Petrol (ES)", "GO":"Diesel (GO)", "EL":"Electric (EL)",
    "GH":"Non-plug-in hybrid (GH)", "EH":"Non-plug-in hybrid (EH)",
    "EE":"Plug-in hybrid (EE)", "GL":"Plug-in hybrid (GL)",
    "GP":"LPG", "GN":"NGV", "FE":"Superethanol-E85 (FE)",
}
FUEL_GROUP = {
    "Electric (EL)":           ["EL"],
    "Non-plug-in hybrid":      ["EH","GH"],
    "Plug-in hybrid":          ["EE","GL"],
    "Petrol (ES)":             ["ES"],
    "Diesel (GO)":             ["GO"],
    "Superethanol-E85 (FE)":   ["FE"],
    "LPG":                     ["GP"],
    "NGV":                     ["GN"],
}

GEAR_MAP = {"M":"Manual","A":"Automatic","V":"CVT","D":"DCT","N":"Automatic","S":"Manual"}

# EU CO₂ label classes (NEDC thresholds – matches 2013 dataset)
CO2_CLASSES = [
    ("A", 0,   100, "#1a8c3c"),
    ("B", 101, 120, "#4db84b"),
    ("C", 121, 140, "#b2d145"),
    ("D", 141, 160, "#f9e000"),
    ("E", 161, 200, "#e07b00"),
    ("F", 201, 250, "#d03200"),
    ("G", 251, 9999,"#a00000"),
]
CLASS_FG = {"A":"#fff","B":"#fff","C":"#111","D":"#111","E":"#fff","F":"#fff","G":"#fff"}

def _co2_class(v):
    if pd.isna(v): return "?"
    for cls, lo, hi, _ in CO2_CLASSES:
        if lo <= v <= hi: return cls
    return "G"

def _cls_color(cls):
    for c, _, _, col in CO2_CLASSES:
        if c == cls: return col
    return "#888"

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    with urllib.request.urlopen(CSV_URL) as r:
        raw = r.read()
    df = None
    for enc in ("latin1","utf-8","cp1252"):
        for sep in (";",","):
            try:
                tmp = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if tmp.shape[1] > 5: df = tmp; break
            except: continue
        if df is not None: break
    if df is None: raise ValueError("Cannot parse CSV")

    df = df.rename(columns={k:v for k,v in COLUMN_MAPPING.items() if k in df.columns})

    # HC/NOx imputation
    if all(c in df.columns for c in ("HC (g/km)","NOx (g/km)","HC+NOx (g/km)")):
        hc  = (df["HC+NOx (g/km)"] - df["NOx (g/km)"]).fillna(df["HC (g/km)"])
        nox = (df["HC+NOx (g/km)"] - df["HC (g/km)"]).fillna(df["NOx (g/km)"])
        df["HC (g/km)"], df["NOx (g/km)"] = hc, nox
        df["HC+NOx (g/km)"] = hc + nox

    # Gearbox
    if "_gearbox_raw" in df.columns:
        df["_gearbox_raw"] = df["_gearbox_raw"].replace({"N 0":"A 0","N 1":"A 0","S 6":"D 6"})
        gs = df["_gearbox_raw"].astype(str).str.split(" ", expand=True)
        df["Gearbox"] = gs[0].map(GEAR_MAP).fillna(gs[0])

    # EV zeros
    ev_cols = ["CO2","Conso Min","Urban Conso","Extra-Urban Conso",
               "HC (g/km)","NOx (g/km)","Particles (g/km)","CO (g/km)"]
    if "_fuel_code" in df.columns:
        mask_ev = df["_fuel_code"] == "EL"
        for c in ev_cols:
            if c in df.columns: df.loc[mask_ev, c] = df.loc[mask_ev, c].fillna(0)

    # Mass avg
    if "_mass_min" in df.columns and "_mass_max" in df.columns:
        df["Mass (kg)"] = (
            pd.to_numeric(df["_mass_min"], errors="coerce") +
            pd.to_numeric(df["_mass_max"], errors="coerce")
        ) / 2

    # Numeric
    for c in ("CO2","Conso Min","Max Power (kW)","Mass (kg)",
              "HC (g/km)","NOx (g/km)","Particles (g/km)","CO (g/km)"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived
    if "_fuel_code" in df.columns:
        df["Energy"] = df["_fuel_code"].map(FUEL_MAP).fillna(df["_fuel_code"])
    if "CO2" in df.columns:
        df["CO2 Class"] = df["CO2"].apply(_co2_class)
        df["Conso Max"] = df["Conso Min"]   # 2013 dataset has single conso value

    return df

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CO\u2082 Vehicle Search · ADEME",
    page_icon="\U0001f697",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── ADEME-style CSS (light theme, blue accents) ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
html, body, .stApp { background: #f0f0f0 !important; font-family: 'Open Sans', sans-serif !important; color: #333 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
h1,h2,h3,h4 { font-family: 'Open Sans', sans-serif !important; }
p, li, label, span { color: #333 !important; }
/* hide default streamlit metric styling */
[data-testid="stMetric"] { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stMetricLabel"] p { font-size: .7rem !important; color: #666 !important; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; color: #1a5276 !important; font-family: 'Open Sans', sans-serif !important; }
[data-testid="stDataFrame"] { border-radius: 0 !important; }
[data-testid="stExpander"] { background: #fff !important; border: 1px solid #ddd !important; border-radius: 4px !important; }
[data-testid="stExpander"] summary { color: #1a5276 !important; font-weight: 600 !important; }
.stMultiSelect > div > div, .stSelectbox > div > div {
  background: #fff !important; border: 1px solid #ccc !important;
  border-radius: 3px !important; color: #333 !important;
}
[data-testid="stMultiSelect"] label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] label, [data-testid="stCheckbox"] label {
  color: #1a5276 !important; font-weight: 700 !important;
  font-size: .82rem !important; text-transform: none !important; letter-spacing: 0 !important;
}
[data-testid="stSlider"] > div > div > div > div { background: #1a5276 !important; }
.stButton > button {
  background: #1a5276 !important; color: #fff !important;
  border: none !important; border-radius: 3px !important;
  font-weight: 700 !important; font-size: .82rem !important;
  padding: .45rem 1.1rem !important; letter-spacing: .04em !important;
}
.stButton > button:hover { background: #154360 !important; }
hr { border: none !important; border-top: 1px solid #ddd !important; margin: .75rem 0 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading ADEME dataset\u2026"):
    df = load_data()

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR  (mimics ADEME header)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:#fff;border-bottom:3px solid #c0392b;padding:.6rem 2rem;
            display:flex;align-items:center;gap:1.5rem'>
  <div>
    <span style='font-size:1.25rem;font-weight:700;color:#c0392b;letter-spacing:-.02em'>Car</span>
    <span style='font-size:1.25rem;font-weight:700;color:#1a5276'> Labelling</span>
    <div style='font-size:.6rem;color:#888;text-transform:uppercase;letter-spacing:.1em;margin-top:-.1rem'>
      V&eacute;hicules particuliers &mdash; ADEME Dataset 2013
    </div>
  </div>
  <div style='height:2rem;width:1px;background:#ddd'></div>
  <nav style='display:flex;gap:2rem;font-size:.78rem;font-weight:600;color:#555'>
    <span style='color:#1a5276;border-bottom:2px solid #1a5276;padding-bottom:.1rem'>Search</span>
  </nav>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH PANEL  (white card, two columns like ADEME)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:#fff;margin:1rem 2rem;padding:1.25rem 1.5rem;
            border:1px solid #ddd;border-radius:4px'>
  <div style='display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem'>
    <div style='background:#1a5276;border-radius:3px;width:2.2rem;height:2.2rem;
                display:flex;align-items:center;justify-content:center;
                font-size:1.1rem'>&#128269;</div>
    <span style='font-size:.95rem;font-weight:700;color:#1a5276;text-transform:uppercase;
                 letter-spacing:.06em'>Multi-Criteria Search</span>
  </div>
""", unsafe_allow_html=True)

with st.container():
    col_left, col_right = st.columns([1, 1.4], gap="large")

    # ── LEFT: Brand / Model / Body / Size ─────────────────────────────────────
    with col_left:
        brands_all = sorted(df["Brand"].dropna().unique())
        sel_brand  = st.selectbox("Brand", ["Choose\u2026"] + brands_all, index=0)

        if sel_brand != "Choose\u2026":
            models_pool = sorted(df[df["Brand"] == sel_brand]["Model"].dropna().unique())
        else:
            models_pool = sorted(df["Model"].dropna().unique())
        sel_model = st.selectbox("Model", ["Choose\u2026"] + models_pool, index=0)

        bodies_all = sorted(df["Body"].dropna().unique())
        sel_body   = st.selectbox("Body", ["Choose\u2026"] + bodies_all, index=0)

        if "Size" in df.columns:
            sizes_all = sorted(df["Size"].dropna().unique())
            sel_size  = st.selectbox("Size", ["Choose\u2026"] + sizes_all, index=0)
        else:
            sel_size = "Choose\u2026"

    # ── RIGHT: Energy checkboxes + Gearbox + Sliders ──────────────────────────
    with col_right:
        st.markdown(
            "<div style='font-weight:700;color:#1a5276;font-size:.82rem;margin-bottom:.4rem'>Energy</div>",
            unsafe_allow_html=True,
        )
        ec1, ec2, ec3 = st.columns(3)
        fuel_sel = {}
        energy_groups = list(FUEL_GROUP.keys())
        for i, grp in enumerate(energy_groups):
            col_ui = [ec1, ec2, ec3][i % 3]
            fuel_sel[grp] = col_ui.checkbox(grp, value=False)

        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        gc1, gc2 = st.columns(2)
        with gc1:
            if "Gearbox" in df.columns:
                gear_opts = sorted(df["Gearbox"].dropna().unique())
                sel_gear  = st.selectbox("Gearbox", ["Choose\u2026"] + gear_opts, index=0)
            else:
                sel_gear = "Choose\u2026"

        with gc2:
            conso_max_all = float(df["Conso Min"].dropna().max()) if "Conso Min" in df.columns else 30.0
            conso_until = st.number_input(
                "Max combined consumption (l/100 km)", min_value=0.0,
                max_value=conso_max_all, value=conso_max_all, step=0.5, format="%.1f",
            )

        # CO2 class slider (A=0 … G=6)
        CLASS_LETTERS = ["A","B","C","D","E","F","G"]
        co2_cls_range = st.select_slider(
            "Energy Class / CO\u2082",
            options=CLASS_LETTERS,
            value=("A","G"),
        )

# close the white card div
st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS
# ══════════════════════════════════════════════════════════════════════════════
mask = pd.Series(True, index=df.index)

if sel_brand != "Choose\u2026":
    mask &= df["Brand"] == sel_brand
if sel_model != "Choose\u2026":
    mask &= df["Model"] == sel_model
if sel_body != "Choose\u2026":
    mask &= df["Body"] == sel_body
if sel_size != "Choose\u2026" and "Size" in df.columns:
    mask &= df["Size"] == sel_size
if sel_gear != "Choose\u2026" and "Gearbox" in df.columns:
    mask &= df["Gearbox"] == sel_gear

# Fuel checkboxes — if none ticked, show all
active_fuels = [code for grp, ticked in fuel_sel.items()
                if ticked for code in FUEL_GROUP[grp]]
if active_fuels and "_fuel_code" in df.columns:
    mask &= df["_fuel_code"].isin(active_fuels)

# Consumption filter
if "Conso Min" in df.columns:
    mask &= (df["Conso Min"] <= conso_until) | df["Conso Min"].isna()

# CO2 class filter
cls_lo_idx = CLASS_LETTERS.index(co2_cls_range[0])
cls_hi_idx = CLASS_LETTERS.index(co2_cls_range[1])
sel_classes = CLASS_LETTERS[cls_lo_idx : cls_hi_idx + 1]
if "CO2 Class" in df.columns:
    mask &= df["CO2 Class"].isin(sel_classes) | (df["CO2 Class"] == "?")

results = df[mask].copy()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS HEADER BAR  (matches ADEME "436 new vehicles" strip)
# ══════════════════════════════════════════════════════════════════════════════
n_res = len(results)
st.markdown(
    "<div style='background:#fff;margin:0 2rem;padding:.8rem 1.25rem;"
    "border:1px solid #ddd;border-top:none;"
    "display:flex;align-items:center;gap:1.5rem'>"
    "<div style='display:flex;align-items:center;gap:.75rem'>"
    "<span style='font-size:1.6rem'>&#128663;</span>"
    "<div><div style='font-size:1.3rem;font-weight:700;color:#1a5276'>"
    "{n:,} vehicle{s}</div>"
    "<div style='font-size:.75rem;color:#888'>match your search</div></div>"
    "</div>"
    "</div>".format(n=n_res, s="s" if n_res != 1 else ""),
    unsafe_allow_html=True,
)

if n_res == 0:
    st.markdown(
        "<div style='background:#fff;margin:0 2rem;padding:2rem;text-align:center;"
        "border:1px solid #ddd;border-top:none;color:#888'>"
        "<div style='font-size:1.1rem;font-weight:600;color:#555;margin-bottom:.4rem'>"
        "No results found</div>"
        "Try adjusting your filters above.</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
# Sort controls
with st.container():
    sc1, sc2, sc3 = st.columns([1.2, 1, 5])
    with sc1:
        sort_options = {
            "CO\u2082 (asc)":        ("CO2", True),
            "CO\u2082 (desc)":       ("CO2", False),
            "Brand (A\u2192Z)":      ("Brand", True),
            "Consumption (asc)":     ("Conso Min", True),
            "Max Power (asc)":       ("Max Power (kW)", True),
        }
        sort_lbl = st.selectbox("Sort by", list(sort_options.keys()), index=0, label_visibility="visible")
    sort_col, sort_asc = sort_options[sort_lbl]

# Build display table
SHOW_COLS = [c for c in [
    "Body", "Brand", "Model",
    "Energy",
    "Conso Min", "Conso Max",
    "CO2", "CO2 Class",
    "HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)", "Particles (g/km)",
    "Euro Norm",
    "Gearbox", "Max Power (kW)", "Mass (kg)",
] if c in results.columns]

if sort_col in results.columns:
    display = results[SHOW_COLS].sort_values(sort_col, ascending=sort_asc, na_position="last").copy()
else:
    display = results[SHOW_COLS].copy()

# ── Render table as HTML to get ADEME-style CO2 badge cells ──────────────────
def _badge(cls):
    if cls in ("?", ""): return cls
    bg  = _cls_color(cls)
    fg  = CLASS_FG.get(cls, "#fff")
    # Arrow-shaped badge via CSS clip-path
    return (
        "<span style='display:inline-block;background:{bg};color:{fg};"
        "font-weight:700;font-size:.75rem;padding:.15rem .55rem .15rem .4rem;"
        "border-radius:2px;min-width:1.8rem;text-align:center'>{cls}</span>"
    ).format(bg=bg, fg=fg, cls=cls)

def _fmt(v, dec=1):
    return "{:.{}f}".format(v, dec) if pd.notna(v) else "\u2013"

def build_html_table(df_t: pd.DataFrame) -> str:
    th = "style='background:#4a6741;color:#fff;font-size:.73rem;font-weight:600;" \
         "padding:.45rem .5rem;text-align:center;white-space:nowrap;border:1px solid #5a7751'"
    th2= "style='background:#5a6e36;color:#fff;font-size:.73rem;font-weight:600;" \
         "padding:.45rem .5rem;text-align:center;white-space:nowrap;border:1px solid #6a7e46'"
    td = "style='padding:.35rem .5rem;font-size:.78rem;border:1px solid #e8e8e8;" \
         "vertical-align:middle;text-align:center'"
    td_l="style='padding:.35rem .6rem;font-size:.78rem;border:1px solid #e8e8e8;" \
         "vertical-align:middle;text-align:left'"

    rows = ["<div style='margin:0 2rem;overflow-x:auto'>",
            "<table style='width:100%;border-collapse:collapse;background:#fff'>",
            "<thead>",
            "<tr>",
            "<th {th}>Body</th>".format(th=th),
            "<th {th}>Brand / Model</th>".format(th=th),
            "<th {th}>Energy</th>".format(th=th),
            "<th {th} colspan='2'>Consumption<br><span style='font-weight:400;font-size:.65rem'>(l/100km)</span></th>".format(th=th),
            "<th {th} colspan='3'>CO\u2082 (g/km)</th>".format(th=th),
            "<th {th2}>CO</th>".format(th2=th2),
            "<th {th2}>HC</th>".format(th2=th2),
            "<th {th2}>NO\u2093</th>".format(th2=th2),
            "<th {th2}>HC+NO\u2093</th>".format(th2=th2),
            "<th {th2}>Particles</th>".format(th2=th2),
            "<th {th}>Euro</th>".format(th=th),
            "</tr>",
            "<tr style='background:#3a5731'>",
            "<th {th}></th>".format(th=th),
            "<th {th}></th>".format(th=th),
            "<th {th}></th>".format(th=th),
            "<th {th}>Min.</th>".format(th=th),
            "<th {th}>Max.</th>".format(th=th),
            "<th {th}>Min. g/km</th>".format(th=th),
            "<th {th}>Max. g/km</th>".format(th=th),
            "<th {th}>Class</th>".format(th=th),
            "<th {th2}></th>".format(th2=th2),
            "<th {th2}></th>".format(th2=th2),
            "<th {th2}></th>".format(th2=th2),
            "<th {th2}></th>".format(th2=th2),
            "<th {th2}></th>".format(th2=th2),
            "<th {th}></th>".format(th=th),
            "</tr>",
            "</thead><tbody>",
    ]

    for i, (_, r) in enumerate(df_t.iterrows()):
        row_bg = "#fff" if i % 2 == 0 else "#f7f7f5"
        tr = "<tr style='background:{bg}'>".format(bg=row_bg)
        rows.append(tr)

        body   = r.get("Body", "\u2013") or "\u2013"
        brand  = r.get("Brand", "\u2013") or "\u2013"
        model  = r.get("Model", "\u2013") or "\u2013"
        energy = r.get("Energy", "\u2013") or "\u2013"
        cmin   = _fmt(r.get("Conso Min"))
        cmax   = _fmt(r.get("Conso Max"))
        co2v   = r.get("CO2")
        co2min = _fmt(co2v)
        co2max = _fmt(co2v)     # same value in 2013 dataset
        cls    = r.get("CO2 Class", "?")
        badge  = _badge(cls)

        rows += [
            "<td {td}><span style='font-size:.72rem;color:#555'>{v}</span></td>".format(td=td, v=body),
            "<td {td}><span style='font-weight:700;font-size:.8rem'>{br}</span><br>"
            "<span style='font-size:.72rem;color:#555'>{mo}</span></td>".format(td=td_l, br=brand, mo=model),
            "<td {td}><span style='font-size:.72rem'>{v}</span></td>".format(td=td, v=energy),
            "<td {td}>{v}</td>".format(td=td, v=cmin),
            "<td {td}>{v}</td>".format(td=td, v=cmax),
            "<td {td}>{v}</td>".format(td=td, v=co2min),
            "<td {td}>{v}</td>".format(td=td, v=co2max),
            "<td {td}>{v}</td>".format(td=td, v=badge),
        ]
        for col in ("CO (g/km)","HC (g/km)","NOx (g/km)","HC+NOx (g/km)","Particles (g/km)"):
            v = _fmt(r.get(col), 4) if col != "CO (g/km)" else _fmt(r.get(col), 3)
            rows.append("<td {td}>{v}</td>".format(td=td, v=v))

        norm = str(r.get("Euro Norm","")) or "\u2013"
        rows.append("<td {td}><span style='font-size:.72rem'>{v}</span></td>".format(td=td, v=norm))
        rows.append("</tr>")

    rows += ["</tbody></table></div>"]
    return "\n".join(rows)

PAGE_SIZE = 50
if "tbl_page" not in st.session_state:
    st.session_state["tbl_page"] = 0

total_pages = max(1, (len(display) - 1) // PAGE_SIZE + 1)
page = st.session_state["tbl_page"]
page = max(0, min(page, total_pages - 1))

page_df = display.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

# Pagination controls
pag1, pag2, pag3 = st.columns([1, 2, 1])
with pag1:
    if st.button("\u2190 Previous", disabled=(page == 0)):
        st.session_state["tbl_page"] = page - 1
        st.rerun()
with pag2:
    st.markdown(
        "<div style='text-align:center;font-size:.78rem;color:#666;padding:.5rem 0'>"
        "Page {cur} of {tot} \u00b7 showing {a}\u2013{b} of {n:,} results</div>".format(
            cur=page+1, tot=total_pages,
            a=page*PAGE_SIZE+1, b=min((page+1)*PAGE_SIZE, n_res), n=n_res,
        ), unsafe_allow_html=True,
    )
with pag3:
    if st.button("Next \u2192", disabled=(page >= total_pages - 1)):
        st.session_state["tbl_page"] = page + 1
        st.rerun()

st.markdown(build_html_table(page_df), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE  (up to 3 vehicles)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
with st.expander("\u21c6  Compare vehicles (up to 3)", expanded=False):
    id_labels = (
        display["Brand"].fillna("") + " " +
        display["Model"].fillna("") + " \u2014 " +
        display.get("Energy", pd.Series([""] * len(display))).fillna("")
    ).tolist()
    # deduplicate labels
    seen, unique_labels = {}, []
    for lbl in id_labels:
        key = lbl
        cnt = seen.get(key, 0)
        seen[key] = cnt + 1
        unique_labels.append(lbl if cnt == 0 else "{} ({})".format(lbl, cnt+1))
    display = display.copy()
    display["_label"] = unique_labels

    cmp_sel = st.multiselect(
        "Select vehicles to compare",
        options=unique_labels, default=[], max_selections=3,
        placeholder="Choose up to 3 vehicles\u2026",
    )
    if cmp_sel:
        cmp_rows = display[display["_label"].isin(cmp_sel)].drop(columns=["_label"])
        cols_cmp = st.columns(len(cmp_rows))
        for col_ui, (_, r) in zip(cols_cmp, cmp_rows.iterrows()):
            co2v  = r.get("CO2")
            cls   = r.get("CO2 Class","?")
            bg    = _cls_color(cls)
            fg    = CLASS_FG.get(cls,"#fff")
            conso = r.get("Conso Min")
            pw    = r.get("Max Power (kW)")
            mass  = r.get("Mass (kg)")

            def fv(v, d=1, u=""):
                return "{:.{}f}{}".format(v,d,u) if pd.notna(v) else "\u2013"

            col_ui.markdown(
                "<div style='border:1px solid #ddd;border-radius:4px;overflow:hidden'>"
                "<div style='background:{bg};padding:1.2rem 1rem;text-align:center'>"
                "<div style='font-size:.6rem;font-weight:700;color:{fg};text-transform:uppercase;"
                "letter-spacing:.12em;opacity:.85'>CO\u2082 Class</div>"
                "<div style='font-size:4rem;font-weight:900;color:{fg};line-height:1'>{cls}</div>"
                "<div style='font-size:1.3rem;font-weight:700;color:{fg}'>{co2} g/km</div>"
                "</div>"
                "<div style='background:#fff;padding:.9rem 1rem'>"
                "<div style='font-size:.85rem;font-weight:700;color:#1a5276'>{brand}</div>"
                "<div style='font-size:.78rem;color:#555;margin-bottom:.6rem'>{model}</div>"
                "<table style='width:100%;font-size:.76rem;border-collapse:collapse'>"
                "<tr><td style='color:#888;padding:.18rem 0'>Energy</td>"
                "<td style='font-weight:600;text-align:right'>{energy}</td></tr>"
                "<tr><td style='color:#888;padding:.18rem 0'>Consumption</td>"
                "<td style='font-weight:600;text-align:right'>{conso}</td></tr>"
                "<tr><td style='color:#888;padding:.18rem 0'>Max Power</td>"
                "<td style='font-weight:600;text-align:right'>{pw}</td></tr>"
                "<tr><td style='color:#888;padding:.18rem 0'>Mass</td>"
                "<td style='font-weight:600;text-align:right'>{mass}</td></tr>"
                "<tr><td style='color:#888;padding:.18rem 0'>Gearbox</td>"
                "<td style='font-weight:600;text-align:right'>{gear}</td></tr>"
                "</table></div></div>".format(
                    bg=bg, fg=fg, cls=cls,
                    co2=fv(co2v),
                    brand=r.get("Brand","\u2013"),
                    model=r.get("Model","\u2013"),
                    energy=r.get("Energy","\u2013"),
                    conso=fv(conso, 1, " l/100km"),
                    pw=fv(pw, 0, " kW"),
                    mass=fv(mass, 0, " kg"),
                    gear=r.get("Gearbox","\u2013"),
                ),
                unsafe_allow_html=True,
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='background:#fff;border-top:1px solid #ddd;margin-top:2rem;"
    "padding:.6rem 2rem;font-size:.68rem;color:#888;display:flex;"
    "justify-content:space-between'>"
    "<span>ADEME Car Labelling Dataset 2013 \u00b7 {:,} vehicles</span>"
    "<span><a href='https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction' "
    "style='color:#1a5276;text-decoration:none'>GitHub \u2197</a></span>"
    "</div>".format(len(df)),
    unsafe_allow_html=True,
)
