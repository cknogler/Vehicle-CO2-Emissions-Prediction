"""
search_app.py  –  CO₂ Vehicle Search
ADEME Car Labelling portal replica with clickable detail pages.
Self-contained – no external module imports.
"""
from __future__ import annotations
import io, urllib.request, warnings
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
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
    "Puissance administrative": "Fiscal Power (CV)",
    "Boîte de vitesse": "_gearbox_raw",
    "Consommation urbaine (l/100km)": "Urban Conso (l/100km)",
    "Consommation extra-urbaine (l/100km)": "Extra-Urban Conso (l/100km)",
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
    "gamme": "Size",
    "Hybride": "Hybrid",
    "Champ V9": "Euro Norm",
    "Date de mise à jour": "Updated",
    "Type Variante Version (TVV)": "TVV",
    "CNIT": "CNIT",
}

FUEL_MAP = {
    "ES": "Petrol (ES)", "GO": "Diesel (GO)", "EL": "Electric (EL)",
    "GH": "Non-plug-in hybrid (GH)", "EH": "Non-plug-in hybrid (EH)",
    "EE": "Plug-in hybrid (EE)", "GL": "Plug-in hybrid (GL)",
    "GP": "LPG", "GN": "NGV", "FE": "Superethanol-E85 (FE)",
}
FUEL_GROUP = {
    "Electric (EL)":          ["EL"],
    "Non-plug-in hybrid":     ["EH", "GH"],
    "Plug-in hybrid":         ["EE", "GL"],
    "Petrol (ES)":            ["ES"],
    "Diesel (GO)":            ["GO"],
    "Superethanol-E85 (FE)":  ["FE"],
    "LPG":                    ["GP"],
    "NGV":                    ["GN"],
}
GEAR_MAP = {"M": "Manual", "A": "Automatic", "V": "CVT", "D": "DCT", "N": "Automatic", "S": "Manual"}

CO2_CLASSES = [
    ("A", 0,    100,  "#1a8c3c"),
    ("B", 101,  120,  "#4db84b"),
    ("C", 121,  140,  "#b2d145"),
    ("D", 141,  160,  "#f9e000"),
    ("E", 161,  200,  "#e07b00"),
    ("F", 201,  250,  "#d03200"),
    ("G", 251,  9999, "#a00000"),
]
CLASS_FG = {"A": "#fff", "B": "#fff", "C": "#111", "D": "#111", "E": "#fff", "F": "#fff", "G": "#fff"}
CLASS_LETTERS = [c for c, *_ in CO2_CLASSES]

def _co2_class(v):
    if pd.isna(v): return "?"
    for cls, lo, hi, _ in CO2_CLASSES:
        if lo <= v <= hi: return cls
    return "G"

def _cls_color(cls):
    for c, _, _, col in CO2_CLASSES:
        if c == cls: return col
    return "#888"

def _fmt(v, dec=1, unit=""):
    if pd.isna(v) or v is None: return "\u2013"
    return "{:.{}f}{}".format(v, dec, unit)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    with urllib.request.urlopen(CSV_URL) as r:
        raw = r.read()
    df = None
    for enc in ("latin1", "utf-8", "cp1252"):
        for sep in (";", ","):
            try:
                tmp = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if tmp.shape[1] > 5: df = tmp; break
            except: continue
        if df is not None: break
    if df is None: raise ValueError("Cannot parse CSV")

    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    # HC/NOx imputation
    if all(c in df.columns for c in ("HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)")):
        hc  = (df["HC+NOx (g/km)"] - df["NOx (g/km)"]).fillna(df["HC (g/km)"])
        nox = (df["HC+NOx (g/km)"] - df["HC (g/km)"]).fillna(df["NOx (g/km)"])
        df["HC (g/km)"], df["NOx (g/km)"] = hc, nox
        df["HC+NOx (g/km)"] = hc + nox

    # Gearbox split
    if "_gearbox_raw" in df.columns:
        df["_gearbox_raw"] = df["_gearbox_raw"].replace({"N 0": "A 0", "N 1": "A 0", "S 6": "D 6"})
        gs = df["_gearbox_raw"].astype(str).str.split(" ", expand=True)
        df["Gearbox Type"] = gs[0].map(GEAR_MAP).fillna(gs[0])
        df["Gear Count"]   = pd.to_numeric(gs[1] if 1 in gs.columns else pd.Series(dtype=float), errors="coerce").astype("Int64")

    # EV zeros
    ev_cols = ["CO2 (g/km)", "Mixed Conso (l/100km)", "Urban Conso (l/100km)",
               "Extra-Urban Conso (l/100km)", "HC (g/km)", "NOx (g/km)", "Particles (g/km)", "CO (g/km)"]
    if "_fuel_code" in df.columns:
        for c in ev_cols:
            if c in df.columns:
                df.loc[df["_fuel_code"] == "EL", c] = df.loc[df["_fuel_code"] == "EL", c].fillna(0)

    # Average mass
    if "Mass Min (kg)" in df.columns and "Mass Max (kg)" in df.columns:
        df["Mass (kg)"] = (
            pd.to_numeric(df["Mass Min (kg)"], errors="coerce") +
            pd.to_numeric(df["Mass Max (kg)"], errors="coerce")
        ) / 2

    # Numeric coercion
    for c in ("CO2 (g/km)", "Mixed Conso (l/100km)", "Max Power (kW)", "Mass (kg)",
              "Urban Conso (l/100km)", "Extra-Urban Conso (l/100km)",
              "HC (g/km)", "NOx (g/km)", "Particles (g/km)", "CO (g/km)", "Fiscal Power (CV)"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived columns
    if "_fuel_code" in df.columns:
        df["Energy"] = df["_fuel_code"].map(FUEL_MAP).fillna(df["_fuel_code"])
    if "CO2 (g/km)" in df.columns:
        df["CO2 Class"] = df["CO2 (g/km)"].apply(_co2_class)
    if "Max Power (kW)" in df.columns:
        df["Max Power (HP)"] = (df["Max Power (kW)"] * 1.36).round(0).astype("Int64")

    # Stable integer index for row lookup
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index
    return df

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CO\u2082 Vehicle Search \u00b7 ADEME",
    page_icon="\U0001f697",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
html, body, .stApp { background: #f0f0f0 !important; font-family: 'Open Sans', sans-serif !important; color: #333 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
h1,h2,h3,h4 { font-family: 'Open Sans', sans-serif !important; }
p, li { color: #333 !important; }
[data-testid="stMetric"] { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stMetricLabel"] p { font-size: .7rem !important; color: #666 !important; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700 !important; color: #1a5276 !important; font-family: 'Open Sans', sans-serif !important; }
[data-testid="stDataFrame"] { border-radius: 0 !important; }
[data-testid="stExpander"] { background: #fff !important; border: 1px solid #ddd !important; border-radius: 4px !important; }
.stMultiSelect > div > div, .stSelectbox > div > div { background: #fff !important; border: 1px solid #ccc !important; border-radius: 3px !important; color: #333 !important; }
[data-testid="stMultiSelect"] label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] label, [data-testid="stCheckbox"] label {
  color: #1a5276 !important; font-weight: 700 !important; font-size: .82rem !important;
  text-transform: none !important; letter-spacing: 0 !important;
}
[data-testid="stSlider"] > div > div > div > div { background: #1a5276 !important; }
.stButton > button { background: #1a5276 !important; color: #fff !important; border: none !important; border-radius: 3px !important; font-weight: 700 !important; font-size: .82rem !important; padding: .45rem 1.1rem !important; }
.stButton > button:hover { background: #154360 !important; }
hr { border: none !important; border-top: 1px solid #ddd !important; margin: .75rem 0 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading ADEME dataset\u2026"):
    df = load_data()

# ── Session state init ─────────────────────────────────────────────────────────
if "selected_row_id" not in st.session_state:
    st.session_state["selected_row_id"] = None
if "tbl_page" not in st.session_state:
    st.session_state["tbl_page"] = 0

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
def render_topbar(show_back=False):
    back_btn = ""
    if show_back:
        back_btn = (
            "<span onclick=\"window.parent.postMessage({type:'back'}, '*')\" "
            "style='cursor:pointer;font-size:.78rem;color:#1a5276;font-weight:600;"
            "border:1px solid #1a5276;border-radius:3px;padding:.2rem .7rem;"
            "margin-left:auto;user-select:none'>&#8592; Back to results</span>"
        )
    st.markdown(
        "<div style='background:#fff;border-bottom:3px solid #c0392b;padding:.6rem 2rem;"
        "display:flex;align-items:center;gap:1.5rem'>"
        "<div>"
        "<span style='font-size:1.25rem;font-weight:700;color:#c0392b'>Car</span>"
        "<span style='font-size:1.25rem;font-weight:700;color:#1a5276'> Labelling</span>"
        "<div style='font-size:.6rem;color:#888;text-transform:uppercase;letter-spacing:.1em;margin-top:-.1rem'>"
        "V&eacute;hicules particuliers &mdash; ADEME Dataset 2013</div>"
        "</div>"
        "<div style='height:2rem;width:1px;background:#ddd'></div>"
        "{back}"
        "</div>".format(back=back_btn),
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_detail(row_id: int):
    render_topbar(show_back=True)

    # Intercept "back" message from the button
    back_html = """
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'back') {
            window.parent.postMessage({type:'streamlit:setComponentValue', value:'__back__'}, '*');
        }
    });
    </script>
    """
    back_val = components.html(back_html, height=0)

    r = df[df["_row_id"] == row_id].iloc[0]

    cls   = r.get("CO2 Class", "?")
    bg    = _cls_color(cls)
    fg    = CLASS_FG.get(cls, "#fff")
    co2v  = r.get("CO2 (g/km)")

    # ── Back button (Streamlit native) ──────────────────────────────────────
    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    bcol, _ = st.columns([1, 9])
    with bcol:
        if st.button("\u2190  Back to results"):
            st.session_state["selected_row_id"] = None
            st.rerun()

    # ── Breadcrumb ───────────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:.75rem;color:#888;padding:.3rem 2rem'>"
        "Home &rsaquo; Search &rsaquo; <strong>{brand} {model}</strong></div>".format(
            brand=r.get("Brand",""), model=r.get("Model","")
        ), unsafe_allow_html=True,
    )

    # ── Main card ────────────────────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        # CO₂ Label card (mimics physical EU label)
        bars = ""
        for c, lo, hi, col in CO2_CLASSES:
            width_map = {"A": 55, "B": 62, "C": 69, "D": 76, "E": 83, "F": 90, "G": 97}
            w = width_map.get(c, 70)
            active = "border:3px solid #000;transform:scaleY(1.12);" if c == cls else ""
            fg_c   = CLASS_FG.get(c, "#fff")
            bars += (
                "<div style='background:{col};width:{w}%;height:1.9rem;margin-bottom:2px;"
                "display:flex;align-items:center;justify-content:space-between;"
                "padding:0 .5rem;border-radius:2px;{active}'>"
                "<span style='font-weight:700;font-size:.85rem;color:{fg}'>{c}</span>"
                "<span style='font-size:.7rem;color:{fg};opacity:.85'>{lo}&ndash;{hi}</span>"
                "</div>"
            ).format(col=col, w=w, c=c, fg=fg_c, lo=lo, hi="+" if hi > 500 else hi, active=active)

        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;"
            "padding:1.25rem 1.5rem;margin-bottom:1rem'>"
            "<div style='font-size:.65rem;font-weight:700;color:#666;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.75rem;text-align:center'>"
            "CO\u2082 Emission Class</div>"
            "{bars}"
            "<div style='text-align:center;margin-top:1rem'>"
            "<span style='font-size:2.2rem;font-weight:900;color:{bg}'>{cls}</span>"
            "<span style='font-size:.9rem;color:#666;margin-left:.4rem'>{co2} g/km</span>"
            "</div></div>".format(bars=bars, bg=bg, cls=cls, co2=_fmt(co2v, 0)),
            unsafe_allow_html=True,
        )

        # Fuel energy cost indicator
        conso = r.get("Mixed Conso (l/100km)")
        if pd.notna(conso) and conso > 0:
            fuel_price = 1.85  # €/l approx
            annual_cost = conso / 100 * 15000 * fuel_price
            st.markdown(
                "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;"
                "padding:1rem 1.25rem'>"
                "<div style='font-size:.65rem;font-weight:700;color:#666;text-transform:uppercase;"
                "letter-spacing:.1em;margin-bottom:.5rem'>Energy Cost Estimate</div>"
                "<div style='font-size:1.4rem;font-weight:700;color:#1a5276'>"
                "&euro;{cost:.0f}/year</div>"
                "<div style='font-size:.7rem;color:#888;margin-top:.2rem'>"
                "Based on {conso:.1f} l/100km \u00b7 15,000 km \u00b7 &euro;{price:.2f}/l</div>"
                "</div>".format(cost=annual_cost, conso=conso, price=fuel_price),
                unsafe_allow_html=True,
            )

    with right:
        brand = str(r.get("Brand","")).upper()
        model = str(r.get("Model",""))
        cname = str(r.get("Commercial Name",""))

        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;padding:1.5rem 2rem'>"
            "<div style='font-size:1.6rem;font-weight:800;color:#1a5276;letter-spacing:-.02em'>{brand}</div>"
            "<div style='font-size:1.1rem;font-weight:600;color:#333;margin-bottom:.2rem'>{model}</div>"
            "<div style='font-size:.82rem;color:#888;margin-bottom:1.5rem'>{cname}</div>".format(
                brand=brand, model=model, cname=cname if cname and cname != "nan" else ""
            ), unsafe_allow_html=True,
        )

        def _section(title, rows_data):
            html = "<div style='margin-bottom:1.25rem'>"
            html += "<div style='font-size:.7rem;font-weight:700;color:#1a5276;text-transform:uppercase;letter-spacing:.1em;border-bottom:2px solid #1a5276;padding-bottom:.2rem;margin-bottom:.6rem'>{}</div>".format(title)
            html += "<table style='width:100%;border-collapse:collapse'>"
            for label, val, unit in rows_data:
                if val is None or (isinstance(val, float) and pd.isna(val)): continue
                html += (
                    "<tr>"
                    "<td style='font-size:.82rem;color:#666;padding:.3rem 0;width:55%'>{label}</td>"
                    "<td style='font-size:.82rem;font-weight:600;color:#222;text-align:right'>"
                    "{val} <span style='font-weight:400;color:#888;font-size:.75rem'>{unit}</span></td>"
                    "</tr>"
                ).format(label=label, val=val, unit=unit)
            html += "</table></div>"
            return html

        pw_kw = r.get("Max Power (kW)")
        pw_hp = r.get("Max Power (HP)")
        power_str = "{} kW ({} HP)".format(_fmt(pw_kw, 0), int(pw_hp)) if pd.notna(pw_kw) and pd.notna(pw_hp) else _fmt(pw_kw, 0)

        gear_str = "{} ({} gears)".format(
            str(r.get("Gearbox Type","")),
            str(int(r["Gear Count"])) if "Gear Count" in r.index and pd.notna(r.get("Gear Count")) else "?"
        ) if "Gearbox Type" in r.index else "\u2013"

        sections = [
            ("Technical Specs", [
                ("Body type",       str(r.get("Body","")),          ""),
                ("Segment",         str(r.get("Size","")),           ""),
                ("Energy / Fuel",   str(r.get("Energy","")),         ""),
                ("Max Power",       power_str,                       ""),
                ("Fiscal Power",    _fmt(r.get("Fiscal Power (CV)"),0), "CV"),
                ("Gearbox",         gear_str,                        ""),
                ("Kerb Mass",       _fmt(r.get("Mass (kg)"),0),      "kg"),
                ("Euro Norm",       str(r.get("Euro Norm","")),      ""),
            ]),
            ("Consumption", [
                ("Urban",           _fmt(r.get("Urban Conso (l/100km)")),      "l/100km"),
                ("Extra-Urban",     _fmt(r.get("Extra-Urban Conso (l/100km)")), "l/100km"),
                ("Combined (NEDC)", _fmt(r.get("Mixed Conso (l/100km)")),      "l/100km"),
            ]),
            ("CO\u2082 Emissions", [
                ("CO\u2082",        _fmt(co2v, 0),                    "g/km"),
                ("Efficiency class",cls,                              ""),
            ]),
            ("Pollutants", [
                ("CO",              _fmt(r.get("CO (g/km)"), 3),      "g/km"),
                ("HC",              _fmt(r.get("HC (g/km)"), 4),      "g/km"),
                ("NO\u2093",        _fmt(r.get("NOx (g/km)"), 4),     "g/km"),
                ("HC + NO\u2093",   _fmt(r.get("HC+NOx (g/km)"), 4),  "g/km"),
                ("Particles",       _fmt(r.get("Particles (g/km)"), 4), "g/km"),
            ]),
        ]
        if "TVV" in r.index or "CNIT" in r.index:
            sections.append(("Homologation", [
                ("TVV",  str(r.get("TVV","")),  ""),
                ("CNIT", str(r.get("CNIT","")), ""),
            ]))

        detail_html = ""
        for title, rows_data in sections:
            detail_html += _section(title, rows_data)
        detail_html += "</div>"   # close main card
        st.markdown(detail_html, unsafe_allow_html=True)

    # ── CO₂ comparison bar ───────────────────────────────────────────────────
    if pd.notna(co2v):
        fleet_med = float(df["CO2 (g/km)"].median())
        same_body = df[df["Body"] == r.get("Body","")]["CO2 (g/km)"].dropna()
        seg_med   = float(same_body.median()) if len(same_body) > 0 else None

        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;"
            "padding:1.25rem 1.5rem;margin:0 0 1rem'>"
            "<div style='font-size:.7rem;font-weight:700;color:#1a5276;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:1rem'>CO\u2082 in Context</div>",
            unsafe_allow_html=True,
        )

        comp_rows = [
            ("This vehicle",        co2v,     _cls_color(cls)),
            ("Fleet median",        fleet_med, "#888"),
        ]
        if seg_med is not None:
            comp_rows.insert(1, ("{} median".format(r.get("Body","")), seg_med, "#1a5276"))

        max_val = max(v for _, v, _ in comp_rows) * 1.15
        bar_html = ""
        for label, val, col in comp_rows:
            pct = val / max_val * 100
            bar_html += (
                "<div style='margin-bottom:.5rem'>"
                "<div style='display:flex;justify-content:space-between;"
                "font-size:.78rem;margin-bottom:.2rem'>"
                "<span style='color:#555'>{label}</span>"
                "<span style='font-weight:700;color:#222'>{val:.0f} g/km</span></div>"
                "<div style='background:#eee;border-radius:3px;height:.65rem'>"
                "<div style='background:{col};width:{pct:.1f}%;height:100%;border-radius:3px'></div>"
                "</div></div>"
            ).format(label=label, val=val, col=col, pct=pct)

        st.markdown(bar_html + "</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def render_search():
    render_topbar(show_back=False)

    # ── Search panel ──────────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:#fff;margin:1rem 2rem;padding:1.25rem 1.5rem;"
        "border:1px solid #ddd;border-radius:4px'>"
        "<div style='display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem'>"
        "<div style='background:#1a5276;border-radius:3px;width:2.2rem;height:2.2rem;"
        "display:flex;align-items:center;justify-content:center;font-size:1.1rem'>&#128269;</div>"
        "<span style='font-size:.95rem;font-weight:700;color:#1a5276;text-transform:uppercase;"
        "letter-spacing:.06em'>Multi-Criteria Search</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
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

    with col_r:
        st.markdown(
            "<div style='font-weight:700;color:#1a5276;font-size:.82rem;margin-bottom:.4rem'>Energy</div>",
            unsafe_allow_html=True,
        )
        ec1, ec2, ec3 = st.columns(3)
        fuel_sel = {}
        for i, grp in enumerate(FUEL_GROUP):
            fuel_sel[grp] = [ec1, ec2, ec3][i % 3].checkbox(grp, value=False, key="fuel_"+grp)

        st.markdown("<div style='height:.3rem'></div>", unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        with gc1:
            if "Gearbox Type" in df.columns:
                gear_opts = sorted(df["Gearbox Type"].dropna().unique())
                sel_gear  = st.selectbox("Gearbox", ["Choose\u2026"] + gear_opts, index=0)
            else:
                sel_gear = "Choose\u2026"
        with gc2:
            conso_max_all = float(df["Mixed Conso (l/100km)"].dropna().max()) if "Mixed Conso (l/100km)" in df.columns else 30.0
            conso_until   = st.number_input("Max consumption (l/100km)", min_value=0.0,
                                             max_value=conso_max_all, value=conso_max_all, step=0.5, format="%.1f")
        co2_cls_range = st.select_slider("CO\u2082 Class", options=CLASS_LETTERS, value=("A","G"))

    st.markdown("</div>", unsafe_allow_html=True)   # close search card

    # ── Filters ───────────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)
    if sel_brand != "Choose\u2026": mask &= df["Brand"] == sel_brand
    if sel_model != "Choose\u2026": mask &= df["Model"] == sel_model
    if sel_body  != "Choose\u2026": mask &= df["Body"]  == sel_body
    if sel_size  != "Choose\u2026" and "Size" in df.columns: mask &= df["Size"] == sel_size
    if sel_gear  != "Choose\u2026" and "Gearbox Type" in df.columns: mask &= df["Gearbox Type"] == sel_gear

    active_fuels = [code for grp, ticked in fuel_sel.items() if ticked for code in FUEL_GROUP[grp]]
    if active_fuels and "_fuel_code" in df.columns:
        mask &= df["_fuel_code"].isin(active_fuels)
    if "Mixed Conso (l/100km)" in df.columns:
        mask &= (df["Mixed Conso (l/100km)"] <= conso_until) | df["Mixed Conso (l/100km)"].isna()

    cls_lo = CLASS_LETTERS.index(co2_cls_range[0])
    cls_hi = CLASS_LETTERS.index(co2_cls_range[1])
    sel_classes = CLASS_LETTERS[cls_lo : cls_hi + 1]
    if "CO2 Class" in df.columns:
        mask &= df["CO2 Class"].isin(sel_classes) | (df["CO2 Class"] == "?")

    results = df[mask].copy()
    n_res   = len(results)

    # ── Result count bar ──────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:#fff;margin:0 2rem;padding:.8rem 1.25rem;"
        "border:1px solid #ddd;border-top:none;"
        "display:flex;align-items:center;gap:1.5rem'>"
        "<div style='display:flex;align-items:center;gap:.75rem'>"
        "<span style='font-size:1.5rem'>&#128663;</span>"
        "<div><div style='font-size:1.25rem;font-weight:700;color:#1a5276'>"
        "{n:,} vehicle{s}</div>"
        "<div style='font-size:.73rem;color:#888'>match your search</div></div>"
        "</div></div>".format(n=n_res, s="s" if n_res != 1 else ""),
        unsafe_allow_html=True,
    )

    if n_res == 0:
        st.markdown(
            "<div style='background:#fff;margin:0 2rem;padding:2rem;text-align:center;"
            "border:1px solid #ddd;border-top:none;color:#888'>"
            "No results found. Try adjusting your filters.</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Sort ──────────────────────────────────────────────────────────────────
    sc1, *_ = st.columns([1.2, 5])
    with sc1:
        sort_opts = {
            "CO\u2082 \u2191":       ("CO2 (g/km)", True),
            "CO\u2082 \u2193":       ("CO2 (g/km)", False),
            "Brand A\u2192Z":        ("Brand", True),
            "Consumption \u2191":    ("Mixed Conso (l/100km)", True),
            "Max Power \u2191":      ("Max Power (kW)", True),
        }
        sort_lbl = st.selectbox("Sort by", list(sort_opts.keys()), index=0)
    sort_col, sort_asc = sort_opts[sort_lbl]

    SHOW_COLS = [c for c in [
        "_row_id", "Body", "Brand", "Model", "Energy",
        "Mixed Conso (l/100km)",
        "CO2 (g/km)", "CO2 Class",
        "CO (g/km)", "HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)", "Particles (g/km)",
        "Euro Norm", "Gearbox Type", "Max Power (kW)", "Mass (kg)",
    ] if c in results.columns]

    display = (
        results[SHOW_COLS]
        .sort_values(sort_col, ascending=sort_asc, na_position="last")
        .copy()
    ) if sort_col in results.columns else results[SHOW_COLS].copy()

    # ── Pagination ────────────────────────────────────────────────────────────
    PAGE_SIZE   = 50
    total_pages = max(1, (n_res - 1) // PAGE_SIZE + 1)
    page        = max(0, min(st.session_state["tbl_page"], total_pages - 1))
    page_df     = display.iloc[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]

    pag1, pag2, pag3 = st.columns([1, 2, 1])
    with pag1:
        if st.button("\u2190 Previous", disabled=(page == 0)):
            st.session_state["tbl_page"] = page - 1
            st.rerun()
    with pag2:
        st.markdown(
            "<div style='text-align:center;font-size:.76rem;color:#888;padding:.5rem 0'>"
            "Page {p} of {t} \u00b7 {a}\u2013{b} of {n:,}</div>".format(
                p=page+1, t=total_pages,
                a=page*PAGE_SIZE+1, b=min((page+1)*PAGE_SIZE, n_res), n=n_res,
            ), unsafe_allow_html=True,
        )
    with pag3:
        if st.button("Next \u2192", disabled=(page >= total_pages - 1)):
            st.session_state["tbl_page"] = page + 1
            st.rerun()

    # ── HTML Table with clickable rows ────────────────────────────────────────
    def _badge(cls):
        if cls in ("?", ""): return cls
        bg = _cls_color(cls); fg = CLASS_FG.get(cls, "#fff")
        return ("<span style='display:inline-block;background:{bg};color:{fg};"
                "font-weight:700;font-size:.75rem;padding:.15rem .55rem;"
                "border-radius:2px;min-width:1.8rem;text-align:center'>{cls}</span>"
                ).format(bg=bg, fg=fg, cls=cls)

    TH  = ("background:#4a6741;color:#fff;font-size:.73rem;font-weight:600;"
           "padding:.45rem .5rem;text-align:center;white-space:nowrap;border:1px solid #5a7751")
    TH2 = ("background:#5a6e36;color:#fff;font-size:.73rem;font-weight:600;"
           "padding:.45rem .5rem;text-align:center;white-space:nowrap;border:1px solid #6a7e46")
    TD  = "padding:.35rem .5rem;font-size:.78rem;border:1px solid #e8e8e8;vertical-align:middle;text-align:center"
    TDL = "padding:.35rem .6rem;font-size:.78rem;border:1px solid #e8e8e8;vertical-align:middle;text-align:left"

    rows = [
        "<div style='margin:0 2rem;overflow-x:auto'>",
        "<table style='width:100%;border-collapse:collapse;background:#fff'>",
        "<thead>",
        "<tr>",
        "<th style='{th}'>Body</th>".format(th=TH),
        "<th style='{th}'>Brand / Model</th>".format(th=TH),
        "<th style='{th}'>Energy</th>".format(th=TH),
        "<th style='{th}' colspan='2'>Consumption<br><span style='font-weight:400;font-size:.65rem'>(l/100km)</span></th>".format(th=TH),
        "<th style='{th}' colspan='3'>CO\u2082 (g/km)</th>".format(th=TH),
        "<th style='{th2}'>CO</th><th style='{th2}'>HC</th><th style='{th2}'>NO\u2093</th>"
        "<th style='{th2}'>HC+NO\u2093</th><th style='{th2}'>Particles</th>".format(th2=TH2),
        "<th style='{th}'>Euro</th>".format(th=TH),
        "</tr>",
        "<tr style='background:#3a5731'>",
        "<th style='{th}'></th><th style='{th}'></th><th style='{th}'></th>".format(th=TH),
        "<th style='{th}'>Min.</th><th style='{th}'>Max.</th>".format(th=TH),
        "<th style='{th}'>Min.</th><th style='{th}'>Max.</th><th style='{th}'>Class</th>".format(th=TH),
        "<th style='{t}'></th><th style='{t}'></th><th style='{t}'></th>"
        "<th style='{t}'></th><th style='{t}'></th>".format(t=TH2),
        "<th style='{th}'></th>".format(th=TH),
        "</tr></thead><tbody>",
    ]

    for i, (_, r) in enumerate(page_df.iterrows()):
        row_id  = int(r["_row_id"])
        row_bg  = "#fff" if i % 2 == 0 else "#f7f7f5"
        co2v    = r.get("CO2 (g/km)")
        cls     = r.get("CO2 Class", "?")

        rows.append(
            "<tr style='background:{bg};cursor:pointer;transition:background .12s'"
            " onmouseover=\"this.style.background='#e8f0f8'\""
            " onmouseout=\"this.style.background='{bg}'\""
            " onclick=\"window.parent.postMessage({{type:'select_vehicle',row_id:{rid}}}, '*')\">".format(
                bg=row_bg, rid=row_id,
            )
        )
        rows += [
            "<td style='{td}'><span style='font-size:.72rem;color:#555'>{v}</span></td>".format(td=TD, v=r.get("Body","") or "\u2013"),
            "<td style='{td}'><span style='font-weight:700;font-size:.8rem;color:#1a5276'>{br}</span><br>"
            "<span style='font-size:.72rem;color:#555;font-style:italic'>{mo}</span></td>".format(
                td=TDL, br=r.get("Brand","") or "\u2013", mo=r.get("Model","") or "\u2013"),
            "<td style='{td}'><span style='font-size:.72rem'>{v}</span></td>".format(td=TD, v=r.get("Energy","") or "\u2013"),
            "<td style='{td}'>{v}</td>".format(td=TD, v=_fmt(r.get("Mixed Conso (l/100km)"))),
            "<td style='{td}'>{v}</td>".format(td=TD, v=_fmt(r.get("Mixed Conso (l/100km)"))),
            "<td style='{td}'>{v}</td>".format(td=TD, v=_fmt(co2v, 0)),
            "<td style='{td}'>{v}</td>".format(td=TD, v=_fmt(co2v, 0)),
            "<td style='{td}'>{v}</td>".format(td=TD, v=_badge(cls)),
        ]
        for col in ("CO (g/km)", "HC (g/km)", "NOx (g/km)", "HC+NOx (g/km)", "Particles (g/km)"):
            dec = 3 if col == "CO (g/km)" else 4
            rows.append("<td style='{td}'>{v}</td>".format(td=TD, v=_fmt(r.get(col), dec)))
        rows.append("<td style='{td}'><span style='font-size:.72rem'>{v}</span></td>".format(td=TD, v=str(r.get("Euro Norm","")) or "\u2013"))
        rows.append("</tr>")

    rows.append("</tbody></table></div>")

    # Inject JS message listener + table HTML
    table_html = "\n".join(rows)
    listener_js = """
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'select_vehicle') {
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: String(e.data.row_id)
            }, '*');
        }
    });
    </script>
    """ + table_html

    clicked = components.html(listener_js, height=max(400, len(page_df) * 42 + 120), scrolling=False)

    if clicked and clicked != "" and str(clicked).lstrip("-").isdigit():
        st.session_state["selected_row_id"] = int(clicked)
        st.rerun()

    # ── Compare expander ──────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    with st.expander("\u21c6  Compare vehicles (up to 3)"):
        id_labels = (
            display["Brand"].fillna("") + " " +
            display["Model"].fillna("") + " \u2014 " +
            display["Energy"].fillna("") +
            " [" + display["_row_id"].astype(str) + "]"
        ).tolist()
        cmp_sel = st.multiselect("Select vehicles", options=id_labels, default=[], max_selections=3,
                                  placeholder="Choose up to 3\u2026")
        if cmp_sel:
            sel_ids = [int(s.split("[")[-1].rstrip("]")) for s in cmp_sel]
            cmp_rows = df[df["_row_id"].isin(sel_ids)]
            cols_cmp = st.columns(len(cmp_rows))
            for col_ui, (_, r) in zip(cols_cmp, cmp_rows.iterrows()):
                co2v = r.get("CO2 (g/km)"); cls = r.get("CO2 Class","?")
                bg = _cls_color(cls); fg = CLASS_FG.get(cls,"#fff")
                col_ui.markdown(
                    "<div style='border:1px solid #ddd;border-radius:4px;overflow:hidden'>"
                    "<div style='background:{bg};padding:1.1rem;text-align:center'>"
                    "<div style='font-size:.6rem;font-weight:700;color:{fg};text-transform:uppercase;letter-spacing:.1em'>CO\u2082 Class</div>"
                    "<div style='font-size:3.5rem;font-weight:900;color:{fg};line-height:1'>{cls}</div>"
                    "<div style='font-size:1.2rem;font-weight:700;color:{fg}'>{co2} g/km</div></div>"
                    "<div style='background:#fff;padding:.85rem 1rem'>"
                    "<div style='font-size:.82rem;font-weight:700;color:#1a5276'>{brand}</div>"
                    "<div style='font-size:.75rem;color:#555;margin-bottom:.5rem'>{model}</div>"
                    "<table style='width:100%;font-size:.74rem;border-collapse:collapse'>"
                    "<tr><td style='color:#888'>Energy</td><td style='font-weight:600;text-align:right'>{energy}</td></tr>"
                    "<tr><td style='color:#888'>Consumption</td><td style='font-weight:600;text-align:right'>{conso}</td></tr>"
                    "<tr><td style='color:#888'>Power</td><td style='font-weight:600;text-align:right'>{pw}</td></tr>"
                    "</table></div></div>".format(
                        bg=bg, fg=fg, cls=cls, co2=_fmt(co2v,0),
                        brand=r.get("Brand","\u2013"), model=r.get("Model","\u2013"),
                        energy=r.get("Energy","\u2013"),
                        conso=_fmt(r.get("Mixed Conso (l/100km)"),1," l/100km"),
                        pw=_fmt(r.get("Max Power (kW)"),0," kW"),
                    ), unsafe_allow_html=True,
                )

    # Footer
    st.markdown(
        "<div style='background:#fff;border-top:1px solid #ddd;margin-top:2rem;"
        "padding:.6rem 2rem;font-size:.68rem;color:#888;display:flex;justify-content:space-between'>"
        "<span>ADEME Car Labelling Dataset 2013 \u00b7 {:,} vehicles</span>"
        "<span><a href='https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction' "
        "style='color:#1a5276;text-decoration:none'>GitHub \u2197</a></span>"
        "</div>".format(len(df)),
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state["selected_row_id"] is not None:
    render_detail(st.session_state["selected_row_id"])
else:
    render_search()
