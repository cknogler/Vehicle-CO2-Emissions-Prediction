"""
search_app.py  –  CO₂ Vehicle Search  (ADEME Car Labelling replica)
Navigation via session_state only – no postMessage, no components.html.
Self-contained.
"""
from __future__ import annotations
import io, urllib.request, warnings
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)
COLUMN_MAPPING = {
    "Marque": "Brand", "Modèle dossier": "Model",
    "Désignation commerciale": "Commercial Name",
    "Carburant": "_fuel_code",
    "Puissance maximale (kW)": "Max Power (kW)",
    "Puissance administrative": "Fiscal Power (CV)",
    "Boîte de vitesse": "_gearbox_raw",
    "Consommation urbaine (l/100km)": "Urban Conso (l/100km)",
    "Consommation extra-urbaine (l/100km)": "Extra-Urban Conso (l/100km)",
    "Consommation mixte (l/100km)": "Mixed Conso (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)", "CO type I (g/km)": "CO (g/km)",
    "HC (g/km)": "HC (g/km)", "NOX (g/km)": "NOx (g/km)",
    "HC+NOX (g/km)": "HC+NOx (g/km)", "Particules (g/km)": "Particles (g/km)",
    "masse vide euro min (kg)": "_mass_min", "masse vide euro max (kg)": "_mass_max",
    "Carrosserie": "Body", "gamme": "Size", "Hybride": "Hybrid",
    "Champ V9": "Euro Norm", "Date de mise à jour": "Updated",
    "Type Variante Version (TVV)": "TVV", "CNIT": "CNIT",
}
FUEL_MAP = {
    "ES":"Petrol (ES)", "GO":"Diesel (GO)", "EL":"Electric (EL)",
    "GH":"Non-plug-in hybrid (GH)", "EH":"Non-plug-in hybrid (EH)",
    "EE":"Plug-in hybrid (EE)", "GL":"Plug-in hybrid (GL)",
    "GP":"LPG", "GN":"NGV", "FE":"Superethanol-E85 (FE)",
}
FUEL_GROUP = {
    "Electric (EL)":["EL"], "Non-plug-in hybrid":["EH","GH"],
    "Plug-in hybrid":["EE","GL"], "Petrol (ES)":["ES"], "Diesel (GO)":["GO"],
    "Superethanol-E85 (FE)":["FE"], "LPG":["GP"], "NGV":["GN"],
}
GEAR_MAP = {"M":"Manual","A":"Automatic","V":"CVT","D":"DCT","N":"Automatic","S":"Manual"}
CO2_CLASSES = [
    ("A",0,100,"#1a8c3c"),("B",101,120,"#4db84b"),("C",121,140,"#b2d145"),
    ("D",141,160,"#f9e000"),("E",161,200,"#e07b00"),("F",201,250,"#d03200"),("G",251,9999,"#a00000"),
]
CLASS_FG = {"A":"#fff","B":"#fff","C":"#111","D":"#111","E":"#fff","F":"#fff","G":"#fff"}
CLASS_LETTERS = [c for c,*_ in CO2_CLASSES]

def _co2_class(v):
    if pd.isna(v): return "?"
    for c,lo,hi,_ in CO2_CLASSES:
        if lo<=v<=hi: return c
    return "G"

def _cls_color(cls):
    for c,_,_,col in CO2_CLASSES:
        if c==cls: return col
    return "#888"

def _fmt(v, dec=1, unit=""):
    if v is None or (isinstance(v,float) and pd.isna(v)): return "\u2013"
    try: return "{:.{}f}{}".format(float(v), dec, unit)
    except: return str(v)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    with urllib.request.urlopen(CSV_URL) as r: raw = r.read()
    df = None
    for enc in ("latin1","utf-8","cp1252"):
        for sep in (";",","):
            try:
                t = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if t.shape[1]>5: df=t; break
            except: continue
        if df is not None: break
    df = df.rename(columns={k:v for k,v in COLUMN_MAPPING.items() if k in df.columns})
    if all(c in df.columns for c in ("HC (g/km)","NOx (g/km)","HC+NOx (g/km)")):
        hc  = (df["HC+NOx (g/km)"]-df["NOx (g/km)"]).fillna(df["HC (g/km)"])
        nox = (df["HC+NOx (g/km)"]-df["HC (g/km)"]).fillna(df["NOx (g/km)"])
        df["HC (g/km)"],df["NOx (g/km)"] = hc,nox
        df["HC+NOx (g/km)"] = hc+nox
    if "_gearbox_raw" in df.columns:
        df["_gearbox_raw"] = df["_gearbox_raw"].replace({"N 0":"A 0","N 1":"A 0","S 6":"D 6"})
        gs = df["_gearbox_raw"].astype(str).str.split(" ",expand=True)
        df["Gearbox Type"] = gs[0].map(GEAR_MAP).fillna(gs[0])
        df["Gear Count"]   = pd.to_numeric(gs[1] if 1 in gs.columns else pd.Series(dtype=float),errors="coerce").astype("Int64")
    if "_fuel_code" in df.columns:
        ev = df["_fuel_code"]=="EL"
        for c in ["CO2 (g/km)","Mixed Conso (l/100km)","HC (g/km)","NOx (g/km)","Particles (g/km)","CO (g/km)"]:
            if c in df.columns: df.loc[ev,c] = df.loc[ev,c].fillna(0)
    if "_mass_min" in df.columns and "_mass_max" in df.columns:
        df["Mass (kg)"] = (pd.to_numeric(df["_mass_min"],errors="coerce")+pd.to_numeric(df["_mass_max"],errors="coerce"))/2
    for c in ("CO2 (g/km)","Mixed Conso (l/100km)","Max Power (kW)","Mass (kg)",
              "Urban Conso (l/100km)","Extra-Urban Conso (l/100km)",
              "HC (g/km)","NOx (g/km)","Particles (g/km)","CO (g/km)","Fiscal Power (CV)"):
        if c in df.columns: df[c] = pd.to_numeric(df[c],errors="coerce")
    if "_fuel_code" in df.columns: df["Energy"] = df["_fuel_code"].map(FUEL_MAP).fillna(df["_fuel_code"])
    if "CO2 (g/km)" in df.columns: df["CO2 Class"] = df["CO2 (g/km)"].apply(_co2_class)
    if "Max Power (kW)" in df.columns: df["Max Power (HP)"] = (df["Max Power (kW)"]*1.36).round(0).astype("Int64")
    df = df.reset_index(drop=True)
    df["_row_id"] = df.index
    return df

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CO\u2082 Vehicle Search \u00b7 ADEME",
                   page_icon="\U0001f697", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&display=swap');
html,body,.stApp{background:#f0f0f0!important;font-family:'Open Sans',sans-serif!important;color:#333!important}
.block-container{padding:0!important;max-width:100%!important}
header[data-testid="stHeader"]{display:none!important}
section[data-testid="stSidebar"]{display:none!important}
p,li{color:#333!important}
/* table row button: invisible wrapper */
div[data-testid="stButton"] button.row-btn{
  background:transparent!important;border:none!important;padding:0!important;
  width:100%!important;text-align:left!important;cursor:pointer!important;
  color:#1a5276!important;font-weight:700!important;font-size:.8rem!important;
  line-height:1.3!important;
}
div[data-testid="stButton"] button.row-btn:hover{text-decoration:underline!important;background:transparent!important}
/* regular buttons */
.stButton>button{background:#1a5276!important;color:#fff!important;
  border:none!important;border-radius:3px!important;font-weight:700!important;
  font-size:.82rem!important;padding:.45rem 1.1rem!important}
.stButton>button:hover{background:#154360!important}
[data-testid="stSelectbox"]>div>div{background:#fff!important;border:1px solid #ccc!important;border-radius:3px!important}
[data-testid="stSelectbox"] label,[data-testid="stCheckbox"] label,
[data-testid="stSlider"] label{color:#1a5276!important;font-weight:700!important;font-size:.82rem!important}
hr{border:none!important;border-top:1px solid #ddd!important;margin:.75rem 0!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:#ccc;border-radius:3px}
/* thin dividers between table rows */
.tbl-row{border-bottom:1px solid #e8e8e8;padding:.3rem 0}
</style>
""", unsafe_allow_html=True)

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading ADEME dataset\u2026"):
    df = load_data()

# ── Session state ──────────────────────────────────────────────────────────────
for k,v in [("selected_row_id",None),("tbl_page",0)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def topbar(back=False):
    st.markdown(
        "<div style='background:#fff;border-bottom:3px solid #c0392b;padding:.6rem 2rem;"
        "display:flex;align-items:center;gap:1.5rem'>"
        "<div><span style='font-size:1.2rem;font-weight:800;color:#c0392b'>Car</span>"
        "<span style='font-size:1.2rem;font-weight:800;color:#1a5276'> Labelling</span>"
        "<div style='font-size:.58rem;color:#999;text-transform:uppercase;letter-spacing:.1em'>"
        "ADEME Dataset 2013</div></div></div>",
        unsafe_allow_html=True,
    )

def badge(cls):
    if not cls or cls=="?": return cls
    bg=_cls_color(cls); fg=CLASS_FG.get(cls,"#fff")
    return ("<span style='background:{};color:{};font-weight:700;font-size:.72rem;"
            "padding:.12rem .5rem;border-radius:2px;display:inline-block;"
            "min-width:1.6rem;text-align:center'>{}</span>").format(bg,fg,cls)

# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_detail(row_id):
    topbar()

    if st.button("\u2190  Back to results", key="back_top"):
        st.session_state["selected_row_id"] = None
        st.rerun()

    r   = df[df["_row_id"]==row_id].iloc[0]
    cls = r.get("CO2 Class","?")
    bg  = _cls_color(cls)
    fg  = CLASS_FG.get(cls,"#fff")
    co2 = r.get("CO2 (g/km)")

    # breadcrumb
    st.markdown(
        "<div style='font-size:.73rem;color:#888;padding:.25rem 0 .75rem'>"
        "Search &rsaquo; <b>{} {}</b></div>".format(r.get("Brand",""),r.get("Model","")),
        unsafe_allow_html=True,
    )

    col_label, col_specs = st.columns([1,2.2], gap="large")

    # ── LEFT: EU Label ────────────────────────────────────────────────────────
    with col_label:
        widths = {"A":52,"B":60,"C":68,"D":76,"E":84,"F":92,"G":100}
        bars = ""
        for c,lo,hi,col in CO2_CLASSES:
            w   = widths.get(c,70)
            brd = "outline:2px solid #111;outline-offset:-2px;" if c==cls else ""
            fgc = CLASS_FG.get(c,"#fff")
            lbl = "{}&ndash;{}".format(lo, hi if hi<9999 else "+")
            bars += (
                "<div style='background:{col};width:{w}%;height:1.85rem;margin-bottom:2px;"
                "display:flex;align-items:center;justify-content:space-between;"
                "padding:0 .55rem;border-radius:2px;{brd}'>"
                "<b style='font-size:.82rem;color:{fg}'>{c}</b>"
                "<span style='font-size:.65rem;color:{fg};opacity:.85'>{lbl}</span>"
                "</div>"
            ).format(col=col,w=w,brd=brd,fg=fgc,c=c,lbl=lbl)

        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;padding:1.25rem 1.4rem'>"
            "<div style='font-size:.62rem;font-weight:700;color:#666;text-transform:uppercase;"
            "letter-spacing:.1em;text-align:center;margin-bottom:.75rem'>CO\u2082 Emission Class</div>"
            "{bars}"
            "<div style='text-align:center;margin-top:.9rem'>"
            "<span style='font-size:2.4rem;font-weight:900;color:{bg}'>{cls}</span>"
            "<span style='font-size:.9rem;color:#666;margin-left:.4rem'>"
            "{co2} g/km</span></div></div>".format(bars=bars,bg=bg,cls=cls,co2=_fmt(co2,0)),
            unsafe_allow_html=True,
        )

        # Energy cost estimate
        conso = r.get("Mixed Conso (l/100km)")
        if pd.notna(conso) and conso>0:
            cost = conso/100*15000*1.85
            st.markdown(
                "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;"
                "padding:1rem 1.2rem;margin-top:.75rem'>"
                "<div style='font-size:.62rem;font-weight:700;color:#666;text-transform:uppercase;"
                "letter-spacing:.1em;margin-bottom:.4rem'>Energy cost estimate</div>"
                "<div style='font-size:1.5rem;font-weight:800;color:#1a5276'>"
                "&euro;{cost:.0f}/year</div>"
                "<div style='font-size:.7rem;color:#999;margin-top:.2rem'>"
                "{c:.1f} l/100km \u00b7 15,000 km \u00b7 &euro;1.85/l</div>"
                "</div>".format(cost=cost,c=conso),
                unsafe_allow_html=True,
            )

    # ── RIGHT: Specs ──────────────────────────────────────────────────────────
    with col_specs:
        brand = str(r.get("Brand","")).upper()
        model = str(r.get("Model",""))
        cname = str(r.get("Commercial Name",""))

        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;padding:1.4rem 1.75rem'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:1.55rem;font-weight:800;color:#1a5276'>{}</div>"
            "<div style='font-size:1rem;font-weight:600;color:#333;margin-bottom:.15rem'>{}</div>"
            "<div style='font-size:.78rem;color:#999;margin-bottom:1.4rem'>{}</div>".format(
                brand, model, cname if cname not in ("","nan") else ""),
            unsafe_allow_html=True,
        )

        def spec_section(title, items):
            html = ("<div style='margin-bottom:1.1rem'>"
                    "<div style='font-size:.65rem;font-weight:700;color:#1a5276;"
                    "text-transform:uppercase;letter-spacing:.1em;"
                    "border-bottom:2px solid #1a5276;padding-bottom:.2rem;"
                    "margin-bottom:.55rem'>{}</div>"
                    "<table style='width:100%;border-collapse:collapse'>").format(title)
            for label, val, unit in items:
                if val in (None,"","nan","\u2013") or (isinstance(val,float) and pd.isna(val)):
                    continue
                html += ("<tr><td style='font-size:.8rem;color:#666;padding:.28rem 0;width:52%'>{}</td>"
                         "<td style='font-size:.8rem;font-weight:600;color:#222;"
                         "text-align:right'>{} <span style='font-weight:400;color:#999;"
                         "font-size:.72rem'>{}</span></td></tr>").format(label,val,unit)
            html += "</table></div>"
            return html

        pw_kw = r.get("Max Power (kW)"); pw_hp = r.get("Max Power (HP)")
        pw_str = ("{} kW ({} HP)".format(_fmt(pw_kw,0),int(pw_hp))
                  if pd.notna(pw_kw) and pd.notna(pw_hp) else _fmt(pw_kw,0))
        gear_str = "{} ({} gears)".format(
            r.get("Gearbox Type",""),
            str(int(r["Gear Count"])) if "Gear Count" in r.index and pd.notna(r.get("Gear Count")) else "?"
        ) if "Gearbox Type" in r.index else "\u2013"

        html_out = ""
        html_out += spec_section("Technical Specifications", [
            ("Body type",       str(r.get("Body","")),          ""),
            ("Segment",         str(r.get("Size","")),           ""),
            ("Energy / Fuel",   str(r.get("Energy","")),         ""),
            ("Max Power",       pw_str,                          ""),
            ("Fiscal Power",    _fmt(r.get("Fiscal Power (CV)"),0), "CV"),
            ("Gearbox",         gear_str,                        ""),
            ("Kerb Mass",       _fmt(r.get("Mass (kg)"),0),      "kg"),
            ("Euro Norm",       str(r.get("Euro Norm","")),      ""),
        ])
        html_out += spec_section("Consumption (NEDC)", [
            ("Urban",           _fmt(r.get("Urban Conso (l/100km)")),       "l/100km"),
            ("Extra-Urban",     _fmt(r.get("Extra-Urban Conso (l/100km)")), "l/100km"),
            ("Combined",        _fmt(r.get("Mixed Conso (l/100km)")),       "l/100km"),
        ])
        html_out += spec_section("CO\u2082 Emissions", [
            ("CO\u2082 emissions",  _fmt(co2,0),  "g/km"),
            ("Efficiency class",    cls,           ""),
        ])
        html_out += spec_section("Pollutants (g/km)", [
            ("CO",          _fmt(r.get("CO (g/km)"),3),      "g/km"),
            ("HC",          _fmt(r.get("HC (g/km)"),4),      "g/km"),
            ("NO\u2093",    _fmt(r.get("NOx (g/km)"),4),     "g/km"),
            ("HC+NO\u2093", _fmt(r.get("HC+NOx (g/km)"),4),  "g/km"),
            ("Particles",   _fmt(r.get("Particles (g/km)"),4),"g/km"),
        ])
        if "TVV" in r.index or "CNIT" in r.index:
            html_out += spec_section("Homologation", [
                ("TVV",  str(r.get("TVV","")),  ""),
                ("CNIT", str(r.get("CNIT","")), ""),
            ])

        st.markdown(html_out + "</div>", unsafe_allow_html=True)

    # ── CO₂ in context ────────────────────────────────────────────────────────
    if pd.notna(co2):
        fleet_med = float(df["CO2 (g/km)"].median())
        same_body = df[df["Body"]==r.get("Body","")]["CO2 (g/km)"].dropna()
        seg_med   = float(same_body.median()) if len(same_body)>0 else None

        comp = [("This vehicle", co2, _cls_color(cls)),
                ("Fleet median", fleet_med, "#888")]
        if seg_med is not None:
            comp.insert(1,("{} median".format(r.get("Body","")), seg_med, "#1a5276"))

        max_v = max(v for _,v,_ in comp)*1.12
        bars2 = ""
        for lbl,val,col in comp:
            pct = val/max_v*100
            bars2 += (
                "<div style='margin-bottom:.55rem'>"
                "<div style='display:flex;justify-content:space-between;"
                "font-size:.78rem;margin-bottom:.18rem'>"
                "<span style='color:#555'>{}</span>"
                "<span style='font-weight:700;color:#222'>{:.0f} g/km</span></div>"
                "<div style='background:#e8e8e8;border-radius:3px;height:.6rem'>"
                "<div style='background:{};width:{:.1f}%;height:100%;border-radius:3px'>"
                "</div></div></div>"
            ).format(lbl,val,col,pct)

        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:6px;"
            "padding:1.1rem 1.4rem;margin-top:.5rem'>"
            "<div style='font-size:.65rem;font-weight:700;color:#1a5276;text-transform:uppercase;"
            "letter-spacing:.1em;margin-bottom:.8rem'>CO\u2082 in Context</div>"
            "{}</div>".format(bars2),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("\u2190  Back to results", key="back_bottom"):
        st.session_state["selected_row_id"] = None
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_search():
    topbar()

    # ── Search panel ──────────────────────────────────────────────────────────
    with st.container():
        st.markdown(
            "<div style='background:#fff;border:1px solid #ddd;border-radius:4px;"
            "padding:1.25rem 1.5rem;margin-bottom:0'>"
            "<div style='display:flex;align-items:center;gap:.75rem;margin-bottom:1.1rem'>"
            "<div style='background:#1a5276;border-radius:3px;width:2rem;height:2rem;"
            "display:flex;align-items:center;justify-content:center;font-size:1rem'>&#128269;</div>"
            "<span style='font-size:.9rem;font-weight:700;color:#1a5276;"
            "text-transform:uppercase;letter-spacing:.06em'>Multi-Criteria Search</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        cl, cr = st.columns([1, 1.4], gap="large")

        with cl:
            brands_all = sorted(df["Brand"].dropna().unique())
            sel_brand  = st.selectbox("Brand", ["Choose\u2026"]+brands_all)
            pool = sorted(df[df["Brand"]==sel_brand]["Model"].dropna().unique()) \
                   if sel_brand!="Choose\u2026" else sorted(df["Model"].dropna().unique())
            sel_model  = st.selectbox("Model", ["Choose\u2026"]+pool)
            sel_body   = st.selectbox("Body",  ["Choose\u2026"]+sorted(df["Body"].dropna().unique()))
            sel_size   = st.selectbox("Size",  ["Choose\u2026"]+sorted(df["Size"].dropna().unique())) \
                         if "Size" in df.columns else "Choose\u2026"

        with cr:
            st.markdown(
                "<div style='font-weight:700;color:#1a5276;font-size:.82rem;"
                "margin-bottom:.35rem'>Energy</div>",
                unsafe_allow_html=True,
            )
            ec1,ec2,ec3 = st.columns(3)
            fuel_sel = {}
            for i,grp in enumerate(FUEL_GROUP):
                fuel_sel[grp] = [ec1,ec2,ec3][i%3].checkbox(grp,key="f_"+grp)

            g1,g2 = st.columns(2)
            with g1:
                gear_opts = sorted(df["Gearbox Type"].dropna().unique()) \
                            if "Gearbox Type" in df.columns else []
                sel_gear  = st.selectbox("Gearbox",["Choose\u2026"]+gear_opts)
            with g2:
                cmax = float(df["Mixed Conso (l/100km)"].dropna().max()) \
                       if "Mixed Conso (l/100km)" in df.columns else 30.0
                conso_until = st.number_input("Max consumption (l/100km)",
                                              min_value=0.0,max_value=cmax,value=cmax,step=0.5,format="%.1f")
            co2_rng = st.select_slider("CO\u2082 Class",options=CLASS_LETTERS,value=("A","G"))

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    mask = pd.Series(True, index=df.index)
    if sel_brand!="Choose\u2026": mask &= df["Brand"]==sel_brand
    if sel_model!="Choose\u2026": mask &= df["Model"]==sel_model
    if sel_body !="Choose\u2026": mask &= df["Body"] ==sel_body
    if sel_size !="Choose\u2026" and "Size" in df.columns: mask &= df["Size"]==sel_size
    if sel_gear !="Choose\u2026" and "Gearbox Type" in df.columns: mask &= df["Gearbox Type"]==sel_gear
    active_fuels=[code for g,t in fuel_sel.items() if t for code in FUEL_GROUP[g]]
    if active_fuels and "_fuel_code" in df.columns: mask &= df["_fuel_code"].isin(active_fuels)
    if "Mixed Conso (l/100km)" in df.columns:
        mask &= (df["Mixed Conso (l/100km)"]<=conso_until)|df["Mixed Conso (l/100km)"].isna()
    lo=CLASS_LETTERS.index(co2_rng[0]); hi=CLASS_LETTERS.index(co2_rng[1])
    if "CO2 Class" in df.columns:
        mask &= df["CO2 Class"].isin(CLASS_LETTERS[lo:hi+1])|(df["CO2 Class"]=="?")

    results = df[mask].copy()
    n_res   = len(results)

    st.markdown(
        "<div style='background:#fff;padding:.75rem 1.25rem;border:1px solid #ddd;"
        "border-top:none;display:flex;align-items:center;gap:1rem;margin-bottom:.5rem'>"
        "<span style='font-size:1.3rem'>&#128663;</span>"
        "<span style='font-size:1.2rem;font-weight:700;color:#1a5276'>{n:,} vehicle{s}</span>"
        "<span style='font-size:.72rem;color:#999'>match your search</span>"
        "</div>".format(n=n_res, s="s" if n_res!=1 else ""),
        unsafe_allow_html=True,
    )

    if n_res==0:
        st.info("No results. Try adjusting your filters.")
        return

    # ── Sort + page ───────────────────────────────────────────────────────────
    sc,_ = st.columns([1.2,5])
    with sc:
        sort_opts={"CO\u2082 \u2191":("CO2 (g/km)",True),"CO\u2082 \u2193":("CO2 (g/km)",False),
                   "Brand A\u2192Z":("Brand",True),"Consumption \u2191":("Mixed Conso (l/100km)",True),
                   "Power \u2191":("Max Power (kW)",True)}
        sl=st.selectbox("Sort by",list(sort_opts.keys()))
    scol,sasc=sort_opts[sl]

    COLS=[c for c in ["_row_id","Body","Brand","Model","Energy","Mixed Conso (l/100km)",
                      "CO2 (g/km)","CO2 Class","CO (g/km)","HC (g/km)","NOx (g/km)",
                      "HC+NOx (g/km)","Particles (g/km)","Euro Norm","Gearbox Type",
                      "Max Power (kW)","Mass (kg)"] if c in results.columns]
    display = results[COLS].sort_values(scol,ascending=sasc,na_position="last").copy() \
              if scol in results.columns else results[COLS].copy()

    PAGE_SIZE=25
    total_pages=max(1,(n_res-1)//PAGE_SIZE+1)
    page=max(0,min(st.session_state["tbl_page"],total_pages-1))
    page_df=display.iloc[page*PAGE_SIZE:(page+1)*PAGE_SIZE]

    p1,p2,p3=st.columns([1,2,1])
    with p1:
        if st.button("\u2190 Prev",disabled=(page==0),key="prev"):
            st.session_state["tbl_page"]=page-1; st.rerun()
    with p2:
        st.markdown(
            "<div style='text-align:center;font-size:.74rem;color:#999;padding:.4rem 0'>"
            "Page {p}/{t} \u00b7 {a}\u2013{b} of {n:,}</div>".format(
                p=page+1,t=total_pages,
                a=page*PAGE_SIZE+1,b=min((page+1)*PAGE_SIZE,n_res),n=n_res),
            unsafe_allow_html=True,
        )
    with p3:
        if st.button("Next \u2192",disabled=(page>=total_pages-1),key="next"):
            st.session_state["tbl_page"]=page+1; st.rerun()

    # ── TABLE: header ─────────────────────────────────────────────────────────
    TH  = "background:#4a6741;color:#fff;font-size:.68rem;font-weight:700;padding:.4rem .3rem;text-align:center;border:1px solid #5a7751"
    TH2 = "background:#5a6e36;color:#fff;font-size:.68rem;font-weight:700;padding:.4rem .3rem;text-align:center;border:1px solid #6a7e46"

    st.markdown(
        "<div style='overflow-x:auto'>"
        "<table style='width:100%;border-collapse:collapse;background:#fff'>"
        "<thead><tr>"
        "<th style='{th}'>Body</th>"
        "<th style='{th}'>Brand / Model</th>"
        "<th style='{th}'>Energy</th>"
        "<th style='{th}' colspan='2'>Consumption<br><small style='font-weight:400'>(l/100km)</small></th>"
        "<th style='{th}' colspan='3'>CO\u2082 (g/km)</th>"
        "<th style='{th2}'>CO</th><th style='{th2}'>HC</th>"
        "<th style='{th2}'>NO\u2093</th><th style='{th2}'>HC+NO\u2093</th>"
        "<th style='{th2}'>Particles</th>"
        "<th style='{th}'>Euro</th>"
        "</tr><tr style='background:#3a5731'>"
        "<th style='{th}'></th><th style='{th}'></th><th style='{th}'></th>"
        "<th style='{th}'>Min</th><th style='{th}'>Max</th>"
        "<th style='{th}'>Min</th><th style='{th}'>Max</th><th style='{th}'>Class</th>"
        "<th style='{th2}'></th><th style='{th2}'></th><th style='{th2}'></th>"
        "<th style='{th2}'></th><th style='{th2}'></th>"
        "<th style='{th}'></th>"
        "</tr></thead></table></div>".format(th=TH,th2=TH2),
        unsafe_allow_html=True,
    )

    # ── TABLE: rows via st.columns + buttons ───────────────────────────────────
    TD  = "font-size:.76rem;padding:.32rem .3rem;border-bottom:1px solid #eee;vertical-align:middle;text-align:center"
    TDL = "font-size:.76rem;padding:.32rem .4rem;border-bottom:1px solid #eee;vertical-align:middle;text-align:left"

    for i,(idx,r) in enumerate(page_df.iterrows()):
        row_id = int(r["_row_id"])
        co2v   = r.get("CO2 (g/km)")
        cls    = r.get("CO2 Class","?")
        bg_row = "#fff" if i%2==0 else "#f7f7f5"

        # 14 visual columns + 1 hidden button column
        c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13 = st.columns(
            [0.7,2.0,1.1,0.7,0.7,0.7,0.7,0.7,0.55,0.55,0.55,0.65,0.65,0.6])

        style = "background:{};".format(bg_row)

        c0.markdown("<div style='{s}{td}'><span style='font-size:.7rem;color:#555'>{v}</span></div>".format(
            s=style,td=TD,v=r.get("Body","") or "\u2013"), unsafe_allow_html=True)

        # Brand/Model as clickable button
        btn_label = "**{}**\n{}".format(r.get("Brand",""),r.get("Model",""))
        if c1.button(
            "{}\n{}".format(r.get("Brand","") or "\u2013", r.get("Model","") or "\u2013"),
            key="row_{}".format(row_id),
            help="View details",
        ):
            st.session_state["selected_row_id"] = row_id
            st.rerun()

        c2.markdown("<div style='{s}{td}'><span style='font-size:.7rem'>{v}</span></div>".format(
            s=style,td=TD,v=r.get("Energy","") or "\u2013"), unsafe_allow_html=True)
        c3.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("Mixed Conso (l/100km)"))), unsafe_allow_html=True)
        c4.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("Mixed Conso (l/100km)"))), unsafe_allow_html=True)
        c5.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(co2v,0)), unsafe_allow_html=True)
        c6.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(co2v,0)), unsafe_allow_html=True)
        c7.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=badge(cls)), unsafe_allow_html=True)
        c8.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("CO (g/km)"),3)), unsafe_allow_html=True)
        c9.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("HC (g/km)"),4)), unsafe_allow_html=True)
        c10.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("NOx (g/km)"),4)), unsafe_allow_html=True)
        c11.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("HC+NOx (g/km)"),4)), unsafe_allow_html=True)
        c12.markdown("<div style='{s}{td}'>{v}</div>".format(s=style,td=TD,v=_fmt(r.get("Particles (g/km)"),4)), unsafe_allow_html=True)
        c13.markdown("<div style='{s}{td}'><span style='font-size:.7rem'>{v}</span></div>".format(
            s=style,td=TD,v=str(r.get("Euro Norm","")) or "\u2013"), unsafe_allow_html=True)

    # Footer
    st.markdown(
        "<div style='background:#fff;border-top:1px solid #ddd;margin-top:1.5rem;"
        "padding:.55rem 2rem;font-size:.67rem;color:#999;display:flex;"
        "justify-content:space-between'>"
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
    page_detail(st.session_state["selected_row_id"])
else:
    page_search()
