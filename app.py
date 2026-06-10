"""
app.py – Vehicle CO₂ · Executive Dashboard
Imports all data and model logic from pipeline.py.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pipeline import CSV_URL, load_data, train_models, predict_co2, segment_filter

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CO₂ Intelligence", page_icon="◈",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Design tokens ─────────────────────────────────────────────────────────────
MINT  = "#00C8A0"; AMBER = "#F5A623"; RED   = "#E84855"
BG    = "#0D0F18"; CARD  = "#13161F"; BORDER= "#1E2130"
TEXT  = "#EDF0F7"; MUTED = "#6B7280"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root{{--mint:{MINT};--amber:{AMBER};--red:{RED};--bg:{BG};--card:{CARD};
      --border:{BORDER};--text:{TEXT};--muted:{MUTED};
      --font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace}}
*{{box-sizing:border-box}}
html,body,.stApp{{background:var(--bg)!important;color:var(--text)!important;font-family:var(--font)!important}}
section[data-testid="stSidebar"]{{display:none}}
header[data-testid="stHeader"]{{background:transparent!important}}
.block-container{{padding:2rem 2.5rem 3rem!important;max-width:1400px}}
h1{{font-size:1.75rem!important;font-weight:800!important;letter-spacing:-.04em!important;
    color:var(--text)!important;margin:0!important}}
h2{{font-size:.75rem!important;font-weight:600!important;color:var(--muted)!important;
    text-transform:uppercase!important;letter-spacing:.12em!important;
    border:none!important;margin:0 0 1rem!important}}
h3{{font-size:1rem!important;font-weight:600!important;color:var(--text)!important}}
p,li{{color:var(--text)!important;line-height:1.6!important}}
[data-testid="stMetric"]{{background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:12px!important;padding:1.1rem 1.3rem!important;border-top:none!important}}
[data-testid="stMetricLabel"] p{{font-size:.65rem!important;font-weight:600!important;
  color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.1em!important;margin:0!important}}
[data-testid="stMetricValue"]{{font-size:1.55rem!important;font-weight:700!important;
  color:var(--text)!important;font-family:var(--mono)!important}}
[data-testid="stMetricDelta"] div{{font-size:.75rem!important}}
[data-testid="stSlider"] label{{font-size:.7rem!important;font-weight:600!important;
  color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.09em!important}}
[data-testid="stSelectbox"] label,[data-testid="stRadio"] label{{font-size:.7rem!important;
  font-weight:600!important;color:var(--muted)!important;text-transform:uppercase!important;letter-spacing:.09em!important}}
[data-testid="stSelectbox"]>div>div{{background:var(--card)!important;border-color:var(--border)!important;
  color:var(--text)!important;border-radius:8px!important}}
[data-testid="stRadio"] div[role="radiogroup"] label{{color:var(--text)!important;
  font-size:.88rem!important;text-transform:none!important;letter-spacing:normal!important}}
[data-testid="stDataFrame"]{{border:1px solid var(--border)!important;border-radius:10px!important;overflow:hidden!important}}
[data-testid="stExpander"]{{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:10px!important}}
[data-testid="stExpander"] summary{{font-weight:500!important;font-size:.85rem!important;color:var(--text)!important}}
hr{{border:none!important;border-top:1px solid var(--border)!important;margin:1.75rem 0!important}}
[data-testid="stCaptionContainer"] p{{color:var(--muted)!important;font-size:.73rem!important}}
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": CARD, "axes.facecolor": CARD,
    "axes.edgecolor": BORDER, "axes.labelcolor": MUTED,
    "axes.titlecolor": TEXT,  "axes.titlesize": 11,
    "axes.titleweight": "600","axes.labelsize": 9,
    "axes.grid": True,        "grid.color": BORDER, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": TEXT,       "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8,     "ytick.labelsize": 8,
    "legend.facecolor": CARD, "legend.edgecolor": BORDER, "legend.fontsize": 8,
    "savefig.facecolor": CARD,"savefig.edgecolor": CARD,
})

def co2_color(v):
    return MINT if v <= 120 else AMBER if v <= 160 else RED

def euro_class(v):
    if v <= 100:  return "A"
    if v <= 120:  return "B"
    if v <= 140:  return "C"
    if v <= 160:  return "D"
    if v <= 200:  return "E"
    return "F/G"

# ── Load & train (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load(source): return load_data(source)

@st.cache_resource(show_spinner=False)
def _train(_df): return train_models(_df)

with st.spinner("Loading data …"):
    df_raw, df_u = _load(CSV_URL)

with st.spinner("Training model …"):
    mb = _train(df_u)

best_model = "Random Forest"
rf_mae     = float(mb.results[mb.results["Model"] == best_model]["Test_MAE"].iloc[0])
rf_r2      = float(mb.results[mb.results["Model"] == best_model]["Test_R2"].iloc[0])

# ── Helper: build a feature row ───────────────────────────────────────────────
def make_row(mass, power, gears, fuel, body, gtype, feature_cols):
    row = {f: 0 for f in feature_cols}
    mapping = {
        "Empty Mass Euro Avg (kg)": float(mass),
        "Maximum Power (kW)":       float(power),
        "GearCount":                float(gears),
        "Fuel": fuel, "Body": body, "GearType": gtype,
    }
    for k, v in mapping.items():
        if k in row: row[k] = v
    return row

# ── Base values ───────────────────────────────────────────────────────────────
base = {
    "mass":  float(df_u["Empty Mass Euro Avg (kg)"].median()),
    "power": float(df_u["Maximum Power (kW)"].median()),
    "gears": float(df_u["GearCount"].median()) if "GearCount" in df_u.columns else 6.0,
    "fuel":  str(df_u["Fuel"].mode().iloc[0]),
    "body":  str(df_u["Body"].mode().iloc[0]),
    "gtype": str(df_u["GearType"].mode().iloc[0]) if "GearType" in df_u.columns else "Manual",
}
base_pred = predict_co2(mb, make_row(**base, feature_cols=mb.feature_cols))

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='display:flex;align-items:baseline;gap:1rem;margin-bottom:.25rem'>
  <span style='font-size:1.75rem;font-weight:800;letter-spacing:-.04em;color:#EDF0F7'>
    CO₂ Intelligence
  </span>
  <span style='font-size:.72rem;font-weight:600;color:#6B7280;
               text-transform:uppercase;letter-spacing:.12em'>
    ADEME · France · 2013
  </span>
</div>
<div style='font-size:.83rem;color:#6B7280;margin-bottom:2rem'>
  Predictive model for vehicle CO₂ emissions — Random Forest ·
  <span style='color:#EDF0F7'>R² {r2:.2f}</span> ·
  MAE <span style='color:#EDF0F7'>{mae:.1f} g/km</span> ·
  {n:,} unique configurations
</div>
""".format(r2=rf_r2, mae=rf_mae, n=len(df_u)), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
fleet_med = float(df_u["CO2 (g/km)"].median())
fleet_mean= float(df_u["CO2 (g/km)"].mean())
n_brands  = df_u["Brand"].nunique()
pct_sub130= (df_u["CO2 (g/km)"] <= 130).mean() * 100

k1.metric("Fleet Median",     f"{fleet_med:.0f} g/km")
k2.metric("Fleet Mean",       f"{fleet_mean:.0f} g/km")
k3.metric("Brands",           f"{n_brands}")
k4.metric("Configs ≤130 g/km",f"{pct_sub130:.0f}%")
k5.metric("Model MAE",        f"{rf_mae:.1f} g/km",  delta=f"R² {rf_r2:.3f}")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT: Simulator (left) | Result + Brand Comparison (right)
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.1, 1.9], gap="large")

with col_left:
    st.markdown("## Configure Vehicle")

    sim_mass  = st.slider("Kerb Weight (kg)",   800,  3200, int(base["mass"]),  50)
    sim_power = st.slider("Max Power (kW)",       40,   560, int(base["power"]),  5)

    if "GearCount" in mb.feature_cols:
        sim_gears = st.slider("Number of Gears",   4,     8, int(base["gears"]),  1)
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
    sim_body  = st.selectbox("Body Style", body_opts,
                              index=body_opts.index(base["body"]) if base["body"] in body_opts else 0)

    fuel_code  = "GO" if "GO" in sim_fuel else "ES"
    gtype_code = "Manual" if sim_fuel == "Manual" else sim_gtype

    sim_row  = make_row(sim_mass, sim_power, sim_gears, fuel_code, sim_body, gtype_code, mb.feature_cols)
    sim_pred = predict_co2(mb, sim_row)
    delta    = sim_pred - base_pred
    clr      = co2_color(sim_pred)
    ecls     = euro_class(sim_pred)
    dclr     = RED if delta > 0 else MINT if delta < 0 else MUTED

    # ── Prediction card ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background:{CARD};border:1px solid {BORDER};border-radius:14px;
                padding:1.5rem 1.75rem;margin-top:1.25rem;
                border-left:4px solid {clr}'>
      <div style='font-size:.65rem;font-weight:600;color:{MUTED};
                  text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem'>
        Predicted CO₂
      </div>
      <div style='display:flex;align-items:baseline;gap:.6rem'>
        <span style='font-size:3.2rem;font-weight:800;color:{clr};
                     font-family:var(--mono);letter-spacing:-.04em'>{sim_pred:.0f}</span>
        <span style='font-size:1rem;color:{MUTED}'>g/km</span>
        <span style='font-size:.75rem;font-weight:700;color:{clr};
                     background:{clr}20;border-radius:5px;padding:.2rem .55rem'>
          Class {ecls}
        </span>
      </div>
      <div style='font-size:.85rem;color:{dclr};font-weight:600;margin-top:.4rem'>
        {'▲' if delta > 0 else '▼' if delta < 0 else '—'} {delta:+.1f} g/km vs fleet base
      </div>
      <div style='font-size:.72rem;color:{MUTED};margin-top:.6rem'>
        Annual CO₂ ≈ {sim_pred * 15000 / 1000:.0f} kg @ 15,000 km ·
        Model MAE ± {rf_mae:.1f} g/km
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sensitivity sparklines ────────────────────────────────────────────────
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4))
    fig.patch.set_facecolor(CARD)

    for ax, key, rng, clr_line, label in [
        (axes[0], "Empty Mass Euro Avg (kg)", np.arange(800, 3300, 80), MINT, "Mass (kg)"),
        (axes[1], "Maximum Power (kW)",        np.arange(40,  570, 15), AMBER, "Power (kW)"),
    ]:
        preds = []
        for v in rng:
            r = sim_row.copy(); r[key] = float(v)
            preds.append(predict_co2(mb, r))
        ax.plot(rng, preds, color=clr_line, lw=1.8)
        ax.fill_between(rng, preds, alpha=0.07, color=clr_line)
        cur_val = sim_row[key]
        ax.axvline(cur_val, color=RED, lw=1.2, linestyle="--", alpha=0.8)
        ax.set_xlabel(label, fontsize=8); ax.set_ylabel("CO₂ g/km", fontsize=8)
        ax.set_title(f"CO₂ vs. {label.split(' ')[0]}", fontsize=9)

    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ── RIGHT COLUMN ──────────────────────────────────────────────────────────────
with col_right:
    st.markdown("## Brand Comparison")

    # Derive power bracket
    if   sim_power <= 55:  kw_lo, kw_hi = 0,   55
    elif sim_power <= 96:  kw_lo, kw_hi = 56,  96
    elif sim_power <= 147: kw_lo, kw_hi = 97,  147
    else:                  kw_lo, kw_hi = 148, 600

    df_seg = segment_filter(df_u, fuel_code, sim_body, kw_lo, kw_hi)
    if len(df_seg) < 5:                              # relax power if too sparse
        df_seg = df_u[df_u["Fuel"].eq(fuel_code) & df_u["Body"].eq(sim_body)].copy()

    avail_brands = (
        df_seg.groupby("Brand")["CO2 (g/km)"].count()
        .where(lambda x: x >= 2).dropna()
        .sort_values(ascending=False).index.tolist()
    )

    if not avail_brands:
        st.info("No brands with ≥2 models for this segment. Adjust Body Style or Fuel.")
    else:
        default_sel = avail_brands[:4]
        selected = st.multiselect(
            "Brands (max 5)",
            options=avail_brands, default=default_sel, max_selections=5,
        )

        if selected:
            PALETTE = [MINT, AMBER, RED, "#A78BFA", "#38BDF8"]
            bclr    = {b: PALETTE[i] for i, b in enumerate(selected)}

            # ── Compute per-brand stats ───────────────────────────────────────
            rows = []
            for brand in selected:
                df_b  = df_seg[df_seg["Brand"] == brand]
                co2_s = df_b["CO2 (g/km)"].dropna()
                if co2_s.empty: continue

                b_mass  = float(df_b["Empty Mass Euro Avg (kg)"].median())
                b_power = float(df_b["Maximum Power (kW)"].median())
                b_gears = float(df_b["GearCount"].median()) if "GearCount" in df_b.columns else sim_gears
                b_gtype = str(df_b["GearType"].mode().iloc[0]) if "GearType" in df_b.columns and len(df_b) > 0 else gtype_code

                b_row   = make_row(b_mass, b_power, b_gears, fuel_code, sim_body, b_gtype, mb.feature_cols)
                b_pred  = predict_co2(mb, b_row)

                rows.append({
                    "Brand":       brand,
                    "N":           len(co2_s),
                    "Median":      co2_s.median(),
                    "P25":         co2_s.quantile(0.25),
                    "P75":         co2_s.quantile(0.75),
                    "Min":         co2_s.min(),
                    "Max":         co2_s.max(),
                    "Pred":        b_pred,
                    "Typical_kW":  b_power,
                    "Typical_kg":  b_mass,
                })

            bdf = pd.DataFrame(rows).sort_values("Median").reset_index(drop=True)

            # ── Metric cards ──────────────────────────────────────────────────
            brand_cols = st.columns(len(bdf))
            for col_ui, (_, r) in zip(brand_cols, bdf.iterrows()):
                saving = (bdf["Median"].iloc[0] - r["Median"]) * 15000 / 1000
                col_ui.markdown(f"""
                <div style='background:{CARD};border:1px solid {BORDER};border-radius:12px;
                            border-top:3px solid {bclr[r["Brand"]]};padding:.9rem;text-align:center'>
                  <div style='font-size:.6rem;font-weight:600;color:{MUTED};
                              text-transform:uppercase;letter-spacing:.1em;margin-bottom:.2rem'>
                    {r["Brand"]}
                  </div>
                  <div style='font-size:1.6rem;font-weight:800;
                              color:{bclr[r["Brand"]]};font-family:var(--mono)'>
                    {r["Median"]:.0f}
                  </div>
                  <div style='font-size:.65rem;color:{MUTED}'>g/km · {int(r["N"])} models</div>
                  {'<div style="font-size:.62rem;color:#00C8A0;font-weight:600;margin-top:.3rem">★ Most efficient</div>' if r.name == 0 else f'<div style="font-size:.62rem;color:{MUTED};margin-top:.3rem">+{bdf["Median"].iloc[0] - r["Median"]:+.0f} g/km</div>' if r["Median"] != bdf["Median"].iloc[0] else ''}
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

            # ── Chart row ─────────────────────────────────────────────────────
            ch1, ch2 = st.columns(2)

            with ch1:
                fig, ax = plt.subplots(figsize=(6, 3.8))
                fig.patch.set_facecolor(CARD)
                x    = np.arange(len(bdf))
                clrs = [bclr[b] for b in bdf["Brand"]]

                ax.bar(x, bdf["Median"], color=clrs, alpha=0.75, edgecolor=CARD, lw=0, width=0.5)
                ax.errorbar(x, bdf["Median"],
                            yerr=[bdf["Median"]-bdf["P25"], bdf["P75"]-bdf["Median"]],
                            fmt="none", color=TEXT, capsize=4, lw=1.2, alpha=0.5)
                ax.scatter(x, bdf["Pred"], color=TEXT, s=40, zorder=5,
                           marker="D", label="Model prediction")
                ax.axhline(sim_pred, color=MUTED, lw=1, linestyle="--", alpha=0.7,
                           label=f"Simulator: {sim_pred:.0f}")
                ax.set_xticks(x); ax.set_xticklabels(bdf["Brand"], rotation=20, ha="right")
                ax.set_ylabel("CO₂ (g/km)"); ax.set_title("Median ± IQR  ◆ Prediction")
                ax.legend(fontsize=7)

                for xi, (_, r) in zip(x, bdf.iterrows()):
                    ax.text(xi, r["P75"] + 1.5, f'{r["Median"]:.0f}',
                            ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="600")
                plt.tight_layout(pad=0.6)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with ch2:
                fig, ax = plt.subplots(figsize=(6, 3.8))
                fig.patch.set_facecolor(CARD)
                pw_rng = np.arange(40, 570, 12)
                for _, r in bdf.iterrows():
                    b = r["Brand"]
                    base_b = make_row(r["Typical_kg"], sim_power, sim_gears,
                                       fuel_code, sim_body, gtype_code, mb.feature_cols)
                    preds = []
                    for p in pw_rng:
                        rb = base_b.copy(); rb["Maximum Power (kW)"] = float(p)
                        preds.append(predict_co2(mb, rb))
                    ax.plot(pw_rng, preds, color=bclr[b], lw=1.8, label=b)
                    ax.scatter([r["Typical_kW"]], [r["Pred"]],
                                color=bclr[b], s=40, zorder=5)
                ax.axvline(sim_power, color=MUTED, lw=1, linestyle="--", alpha=0.7,
                            label=f"Current: {sim_power} kW")
                ax.set_xlabel("Max Power (kW)"); ax.set_ylabel("Predicted CO₂ (g/km)")
                ax.set_title("CO₂ vs. Power by Brand")
                ax.legend(fontsize=7)
                plt.tight_layout(pad=0.6)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ── Detail table ──────────────────────────────────────────────────
            with st.expander("Full comparison table"):
                disp = bdf.rename(columns={
                    "Brand":"Brand","N":"Models","Median":"Median CO₂","P25":"P25",
                    "P75":"P75","Min":"Min","Max":"Max","Pred":"Model Pred.",
                    "Typical_kW":"Typical kW","Typical_kg":"Typical kg",
                })
                st.dataframe(
                    disp.style
                    .format({c: "{:.1f}" for c in ["Median CO₂","P25","P75","Min","Max","Model Pred.","Typical kW","Typical kg"]})
                    .highlight_min(subset=["Median CO₂"], color="#00C8A018")
                    .highlight_max(subset=["Median CO₂"], color="#E8485518"),
                    use_container_width=True, hide_index=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;
            font-size:.7rem;color:{MUTED}'>
  <span>ADEME Car Labelling Dataset 2013 · {len(df_u):,} unique ES/GO configurations</span>
  <span>Random Forest · {mb.best_fs} · R² {rf_r2:.3f} · MAE {rf_mae:.1f} g/km ·
    <a href='https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction'
       style='color:{MINT};text-decoration:none'>GitHub ↗</a>
  </span>
</div>
""", unsafe_allow_html=True)
