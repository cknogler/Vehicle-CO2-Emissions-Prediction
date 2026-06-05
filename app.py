"""
app.py  –  Vehicle CO₂ Emissions Dashboard
Streamlit App für ADEME Car Labelling Dataset
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
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Vehicle CO₂ Emissions",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
sns.set_theme(style="whitegrid")

CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)

# Mögliche Spaltennamen für jede semantische Spalte
COL_ALIASES = {
    "Brand":        ["Brand", "Marque"],
    "Folder Model": ["Folder Model", "Modèle dossier", "Modele dossier"],
    "Body":         ["Body", "Carrosserie"],
    "Fuel":         ["Fuel", "Energie", "Énergie"],
    "Gearbox":      ["Gearbox", "Boîte de vitesse", "Boite de vitesse"],
    "Power":        ["Maximum Power (kW)", "Puissance maximale (kW)"],
    "Mass_min":     ["Empty Mass Euro Min (kg)", "masse vide euro min (kg)"],
    "Mass_max":     ["Empty Mass Euro Max (kg)", "masse vide euro max (kg)"],
    "Mass_avg":     ["Empty Mass Euro Avg (kg)"],
    "Consumption":  ["Combined Consumption (l/100km)", "Consommation mixte (l/100km)"],
    "CO2":          ["CO2 (g/km)"],
}

GEARBOX_CLEAN = {
    "M": "Manual", "A": "Automatic", "V": "CVT",
    "Manuel": "Manual", "Automatique": "Automatic",
    "manuelle": "Manual", "automatique": "Automatic",
}


def find_col(df, key):
    """Gibt den ersten gefundenen Spaltennamen für einen semantischen Key zurück."""
    for alias in COL_ALIASES.get(key, []):
        if alias in df.columns:
            return alias
    return None


@st.cache_data(show_spinner=False)
def load_raw(source) -> pd.DataFrame:
    """Lädt CSV aus URL (str) oder bytes."""
    if isinstance(source, str):
        with urllib.request.urlopen(source) as r:
            raw = r.read()
    else:
        raw = source
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [";", ","]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 3:
                    return df
            except Exception:
                continue
    raise ValueError("CSV konnte nicht gelesen werden.")


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Spaltennamen normalisieren
    rename = {}
    for std_name, aliases in COL_ALIASES.items():
        for alias in aliases[1:]:  # erstes ist bereits Standard
            if alias in df.columns and aliases[0] not in df.columns:
                rename[alias] = aliases[0]
    df = df.rename(columns=rename)

    # Gearbox bereinigen
    gb_col = find_col(df, "Gearbox")
    if gb_col:
        df[gb_col] = df[gb_col].map(
            lambda x: GEARBOX_CLEAN.get(str(x).strip(), str(x).strip()))

    # HC/NOX aus Summe
    if all(c in df.columns for c in ["HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)"]):
        m = df["HC (g/km)"].isna() & df["HC+NOX (g/km)"].notna()
        df.loc[m, "HC (g/km)"] = df.loc[m, "HC+NOX (g/km)"] / 2
        m2 = df["NOX (g/km)"].isna() & df["HC+NOX (g/km)"].notna()
        df.loc[m2, "NOX (g/km)"] = df.loc[m2, "HC+NOX (g/km)"] / 2

    # Elektro: Schadstoff-NaN → 0
    fuel_col = find_col(df, "Fuel")
    if fuel_col:
        el = df[fuel_col].astype(str).str.upper() == "EL"
        for c in [x for x in df.columns if any(p in x for p in ["HC", "NOX", "CO2", "Consumption"])]:
            df.loc[el, c] = df.loc[el, c].fillna(0)

    # Empty Mass Avg berechnen
    min_c = find_col(df, "Mass_min")
    max_c = find_col(df, "Mass_max")
    avg_c = find_col(df, "Mass_avg")
    if avg_c is None:
        if min_c and max_c:
            df["Empty Mass Euro Avg (kg)"] = (
                pd.to_numeric(df[min_c], errors="coerce") +
                pd.to_numeric(df[max_c], errors="coerce")
            ) / 2
            df.drop(columns=[min_c, max_c], inplace=True, errors="ignore")
        elif min_c:
            df["Empty Mass Euro Avg (kg)"] = pd.to_numeric(df[min_c], errors="coerce")

    # Numerische Spalten sicherstellen
    for key in ["CO2", "Consumption", "Power", "Mass_avg"]:
        c = find_col(df, key)
        if c:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    fuel_col = find_col(df, "Fuel")
    if fuel_col is None:
        return df

    # Filter ES + GO
    df_f = df[df[fuel_col].astype(str).str.upper().isin(["ES", "GO"])].copy()

    # Dedup-Keys: nur Spalten die existieren und nicht komplett leer sind
    candidate_keys = [
        find_col(df_f, k) for k in
        ["Brand", "Folder Model", "Fuel", "Body", "Gearbox",
         "Power", "Mass_avg", "CO2", "Consumption"]
    ]
    keys = [k for k in candidate_keys if k is not None and df_f[k].notna().any()]

    if len(keys) < 3:
        return df_f

    return df_f.drop_duplicates(subset=keys).copy()


@st.cache_data(show_spinner=False)
def run_clustering(_df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    power_c = find_col(_df, "Power")
    mass_c = find_col(_df, "Mass_avg")
    fuel_c = find_col(_df, "Fuel")
    body_c = find_col(_df, "Body")
    gear_c = find_col(_df, "Gearbox")

    num_f = [c for c in [power_c, mass_c] if c]
    cat_f = [c for c in [fuel_c, body_c, gear_c] if c]

    if not num_f:
        return _df

    df_c = _df.copy()
    for col in num_f:
        df_c[col] = df_c[col].fillna(df_c[col].median())

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df_c[num_f])
    parts = [X_num]
    for col in cat_f:
        le = LabelEncoder()
        enc = le.fit_transform(df_c[col].astype(str).fillna("Unknown"))
        parts.append(enc.reshape(-1, 1))

    X = np.hstack(parts)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_c["Cluster"] = km.fit_predict(X).astype(str)
    return df_c


@st.cache_resource(show_spinner=False)
def train_models(_df: pd.DataFrame):
    co2_c   = find_col(_df, "CO2")
    power_c = find_col(_df, "Power")
    mass_c  = find_col(_df, "Mass_avg")
    cons_c  = find_col(_df, "Consumption")
    fuel_c  = find_col(_df, "Fuel")
    body_c  = find_col(_df, "Body")
    gear_c  = find_col(_df, "Gearbox")

    if co2_c is None:
        raise ValueError("CO2-Spalte nicht gefunden.")

    feature_cols = [c for c in [power_c, mass_c, cons_c, fuel_c, body_c, gear_c] if c]
    df_m = _df[feature_cols + [co2_c]].copy()

    # Kategoriale encodieren
    encoders = {}
    cat_cols = [c for c in feature_cols if df_m[c].dtype == object]
    for col in cat_cols:
        le = LabelEncoder()
        df_m[col] = le.fit_transform(df_m[col].astype(str).fillna("Unknown"))
        encoders[col] = le

    # Alle numerisch machen
    for col in feature_cols:
        df_m[col] = pd.to_numeric(df_m[col], errors="coerce")

    # NaN entfernen
    df_m = df_m.dropna()

    if len(df_m) < 10:
        raise ValueError(f"Zu wenige Zeilen nach NaN-Bereinigung: {len(df_m)}")

    X = df_m[feature_cols].astype(float).values
    y = df_m[co2_c].astype(float).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)

    models = {
        "Ridge":            (Ridge(alpha=1.0), Xs_tr, Xs_te),
        "Random Forest":    (RandomForestRegressor(100, random_state=42, n_jobs=-1), X_tr, X_te),
        "Gradient Boosting":(GradientBoostingRegressor(100, learning_rate=0.05,
                                                        random_state=42), X_tr, X_te),
    }
    results, best_name, best_mae, best_model = {}, None, 1e9, None
    best_Xte, best_yte = None, None
    for name, (model, Xtr_, Xte_) in models.items():
        model.fit(Xtr_, y_tr)
        p = model.predict(Xte_)
        mae = mean_absolute_error(y_te, p)
        results[name] = {
            "MAE":  round(mae, 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p))), 2),
            "R2":   round(r2_score(y_te, p), 4),
        }
        if mae < best_mae:
            best_mae, best_name, best_model = mae, name, model
            best_Xte, best_yte = Xte_, y_te

    fi = {}
    if hasattr(best_model, "feature_importances_"):
        fi = dict(zip(feature_cols, best_model.feature_importances_))

    return best_model, best_name, results, fi, feature_cols, encoders, scaler, best_Xte, best_yte


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 CO₂ Dashboard")
    st.markdown("**ADEME Car Labelling Dataset**")
    st.markdown("---")
    st.caption("Datensatz wird automatisch aus dem Repo geladen.")
    uploaded = st.file_uploader("Eigene CSV hochladen (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("**Projekt:** [GitHub ↗](https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction)",
                unsafe_allow_html=True)
    st.caption("Preprocessing · EDA · Clustering · ML")

# ── Daten laden ──────────────────────────────────────────────────────────────
source = uploaded.read() if uploaded is not None else CSV_URL

with st.spinner("Daten werden geladen …"):
    try:
        df_raw   = load_raw(source)
        df_clean = preprocess(df_raw)
        df_unique = deduplicate(df_clean)
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        st.stop()

# Kurze Debug-Info in die Sidebar
with st.sidebar:
    st.markdown("---")
    st.caption(f"Rohdaten: {len(df_raw):,} Zeilen")
    st.caption(f"Unique (ES/GO): {len(df_unique):,} Zeilen")
    co2_col = find_col(df_clean, "CO2")
    st.caption(f"CO2-Spalte: `{co2_col}`")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📋 Überblick", "📊 EDA", "🔵 Clustering", "🤖 Prediction", "🎯 CO₂-Rechner"])

# ═══════════════════════ TAB 1 – ÜBERBLICK ═══════════════════════════════════
with tabs[0]:
    st.header("📋 Datensatz-Überblick")
    co2_col  = find_col(df_clean, "CO2")
    fuel_col = find_col(df_clean, "Fuel")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rohdaten (Zeilen)", f"{len(df_raw):,}")
    c2.metric("Spalten", len(df_clean.columns))
    es_go_n = len(df_clean[df_clean[fuel_col].astype(str).str.upper().isin(["ES","GO"])]) \
              if fuel_col else len(df_clean)
    c3.metric("ES+GO Fahrzeuge", f"{es_go_n:,}")
    c4.metric("Unique Konfigurationen", f"{len(df_unique):,}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Erste Zeilen")
        st.dataframe(df_clean.head(8), use_container_width=True)
    with col_r:
        st.subheader("Fehlende Werte")
        missing = df_clean.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(15)
        if len(missing) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            missing.plot(kind="barh", ax=ax, color=PALETTE[0])
            ax.set_xlabel("Anzahl")
            plt.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.success("Keine fehlenden Werte!")

    st.subheader("Statistik (numerisch)")
    st.dataframe(df_clean.select_dtypes(include=np.number).describe().T
                 .style.format("{:.2f}"), use_container_width=True)

    dedup_pct = (1 - len(df_unique)/es_go_n)*100 if es_go_n > 0 else 0
    st.info(f"**Deduplication:** {es_go_n:,} ES/GO → **{len(df_unique):,} unique** "
            f"({dedup_pct:.1f}% Duplikate entfernt)")

# ═══════════════════════ TAB 2 – EDA ═════════════════════════════════════════
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    co2_col  = find_col(df_unique, "CO2")
    fuel_col = find_col(df_unique, "Fuel")
    power_col = find_col(df_unique, "Power")
    mass_col  = find_col(df_unique, "Mass_avg")
    cons_col  = find_col(df_unique, "Consumption")

    if co2_col:
        co2 = df_unique[co2_col].dropna()
        st.subheader("CO₂-Verteilung")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(co2, bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
        axes[0].set_title("Histogramm"); axes[0].set_xlabel("CO₂ (g/km)")
        axes[1].boxplot(co2, patch_artist=True, boxprops=dict(facecolor=PALETTE[1], alpha=0.7))
        axes[1].set_title("Boxplot")
        stats.probplot(co2, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mittelwert", f"{co2.mean():.1f} g/km")
        c2.metric("Median",     f"{co2.median():.1f} g/km")
        c3.metric("Min",        f"{co2.min():.0f} g/km")
        c4.metric("Max",        f"{co2.max():.0f} g/km")

    st.markdown("---")

    # Korrelationsmatrix
    num_cols_corr = [c for c in [co2_col, power_col, mass_col, cons_col] if c]
    if len(num_cols_corr) >= 2:
        st.subheader("Korrelationsmatrix")
        corr = df_unique[num_cols_corr].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    ax=ax, square=True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Scatter CO2 vs andere
    scatter_vars = [c for c in [cons_col, mass_col, power_col] if c and co2_col]
    if scatter_vars:
        st.subheader("CO₂ vs. Schlüsselvariablen")
        fig, axes = plt.subplots(1, len(scatter_vars), figsize=(5*len(scatter_vars), 4))
        if len(scatter_vars) == 1: axes = [axes]
        for ax, var in zip(axes, scatter_vars):
            d = df_unique[[var, co2_col]].dropna()
            ax.scatter(d[var], d[co2_col], alpha=0.3, s=8, color=PALETTE[0])
            m, b = np.polyfit(d[var], d[co2_col], 1)
            xl = np.linspace(d[var].min(), d[var].max(), 100)
            ax.plot(xl, m*xl+b, color=PALETTE[3], lw=2)
            r, _ = stats.pearsonr(d[var], d[co2_col])
            ax.set_title(f"r = {r:.2f}"); ax.set_xlabel(var); ax.set_ylabel("CO₂ (g/km)")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Boxplot nach Kategorie
    st.markdown("---")
    cat_options = [c for c in [find_col(df_unique,"Fuel"),
                                find_col(df_unique,"Body"),
                                find_col(df_unique,"Gearbox")] if c]
    if cat_options and co2_col:
        st.subheader("CO₂ nach Kategorie")
        cat_choice = st.selectbox("Kategorie:", cat_options)
        top = df_unique[cat_choice].value_counts().head(8).index
        df_box = df_unique[df_unique[cat_choice].isin(top)]
        order = df_box.groupby(cat_choice)[co2_col].median().sort_values().index
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df_box, x=cat_choice, y=co2_col, order=order,
                    palette=PALETTE, ax=ax)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Hypothesis Test
    st.markdown("---")
    st.subheader("Mann-Whitney Test: Petrol vs. Diesel")
    if fuel_col and co2_col:
        es_s = df_unique[df_unique[fuel_col].astype(str).str.upper()=="ES"][co2_col].dropna()
        go_s = df_unique[df_unique[fuel_col].astype(str).str.upper()=="GO"][co2_col].dropna()
        if len(es_s)>0 and len(go_s)>0:
            stat, pval = stats.mannwhitneyu(es_s, go_s, alternative="two-sided")
            c1,c2,c3 = st.columns(3)
            c1.metric("Ø CO₂ Petrol (ES)", f"{es_s.mean():.1f} g/km")
            c2.metric("Ø CO₂ Diesel (GO)", f"{go_s.mean():.1f} g/km")
            c3.metric("p-Wert", f"{pval:.2e}")
            if pval < 0.05:
                st.success("✅ Signifikanter Unterschied (p < 0.05)")
            else:
                st.warning("Kein signifikanter Unterschied.")

# ═══════════════════════ TAB 3 – CLUSTERING ══════════════════════════════════
with tabs[2]:
    st.header("🔵 Clustering")
    k = st.slider("Anzahl Cluster (k)", 2, 8, 4)
    df_cl = run_clustering(df_unique, k=k)

    if "Cluster" in df_cl.columns:
        co2_col   = find_col(df_cl, "CO2")
        power_col = find_col(df_cl, "Power")
        mass_col  = find_col(df_cl, "Mass_avg")

        sizes = df_cl["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        sizes.plot(kind="bar", ax=ax, color=PALETTE[:k])
        ax.set_xlabel("Cluster"); ax.set_ylabel("Anzahl")
        ax.set_title("Fahrzeuge pro Cluster")
        plt.xticks(rotation=0); plt.tight_layout()
        st.pyplot(fig); plt.close()

        num_p = [c for c in [power_col, mass_col, co2_col] if c]
        if num_p:
            profile = df_cl.groupby("Cluster")[num_p].mean().round(1)
            st.subheader("Cluster-Profile")
            st.dataframe(profile.style.background_gradient(cmap="Blues"),
                         use_container_width=True)

        if power_col and mass_col:
            st.subheader("Power vs. Masse")
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, cl in enumerate(sorted(df_cl["Cluster"].unique())):
                m = df_cl["Cluster"]==cl
                ax.scatter(df_cl.loc[m, power_col], df_cl.loc[m, mass_col],
                           alpha=0.4, s=12, label=f"Cluster {cl}",
                           color=PALETTE[i%len(PALETTE)])
            ax.set_xlabel(power_col); ax.set_ylabel(mass_col)
            ax.legend(); plt.tight_layout()
            st.pyplot(fig); plt.close()

        if co2_col:
            st.subheader("CO₂ je Cluster")
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, cl in enumerate(sorted(df_cl["Cluster"].unique())):
                co2_cl = df_cl[df_cl["Cluster"]==cl][co2_col].dropna()
                ax.hist(co2_cl, bins=30, alpha=0.6, label=f"Cluster {cl}",
                        color=PALETTE[i%len(PALETTE)])
            ax.set_xlabel("CO₂ (g/km)"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════ TAB 4 – PREDICTION ══════════════════════════════════
with tabs[3]:
    st.header("🤖 Predictive Modeling")
    co2_col = find_col(df_unique, "CO2")
    if co2_col is None:
        st.error("CO2-Spalte nicht gefunden.")
    else:
        with st.spinner("Modelle werden trainiert …"):
            try:
                (best_model, best_name, results, fi,
                 features, encoders, scaler, X_te, y_te) = train_models(df_unique)

                st.subheader("Modellvergleich")
                res_df = pd.DataFrame(results).T
                st.dataframe(res_df.style
                    .highlight_min(subset=["MAE","RMSE"], color="#c8e6c9")
                    .highlight_max(subset=["R2"], color="#c8e6c9")
                    .format("{:.3f}"), use_container_width=True)
                st.success(f"✅ Bestes Modell: **{best_name}** "
                           f"(MAE = {results[best_name]['MAE']} g/km)")

                if fi:
                    st.subheader("Feature Importance")
                    fi_s = pd.Series(fi).sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    fi_s.plot(kind="barh", ax=ax, color=PALETTE[0])
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                st.subheader("Predicted vs. Actual")
                preds = best_model.predict(X_te)
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(y_te, preds, alpha=0.3, s=12, color=PALETTE[0])
                lims = [min(y_te.min(), preds.min())-10,
                        max(y_te.max(), preds.max())+10]
                ax.plot(lims, lims, "r--", lw=1.5)
                ax.set_xlabel("Tatsächlich (g/km)")
                ax.set_ylabel("Vorhergesagt (g/km)")
                ax.set_title(f"{best_name}")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            except Exception as e:
                st.error(f"Training fehlgeschlagen: {e}")

# ═══════════════════════ TAB 5 – CO₂-RECHNER ═════════════════════════════════
with tabs[4]:
    st.header("🎯 CO₂-Rechner")
    co2_col = find_col(df_unique, "CO2")
    if co2_col is None:
        st.warning("CO2-Spalte nicht gefunden.")
    else:
        try:
            (best_model, best_name, results, fi,
             features, encoders, scaler, X_te, y_te) = train_models(df_unique)

            power_col = find_col(df_unique, "Power")
            mass_col  = find_col(df_unique, "Mass_avg")
            cons_col  = find_col(df_unique, "Consumption")
            fuel_col  = find_col(df_unique, "Fuel")
            body_col  = find_col(df_unique, "Body")
            gear_col  = find_col(df_unique, "Gearbox")

            with st.form("co2_form"):
                col1, col2 = st.columns(2)
                inputs = {}
                with col1:
                    if power_col and power_col in features:
                        inputs[power_col] = st.number_input(
                            "Leistung (kW)", 30, 600, 110, 5)
                    if mass_col and mass_col in features:
                        inputs[mass_col] = st.number_input(
                            "Leergewicht (kg)", 700, 4000, 1400, 25)
                    if cons_col and cons_col in features:
                        inputs[cons_col] = st.number_input(
                            "Verbrauch komb. (l/100km)", 2.0, 30.0, 7.0, 0.1)
                with col2:
                    for col, key in [(fuel_col,"Fuel"),(body_col,"Body"),(gear_col,"Gearbox")]:
                        if col and col in features:
                            opts = sorted(df_unique[col].dropna().unique().tolist())
                            if opts:
                                inputs[col] = st.selectbox(col, opts)
                submitted = st.form_submit_button("CO₂ berechnen 🚀")

            if submitted:
                row = {f: inputs.get(f, 0) for f in features}
                df_input = pd.DataFrame([row])
                for col, le in encoders.items():
                    if col in df_input.columns:
                        val = str(df_input[col].iloc[0])
                        if val not in le.classes_:
                            val = le.classes_[0]
                        df_input[col] = le.transform([val])
                X_in = df_input[features].astype(float).values
                pred = best_model.predict(X_in)[0]

                euro = ("A (≤100)" if pred<=100 else "B (101–120)" if pred<=120
                        else "C (121–140)" if pred<=140 else "D (141–160)" if pred<=160
                        else "E (161–200)" if pred<=200 else "F/G (>200)")
                color = "green" if pred<=120 else "orange" if pred<=160 else "red"

                st.markdown(f"""
                <div style="background:{color}22;border-left:5px solid {color};
                            padding:20px;border-radius:8px;">
                  <h2 style="color:{color}">🚗 {pred:.1f} g/km CO₂</h2>
                  <p style="font-size:18px">EU-Klasse: <strong>{euro}</strong></p>
                  <p style="color:gray;font-size:13px">Modell: {best_name} · 
                  MAE ≈ {results[best_name]['MAE']} g/km</p>
                </div>""", unsafe_allow_html=True)

                pct = (df_unique[co2_col] <= pred).mean() * 100
                st.metric("Flottenvergleich",
                          f"Besser als {pct:.0f}% aller Fahrzeuge")

                fig, ax = plt.subplots(figsize=(9, 3))
                ax.hist(df_unique[co2_col].dropna(), bins=60,
                        color=PALETTE[0], alpha=0.6, label="Alle Fahrzeuge")
                ax.axvline(pred, color="red", lw=2.5, linestyle="--",
                           label=f"Deine Eingabe: {pred:.0f} g/km")
                ax.set_xlabel("CO₂ (g/km)"); ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()

        except Exception as e:
            st.error(f"Rechner nicht verfügbar: {e}")
