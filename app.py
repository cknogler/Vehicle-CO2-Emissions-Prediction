"""
app.py  –  Vehicle CO₂ Emissions Dashboard
==========================================
Streamlit-App für das ADEME Car Labelling Projekt.

Starten:
    streamlit run app.py

Benötigte Pakete:
    pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy
"""

import io
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

# ── Styling ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle CO₂ Emissions",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
sns.set_theme(style="whitegrid", palette=PALETTE)

# ── Hilfsfunktionen ────────────────────────────────────────────────────────

RENAME_MAP = {
    "Marque": "Brand",
    "Modèle dossier": "Folder Model",
    "Désignation commerciale": "Commercial Designation",
    "Carrosserie": "Body",
    "masse vide euro min (kg)": "Empty Mass Euro Min (kg)",
    "masse vide euro max (kg)": "Empty Mass Euro Max (kg)",
    "Boîte de vitesse": "Gearbox",
    "Type de boîte de vitesse": "Gearbox Type",
    "Energie": "Fuel",
    "Gamme": "Range",
    "Puissance maximale (kW)": "Maximum Power (kW)",
    "Consommation mixte (l/100km)": "Combined Consumption (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)",
    "HC (g/km)": "HC (g/km)",
    "NOX (g/km)": "NOX (g/km)",
    "HC+NOX (g/km)": "HC+NOX (g/km)",
    "Champ V9": "Field V9",
}

GEARBOX_CLEAN = {
    "M": "Manual", "A": "Automatic", "V": "CVT",
    "Manuel": "Manual", "Automatique": "Automatic",
    "manuelle": "Manual", "automatique": "Automatic",
}

DEDUP_KEYS = [
    "Brand", "Folder Model", "Fuel", "Body", "Gearbox",
    "Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
    "CO2 (g/km)", "Combined Consumption (l/100km)",
]

FEATURE_COLS = [
    "Maximum Power (kW)",
    "Empty Mass Euro Avg (kg)",
    "Combined Consumption (l/100km)",
    "Fuel", "Body", "Gearbox",
]
TARGET = "CO2 (g/km)"


CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)


@st.cache_data(show_spinner=False)
def load_and_preprocess(source, filename: str = "") -> pd.DataFrame:
    """source kann bytes (Upload) oder ein URL-String sein."""
    import urllib.request

    if isinstance(source, str):
        # URL → bytes laden
        with urllib.request.urlopen(source) as resp:
            source = resp.read()

    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [";", ","]:
            try:
                df = pd.read_csv(io.BytesIO(source), sep=sep,
                                 encoding=enc, low_memory=False)
                if df.shape[1] > 3:
                    df = df.rename(columns={k: v for k, v in RENAME_MAP.items()
                                            if k in df.columns})
                    if "Gearbox" in df.columns:
                        df["Gearbox"] = df["Gearbox"].map(
                            lambda x: GEARBOX_CLEAN.get(str(x).strip(), str(x).strip()))
                    if all(c in df.columns for c in ["HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)"]):
                        m = df["HC (g/km)"].isna() & df["HC+NOX (g/km)"].notna()
                        df.loc[m, "HC (g/km)"] = df.loc[m, "HC+NOX (g/km)"] / 2
                        m2 = df["NOX (g/km)"].isna() & df["HC+NOX (g/km)"].notna()
                        df.loc[m2, "NOX (g/km)"] = df.loc[m2, "HC+NOX (g/km)"] / 2
                    if "Fuel" in df.columns:
                        el = df["Fuel"].astype(str).str.upper() == "EL"
                        for col in [c for c in df.columns if
                                    any(p in c for p in ["HC", "NOX", "CO2", "Consumption"])]:
                            df.loc[el, col] = df.loc[el, col].fillna(0)
                    min_c, max_c = "Empty Mass Euro Min (kg)", "Empty Mass Euro Max (kg)"
                    if min_c in df.columns and max_c in df.columns:
                        df["Empty Mass Euro Avg (kg)"] = (
                            pd.to_numeric(df[min_c], errors="coerce") +
                            pd.to_numeric(df[max_c], errors="coerce")
                        ) / 2
                        df.drop(columns=[min_c, max_c], inplace=True, errors="ignore")
                    elif min_c in df.columns:
                        df["Empty Mass Euro Avg (kg)"] = pd.to_numeric(df[min_c], errors="coerce")
                    for col in [TARGET, "Combined Consumption (l/100km)",
                                 "Maximum Power (kW)", "Empty Mass Euro Avg (kg)"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
            except Exception:
                continue
    raise ValueError("Datei konnte nicht eingelesen werden.")


@st.cache_data(show_spinner=False)
def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if "Fuel" not in df.columns:
        return df
    df_f = df[df["Fuel"].astype(str).str.upper().isin(["ES", "GO"])].copy()
    keys = [k for k in DEDUP_KEYS if k in df_f.columns]
    return df_f.drop_duplicates(subset=keys).copy()


@st.cache_data(show_spinner=False)
def run_clustering(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    num_f = [c for c in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)"] if c in df.columns]
    cat_f = [c for c in ["Body", "Fuel", "Gearbox"] if c in df.columns]
    if not num_f:
        return df
    df_c = df.copy()
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
def train_models(df_bytes: bytes):
    df = pd.read_csv(io.BytesIO(df_bytes))
    avail = [f for f in FEATURE_COLS if f in df.columns]
    df_m = df.dropna(subset=[TARGET]).copy()
    for col in [c for c in avail if df_m[c].dtype in [np.float64, float]]:
        df_m[col] = df_m[col].fillna(df_m[col].median())
    cat_cols = [c for c in avail if df_m[c].dtype == object]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_m[col] = le.fit_transform(df_m[col].astype(str).fillna("Unknown"))
        encoders[col] = le
    X = df_m[avail].values
    y = df_m[TARGET].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    Xs_tr, Xs_te = train_test_split(X_s, test_size=0.2, random_state=42)[0], \
                   train_test_split(X_s, test_size=0.2, random_state=42)[1]
    models = {
        "Ridge": (Ridge(alpha=1.0), Xs_tr, Xs_te),
        "Random Forest": (RandomForestRegressor(200, random_state=42, n_jobs=-1), X_tr, X_te),
        "Gradient Boosting": (GradientBoostingRegressor(200, learning_rate=0.05,
                                                         random_state=42), X_tr, X_te),
    }
    results, best_name, best_mae, best_model = {}, None, 1e9, None
    best_Xte, best_yte = None, None
    for name, (model, Xtr_, Xte_) in models.items():
        model.fit(Xtr_, y_tr)
        p = model.predict(Xte_)
        mae = mean_absolute_error(y_te, p)
        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, p)), 2),
            "R²": round(r2_score(y_te, p), 4),
        }
        if mae < best_mae:
            best_mae, best_name, best_model = mae, name, model
            best_Xte, best_yte = Xte_, y_te
    fi = {}
    if hasattr(best_model, "feature_importances_"):
        fi = dict(zip(avail, best_model.feature_importances_))
    return best_model, best_name, results, fi, avail, encoders, scaler, best_Xte, best_yte


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/CO2_Logo.svg/120px-CO2_Logo.svg.png",
             width=80)
    st.title("🚗 CO₂ Dashboard")
    st.markdown("**ADEME Car Labelling Dataset**")
    st.markdown("---")

    st.caption("Datensatz wird automatisch aus dem Repo geladen.")
    uploaded = st.file_uploader(
        "Eigene CSV hochladen (optional)",
        type=["csv"],
        help="Überschreibt den Standard-Datensatz aus dem Repo.",
    )

    st.markdown("---")
    st.markdown(
        "**Projekt:** [GitHub ↗](https://github.com/cknogler/Vehicle-CO2-Emissions-Prediction)",
        unsafe_allow_html=True,
    )
    st.caption("Analyse: Preprocessing · EDA · Clustering · ML")


# ── Daten laden ────────────────────────────────────────────────────────────
# Priorität: manueller Upload > CSV direkt aus GitHub-Repo
if uploaded is not None:
    source = uploaded.read()
    source_name = uploaded.name
else:
    source = CSV_URL
    source_name = "cl_JUIN_2013-complet3.csv"

with st.spinner("Daten werden geladen und vorverarbeitet …"):
    try:
        df_clean = load_and_preprocess(source, source_name)
        df_unique = deduplicate(df_clean)
    except Exception as e:
        st.error(f"Fehler beim Laden: {e}")
        st.info("Stelle sicher, dass `cl_JUIN_2013-complet3.csv` im Root des Repos liegt.")
        st.stop()

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Überblick",
    "📊 EDA",
    "🔵 Clustering",
    "🤖 Prediction",
    "🎯 CO₂-Rechner",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – ÜBERBLICK / PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📋 Datensatz-Überblick")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rohdatensatz (Zeilen)", f"{len(df_clean):,}")
    c2.metric("Spalten", len(df_clean.columns))
    es_go = df_clean[df_clean["Fuel"].astype(str).str.upper().isin(["ES", "GO"])] \
        if "Fuel" in df_clean.columns else df_clean
    c3.metric("ES+GO Fahrzeuge", f"{len(es_go):,}")
    c4.metric("Unique Konfigurationen", f"{len(df_unique):,}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Erste Zeilen (bereinigt)")
        st.dataframe(df_clean.head(10), use_container_width=True)

    with col_r:
        st.subheader("Fehlende Werte")
        missing = df_clean.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False).head(20)
        if len(missing) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            missing.plot(kind="barh", ax=ax, color=PALETTE[0])
            ax.set_xlabel("Anzahl fehlender Werte")
            ax.set_title("Top fehlende Werte")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.success("Keine fehlenden Werte nach Preprocessing!")

    st.subheader("Deskriptive Statistik (numerisch)")
    num_df = df_clean.select_dtypes(include=np.number)
    st.dataframe(num_df.describe().T.style.format("{:.2f}"), use_container_width=True)

    if "Fuel" in df_clean.columns:
        st.subheader("Kraftstoff-Verteilung (Rohdatensatz)")
        fuel_counts = df_clean["Fuel"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 3))
        fuel_counts.plot(kind="bar", ax=ax, color=PALETTE)
        ax.set_xlabel("Kraftstofftyp"); ax.set_ylabel("Anzahl")
        ax.set_title("Häufigkeiten Kraftstofftypen")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    dedup_pct = (1 - len(df_unique) / len(es_go)) * 100 if len(es_go) > 0 else 0
    st.info(f"**Deduplication:** {len(es_go):,} ES/GO-Fahrzeuge → "
            f"**{len(df_unique):,} unique Konfigurationen** "
            f"({dedup_pct:.1f}% Duplikate entfernt)")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")

    # --- CO₂ Verteilung ---
    if TARGET in df_unique.columns:
        st.subheader("CO₂-Verteilung (dedupliziert)")
        co2 = df_unique[TARGET].dropna()

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].hist(co2, bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
        axes[0].set_title("Histogramm"); axes[0].set_xlabel("CO₂ (g/km)")

        axes[1].boxplot(co2, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=PALETTE[1], alpha=0.7))
        axes[1].set_title("Boxplot"); axes[1].set_ylabel("CO₂ (g/km)")

        stats.probplot(co2, dist="norm", plot=axes[2])
        axes[2].set_title("Q-Q Plot")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mittelwert CO₂", f"{co2.mean():.1f} g/km")
        col2.metric("Median CO₂", f"{co2.median():.1f} g/km")
        col3.metric("Min CO₂", f"{co2.min():.0f} g/km")
        col4.metric("Max CO₂", f"{co2.max():.0f} g/km")

    st.markdown("---")

    # --- Korrelation ---
    st.subheader("Korrelationsmatrix (Pearson)")
    num_cols = [c for c in [TARGET, "Maximum Power (kW)",
                              "Empty Mass Euro Avg (kg)",
                              "Combined Consumption (l/100km)"]
                if c in df_unique.columns]
    if len(num_cols) >= 2:
        corr = df_unique[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, square=True)
        ax.set_title("Pearson-Korrelation")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # --- Scatter-Plots ---
    st.subheader("CO₂ vs. Schlüsselvariablen")
    scatter_vars = [c for c in ["Combined Consumption (l/100km)",
                                  "Empty Mass Euro Avg (kg)",
                                  "Maximum Power (kW)"] if c in df_unique.columns]

    if scatter_vars and TARGET in df_unique.columns:
        fig, axes = plt.subplots(1, len(scatter_vars), figsize=(5 * len(scatter_vars), 4))
        if len(scatter_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, scatter_vars):
            d = df_unique[[var, TARGET]].dropna()
            ax.scatter(d[var], d[TARGET], alpha=0.3, s=8, color=PALETTE[0])
            m, b = np.polyfit(d[var], d[TARGET], 1)
            x_line = np.linspace(d[var].min(), d[var].max(), 100)
            ax.plot(x_line, m * x_line + b, color=PALETTE[3], lw=2)
            r, p = stats.pearsonr(d[var], d[TARGET])
            ax.set_title(f"r = {r:.2f}")
            ax.set_xlabel(var); ax.set_ylabel("CO₂ (g/km)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("---")

    # --- Boxplots Kategorisch ---
    st.subheader("CO₂ nach Kategorie")
    cat_choice = st.selectbox("Kategorie auswählen:",
                               [c for c in ["Fuel", "Body", "Gearbox"] if c in df_unique.columns])
    if cat_choice and TARGET in df_unique.columns:
        top_cats = df_unique[cat_choice].value_counts().head(8).index
        df_box = df_unique[df_unique[cat_choice].isin(top_cats)]
        fig, ax = plt.subplots(figsize=(10, 5))
        order = df_box.groupby(cat_choice)[TARGET].median().sort_values().index
        sns.boxplot(data=df_box, x=cat_choice, y=TARGET, order=order,
                    palette=PALETTE, ax=ax)
        ax.set_title(f"CO₂ nach {cat_choice}")
        ax.set_xlabel(cat_choice); ax.set_ylabel("CO₂ (g/km)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # --- Hypothesis Test ---
    st.markdown("---")
    st.subheader("Statistischer Hypothesentest: Petrol vs. Diesel")
    if "Fuel" in df_unique.columns and TARGET in df_unique.columns:
        es = df_unique[df_unique["Fuel"].astype(str).str.upper() == "ES"][TARGET].dropna()
        go = df_unique[df_unique["Fuel"].astype(str).str.upper() == "GO"][TARGET].dropna()
        if len(es) > 0 and len(go) > 0:
            stat, pval = stats.mannwhitneyu(es, go, alternative="two-sided")
            col1, col2, col3 = st.columns(3)
            col1.metric("Ø CO₂ Petrol (ES)", f"{es.mean():.1f} g/km")
            col2.metric("Ø CO₂ Diesel (GO)", f"{go.mean():.1f} g/km")
            col3.metric("Mann-Whitney p-Wert", f"{pval:.2e}")
            if pval < 0.05:
                st.success("✅ Signifikanter Unterschied (p < 0.05) – Kraftstofftyp beeinflusst CO₂")
            else:
                st.warning("Kein signifikanter Unterschied gefunden.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🔵 Clustering – Fahrzeugsegmente")

    k = st.slider("Anzahl Cluster (k)", min_value=2, max_value=8, value=4)
    df_cl = run_clustering(df_unique, k=k)

    if "Cluster" in df_cl.columns:
        # Cluster-Übersicht
        st.subheader("Cluster-Größen")
        sizes = df_cl["Cluster"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        sizes.plot(kind="bar", ax=ax, color=PALETTE[:k])
        ax.set_xlabel("Cluster"); ax.set_ylabel("Anzahl Fahrzeuge")
        ax.set_title("Fahrzeuge pro Cluster")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Cluster-Profile
        num_profile_cols = [c for c in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)", TARGET]
                            if c in df_cl.columns]
        if num_profile_cols:
            st.subheader("Cluster-Profile (Mittelwerte)")
            profile = df_cl.groupby("Cluster")[num_profile_cols].mean().round(1)
            st.dataframe(profile.style.background_gradient(cmap="Blues"), use_container_width=True)

        # Scatter Power vs. Mass
        if all(c in df_cl.columns for c in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)"]):
            st.subheader("Cluster-Visualisierung: Power vs. Masse")
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, cluster in enumerate(sorted(df_cl["Cluster"].unique())):
                mask = df_cl["Cluster"] == cluster
                ax.scatter(df_cl.loc[mask, "Maximum Power (kW)"],
                           df_cl.loc[mask, "Empty Mass Euro Avg (kg)"],
                           alpha=0.4, s=12, label=f"Cluster {cluster}",
                           color=PALETTE[i % len(PALETTE)])
            ax.set_xlabel("Maximum Power (kW)")
            ax.set_ylabel("Empty Mass Euro Avg (kg)")
            ax.set_title("Fahrzeugsegmente nach Leistung & Masse")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # CO₂ per Cluster
        if TARGET in df_cl.columns:
            st.subheader("CO₂-Verteilung je Cluster")
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, cl in enumerate(sorted(df_cl["Cluster"].unique())):
                co2_cl = df_cl[df_cl["Cluster"] == cl][TARGET].dropna()
                ax.hist(co2_cl, bins=30, alpha=0.6, label=f"Cluster {cl}",
                        color=PALETTE[i % len(PALETTE)])
            ax.set_xlabel("CO₂ (g/km)"); ax.set_ylabel("Häufigkeit")
            ax.set_title("CO₂ nach Cluster")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Heatmap normalisiert
        if num_profile_cols:
            st.subheader("Normalisiertes Cluster-Profil (Heatmap)")
            profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.heatmap(profile_norm, annot=profile.values, fmt=".0f",
                        cmap="YlOrRd", ax=ax, linewidths=0.5)
            ax.set_title("Normalisierter Cluster-Vergleich")
            plt.tight_layout()
            st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🤖 Predictive Modeling")

    avail_features = [f for f in FEATURE_COLS if f in df_unique.columns]
    if TARGET not in df_unique.columns or len(avail_features) < 2:
        st.warning("Nicht genug Features oder Zielvariable für Modelltraining.")
    else:
        with st.spinner("Modelle werden trainiert (Ridge / RF / GBR) …"):
            df_bytes = df_unique.to_csv(index=False).encode()
            (best_model, best_name, results, fi,
             features, encoders, scaler, X_te, y_te) = train_models(df_bytes)

        st.subheader("Modellvergleich")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.style.highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9")
                                     .highlight_max(subset=["R²"], color="#c8e6c9")
                                     .format("{:.3f}"), use_container_width=True)
        st.success(f"✅ Bestes Modell: **{best_name}** (MAE = {results[best_name]['MAE']} g/km)")

        # Feature Importance
        if fi:
            st.subheader("Feature Importance")
            fi_df = pd.Series(fi).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            fi_df.plot(kind="barh", ax=ax, color=PALETTE[0])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance – {best_name}")
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Predicted vs. Actual
        st.subheader("Predicted vs. Actual (Test-Set)")
        preds = best_model.predict(X_te)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_te, preds, alpha=0.3, s=12, color=PALETTE[0])
        lims = [min(y_te.min(), preds.min()) - 10, max(y_te.max(), preds.max()) + 10]
        ax.plot(lims, lims, "r--", lw=1.5, label="Perfekte Vorhersage")
        ax.set_xlabel("Tatsächlicher CO₂-Wert (g/km)")
        ax.set_ylabel("Vorhergesagter CO₂-Wert (g/km)")
        ax.set_title(f"{best_name} – Predicted vs. Actual")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Residuals
        residuals = y_te - preds
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(preds, residuals, alpha=0.3, s=10, color=PALETTE[2])
        axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
        axes[0].set_xlabel("Vorhersage (g/km)"); axes[0].set_ylabel("Residual")
        axes[0].set_title("Residuals vs. Fitted")
        axes[1].hist(residuals, bins=50, color=PALETTE[1], edgecolor="white", alpha=0.85)
        axes[1].set_xlabel("Residual (g/km)"); axes[1].set_ylabel("Häufigkeit")
        axes[1].set_title("Residuals-Verteilung")
        plt.tight_layout()
        st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 – CO₂-RECHNER
# ═══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🎯 CO₂-Rechner")
    st.markdown("Gib Fahrzeugparameter ein und erhalte eine CO₂-Vorhersage.")

    avail_features = [f for f in FEATURE_COLS if f in df_unique.columns]
    if TARGET not in df_unique.columns or len(avail_features) < 2:
        st.warning("Modell nicht verfügbar – bitte zuerst Tab 'Prediction' öffnen.")
    else:
        df_bytes = df_unique.to_csv(index=False).encode()
        (best_model, best_name, results, fi,
         features, encoders, scaler, X_te, y_te) = train_models(df_bytes)

        with st.form("co2_form"):
            col1, col2 = st.columns(2)

            inputs = {}
            with col1:
                if "Maximum Power (kW)" in features:
                    inputs["Maximum Power (kW)"] = st.number_input(
                        "Motorleistung (kW)", min_value=30, max_value=600,
                        value=110, step=5)
                if "Empty Mass Euro Avg (kg)" in features:
                    inputs["Empty Mass Euro Avg (kg)"] = st.number_input(
                        "Leergewicht (kg)", min_value=700, max_value=4000,
                        value=1400, step=25)
                if "Combined Consumption (l/100km)" in features:
                    inputs["Combined Consumption (l/100km)"] = st.number_input(
                        "Kraftstoffverbrauch kombiniert (l/100km)",
                        min_value=2.0, max_value=30.0, value=7.0, step=0.1)

            with col2:
                cat_options = {
                    "Fuel": ["GO", "ES"],
                    "Body": sorted(df_unique["Body"].dropna().unique().tolist())
                             if "Body" in df_unique.columns else [],
                    "Gearbox": sorted(df_unique["Gearbox"].dropna().unique().tolist())
                                if "Gearbox" in df_unique.columns else [],
                }
                for cat_col, options in cat_options.items():
                    if cat_col in features and options:
                        inputs[cat_col] = st.selectbox(cat_col, options)

            submitted = st.form_submit_button("CO₂ berechnen 🚀")

        if submitted:
            # Feature-Vektor bauen
            row = {}
            for feat in features:
                row[feat] = inputs.get(feat, 0)
            df_input = pd.DataFrame([row])

            # Kategorien encodieren
            for col, le in encoders.items():
                if col in df_input.columns:
                    val = str(df_input[col].iloc[0])
                    if val not in le.classes_:
                        val = le.classes_[0]
                    df_input[col] = le.transform([val])

            X_input = df_input[features].values
            prediction = best_model.predict(X_input)[0]

            # Ergebnis
            st.markdown("---")
            euro_class = (
                "A (≤100 g/km)" if prediction <= 100 else
                "B (101–120)" if prediction <= 120 else
                "C (121–140)" if prediction <= 140 else
                "D (141–160)" if prediction <= 160 else
                "E (161–200)" if prediction <= 200 else
                "F/G (>200)"
            )
            color = (
                "green" if prediction <= 120 else
                "orange" if prediction <= 160 else
                "red"
            )

            st.markdown(f"""
            <div style="background:{color}22; border-left:5px solid {color};
                        padding:20px; border-radius:8px;">
                <h2 style="color:{color}">🚗 Vorhergesagter CO₂-Ausstoß: {prediction:.1f} g/km</h2>
                <p style="font-size:18px">EU-Effizienzklasse: <strong>{euro_class}</strong></p>
                <p style="color:gray; font-size:13px">Modell: {best_name} | MAE ≈ {results[best_name]['MAE']} g/km</p>
            </div>
            """, unsafe_allow_html=True)

            # Einordnung im Fleet
            if TARGET in df_unique.columns:
                pct = (df_unique[TARGET] <= prediction).mean() * 100
                st.metric("Einordnung im Datensatz",
                          f"Top {100-pct:.0f}% | besser als {pct:.0f}% der Fahrzeuge")

                fig, ax = plt.subplots(figsize=(9, 3))
                co2_all = df_unique[TARGET].dropna()
                ax.hist(co2_all, bins=60, color=PALETTE[0], alpha=0.6, label="Alle Fahrzeuge")
                ax.axvline(prediction, color="red", lw=2.5, linestyle="--",
                           label=f"Deine Eingabe: {prediction:.0f} g/km")
                ax.set_xlabel("CO₂ (g/km)"); ax.set_ylabel("Häufigkeit")
                ax.set_title("Einordnung im Flottenvergleich")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig); plt.close()
