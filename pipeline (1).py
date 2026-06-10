"""
pipeline.py – Vehicle CO₂ Emissions · Data Pipeline & Model Training
ADEME Car Labelling Dataset (cl_JUIN_2013-complet3.csv)

All data loading, preprocessing, feature engineering, clustering and
model training lives here. app.py imports only the public functions.

Public API
----------
load_data(source)          → df_raw, df_unique
train_models(df_unique)    → ModelBundle (dataclass)
run_clustering(df_unique)  → df_with_cluster
"""

from __future__ import annotations
import io
import urllib.request
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from kmodes.kprototypes import KPrototypes
    KPROTO_AVAILABLE = True
except ImportError:
    KPROTO_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

CSV_URL = (
    "https://raw.githubusercontent.com/cknogler/"
    "Vehicle-CO2-Emissions-Prediction/main/cl_JUIN_2013-complet3.csv"
)

RANDOM_STATE = 42

COLUMN_MAPPING = {
    "Marque": "Brand", "Modèle dossier": "Folder Model",
    "Modèle UTAC": "Utac Model", "Désignation commerciale": "Commercial Designation",
    "CNIT": "cnit", "Type Variante Version (TVV)": "Type Variant Version",
    "Carburant": "Fuel", "Hybride": "Hybrid",
    "Puissance administrative": "Administrative Power",
    "Puissance maximale (kW)": "Maximum Power (kW)",
    "Boîte de vitesse": "Gearbox",
    "Consommation urbaine (l/100km)": "Urban Consumption (l/100km)",
    "Consommation extra-urbaine (l/100km)": "Extra Urban Consumption (l/100km)",
    "Consommation mixte (l/100km)": "Combined Consumption (l/100km)",
    "CO2 (g/km)": "CO2 (g/km)", "CO type I (g/km)": "CO type 1 (g/km)",
    "HC (g/km)": "HC (g/km)", "NOX (g/km)": "NOX (g/km)",
    "HC+NOX (g/km)": "HC+NOX (g/km)", "Particules (g/km)": "Particles (g/km)",
    "masse vide euro min (kg)": "Empty Mass Euro Min (kg)",
    "masse vide euro max (kg)": "Empty Mass Euro Max (kg)",
    "Champ V9": "Field V9", "Date de mise à jour": "Update Date",
    "Carrosserie": "Body", "gamme": "Range",
}

DEDUP_COLS = [
    "Brand", "Folder Model", "Fuel", "Body", "Gearbox",
    "Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
    "CO2 (g/km)", "Combined Consumption (l/100km)", "Range",
]

GEAR_TYPE_MAP = {
    "M": "Manual", "A": "Automatic", "V": "CVT",
    "D": "DCT",    "N": "Automatic", "S": "Manual",
}

FEATURE_SETS = {
    "all_features":    ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel", "GearType", "GearCount", "Body"],
    "no_body":         ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel", "GearType", "GearCount"],
    "mass_power_fuel": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel"],
    "mass_power_only": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)"],
}

TARGET = "CO2 (g/km)"


# ── Step 1 · Load & preprocess raw CSV ──────────────────────────────────────

def _read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("latin1", "utf-8", "cp1252"):
        for sep in (";", ","):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    return df
            except Exception:
                continue
    raise ValueError("CSV could not be parsed with any known encoding/separator.")


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    # HC/NOX imputation via sum-column
    if all(c in df.columns for c in ("HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)")):
        hc  = (df["HC+NOX (g/km)"] - df["NOX (g/km)"]).fillna(df["HC (g/km)"])
        nox = (df["HC+NOX (g/km)"] - df["HC (g/km)"]).fillna(df["NOX (g/km)"])
        df["HC (g/km)"], df["NOX (g/km)"] = hc, nox
        df["HC+NOX (g/km)"] = hc + nox

    # Gearbox entry-error corrections
    if "Gearbox" in df.columns:
        df["Gearbox"] = df["Gearbox"].replace({"N 0": "A 0", "N 1": "A 0", "S 6": "D 6"})

    # Electric vehicles → 0 emissions (correct, not missing)
    ev_cols = [
        "CO type 1 (g/km)", "Urban Consumption (l/100km)",
        "Extra Urban Consumption (l/100km)", "Combined Consumption (l/100km)",
        "CO2 (g/km)", "HC+NOX (g/km)", "HC (g/km)", "Particles (g/km)",
    ]
    if "Fuel" in df.columns:
        mask_ev = df["Fuel"] == "EL"
        for c in ev_cols:
            if c in df.columns:
                df.loc[mask_ev, c] = df.loc[mask_ev, c].fillna(0)

    # Kerb weight → single average column
    if "Empty Mass Euro Min (kg)" in df.columns and "Empty Mass Euro Max (kg)" in df.columns:
        df["Empty Mass Euro Avg (kg)"] = (
            pd.to_numeric(df["Empty Mass Euro Min (kg)"], errors="coerce")
            + pd.to_numeric(df["Empty Mass Euro Max (kg)"], errors="coerce")
        ) / 2
        df.drop(columns=["Empty Mass Euro Min (kg)", "Empty Mass Euro Max (kg)"], inplace=True)

    # Numeric coercion
    for col in (TARGET, "Combined Consumption (l/100km)", "Maximum Power (kW)", "Empty Mass Euro Avg (kg)"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _make_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to unique mechanical configurations (ES + GO only)."""
    df_c = df[df["Fuel"].isin(["ES", "GO"])].copy()

    if "Gearbox" in df_c.columns:
        gs = df_c["Gearbox"].astype(str).str.split(" ", expand=True)
        df_c["GearType"]  = gs[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_c["GearCount"] = pd.to_numeric(gs[1] if 1 in gs.columns else pd.Series(dtype=float), errors="coerce")

    cols = [c for c in DEDUP_COLS if c in df_c.columns]
    df_u = (
        df_c.groupby(cols, dropna=False).size()
        .reset_index(name="Clone_Count")
        .sort_values("Clone_Count", ascending=False)
        .reset_index(drop=True)
    )

    # Re-attach GearType / GearCount after groupby
    if "Gearbox" in df_u.columns:
        gs2 = df_u["Gearbox"].astype(str).str.split(" ", expand=True)
        df_u["GearType"]  = gs2[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_u["GearCount"] = pd.to_numeric(gs2[1] if 1 in gs2.columns else pd.Series(dtype=float), errors="coerce")

    return df_u


def load_data(source=CSV_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the ADEME dataset.

    Parameters
    ----------
    source : str | bytes
        URL string or raw CSV bytes (from st.file_uploader).

    Returns
    -------
    df_raw    : full preprocessed dataset (all fuel types)
    df_unique : deduplicated ES/GO configurations with GearType/GearCount
    """
    if isinstance(source, str):
        with urllib.request.urlopen(source) as r:
            raw = r.read()
    else:
        raw = bytes(source)

    df_raw    = _preprocess(_read_csv(raw))
    df_unique = _make_unique(df_raw)
    return df_raw, df_unique


# ── Step 2 · Model training ──────────────────────────────────────────────────

@dataclass
class ModelBundle:
    """All trained artifacts needed by the Streamlit app."""
    fitted:       dict          # name → fitted Pipeline
    results:      pd.DataFrame  # Model / Train_R2 / Test_R2 / Train_MAE / Test_MAE
    feature_cols: list[str]     # winning feature set columns
    best_fs:      str           # winning feature set name
    fs_comparison:pd.DataFrame  # CV results for all four feature sets
    X_train:      pd.DataFrame
    X_test:       pd.DataFrame
    y_train:      pd.Series
    y_test:       pd.Series
    rf_pipe:      Pipeline      # Random Forest pipeline (for PDP, feature importance)
    feature_importance: pd.DataFrame  # Feature / Importance
    numeric_features:   list[str]
    categorical_features: list[str]


def train_models(df_unique: pd.DataFrame) -> ModelBundle:
    """
    Cross-validate four feature sets, pick the best, then train five regressors.
    Returns a ModelBundle with all artifacts cached for the Streamlit app.
    """
    all_cols = sorted({TARGET} | {c for cols in FEATURE_SETS.values() for c in cols})
    df_m = df_unique[[c for c in all_cols if c in df_unique.columns]].dropna().copy()

    def split_types(feats):
        num = df_m[feats].select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat = df_m[feats].select_dtypes(include=["object", "category"]).columns.tolist()
        return num, cat

    def preprocessors(num, cat):
        scaled = ColumnTransformer([
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
            ("num", StandardScaler(), num),
        ])
        tree = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", "passthrough", num),
        ])
        return scaled, tree

    # ── Feature-set CV ───────────────────────────────────────────────────────
    fs_rows = []
    for name, feats in FEATURE_SETS.items():
        avail = [f for f in feats if f in df_m.columns]
        if not avail:
            continue
        _, tree_pre = preprocessors(*split_types(avail))
        pipe = Pipeline([("pre", tree_pre),
                          ("m", RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1))])
        scores = cross_val_score(pipe, df_m[avail], df_m[TARGET],
                                  cv=5, scoring="neg_mean_absolute_error")
        fs_rows.append({
            "Feature_Set": name, "Features": ", ".join(avail),
            "CV_MAE_mean": -scores.mean(), "CV_MAE_std": scores.std(),
        })

    fs_df   = pd.DataFrame(fs_rows).sort_values("CV_MAE_mean")
    best_fs = fs_df.iloc[0]["Feature_Set"]
    feat_cols = [f for f in FEATURE_SETS[best_fs] if f in df_m.columns]

    X, y = df_m[feat_cols], df_m[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    num_f, cat_f = split_types(feat_cols)
    scaled_pre, tree_pre = preprocessors(num_f, cat_f)

    model_defs = {
        "Linear Regression": Pipeline([("pre", scaled_pre), ("m", LinearRegression())]),
        "Ridge":             Pipeline([("pre", scaled_pre), ("m", Ridge(alpha=1.0))]),
        "Lasso":             Pipeline([("pre", scaled_pre), ("m", Lasso(alpha=0.1))]),
        "Random Forest":     Pipeline([("pre", tree_pre),
                                        ("m", RandomForestRegressor(
                                            n_estimators=300, max_depth=20,
                                            max_features=0.8, random_state=RANDOM_STATE, n_jobs=-1,
                                        ))]),
        "Gradient Boosting": Pipeline([("pre", tree_pre),
                                        ("m", GradientBoostingRegressor(
                                            n_estimators=200, learning_rate=0.2,
                                            max_depth=6, subsample=1.0,
                                            max_features=0.5, random_state=RANDOM_STATE,
                                        ))]),
    }

    # ── Train & evaluate ─────────────────────────────────────────────────────
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

    # ── Random Forest feature importance ─────────────────────────────────────
    rf_pipe  = fitted["Random Forest"]
    fi_names = rf_pipe.named_steps["pre"].get_feature_names_out()
    fi_vals  = rf_pipe.named_steps["m"].feature_importances_
    fi_df    = (
        pd.DataFrame({"Feature": fi_names, "Importance": fi_vals})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return ModelBundle(
        fitted=fitted, results=results_df, feature_cols=feat_cols,
        best_fs=best_fs, fs_comparison=fs_df,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        rf_pipe=rf_pipe, feature_importance=fi_df,
        numeric_features=num_f, categorical_features=cat_f,
    )


# ── Step 3 · Clustering ──────────────────────────────────────────────────────

def run_clustering(df_unique: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    K-Prototypes clustering on mixed numeric + categorical features.
    Returns df_unique slice (no NaNs) with an added 'Cluster' column.
    Requires kmodes >= 0.12.2.
    """
    if not KPROTO_AVAILABLE:
        raise RuntimeError("kmodes is not installed. Add 'kmodes>=0.12.2' to requirements.txt.")

    cat_cols = [c for c in ("Body", "Fuel", "Gearbox") if c in df_unique.columns]
    num_cols = [c for c in ("Maximum Power (kW)", "Empty Mass Euro Avg (kg)") if c in df_unique.columns]
    feat_cols = cat_cols + num_cols

    df_c   = df_unique[feat_cols + [TARGET]].dropna().copy()
    df_kp  = df_c.copy()
    scaler = StandardScaler()
    df_kp[num_cols] = scaler.fit_transform(df_kp[num_cols])
    for c in cat_cols:
        df_kp[c] = df_kp[c].astype(str)

    X_mat  = df_kp[feat_cols].to_numpy(dtype=object)
    cat_ix = [feat_cols.index(c) for c in cat_cols]

    kp = KPrototypes(n_clusters=k, init="Cao", n_init=5, verbose=0, random_state=RANDOM_STATE)
    df_c["Cluster"] = kp.fit_predict(X_mat, categorical=cat_ix)
    return df_c


# ── Convenience helpers (used by analysis notebooks / app) ──────────────────

def segment_filter(
    df_unique: pd.DataFrame,
    fuel: str,
    body: str,
    kw_lo: float = 0,
    kw_hi: float = 9999,
    gear_prefix: str | None = None,
) -> pd.DataFrame:
    """Return rows matching the given segment criteria."""
    mask = (
        df_unique["Fuel"].eq(fuel)
        & df_unique["Body"].eq(body)
        & df_unique["Maximum Power (kW)"].between(kw_lo, kw_hi)
    )
    if gear_prefix:
        mask &= df_unique["Gearbox"].astype(str).str.startswith(gear_prefix)
    return df_unique[mask].copy()


def predict_co2(bundle: ModelBundle, row_dict: dict, model_name: str = "Random Forest") -> float:
    """Predict CO₂ for a single feature row dict."""
    return float(bundle.fitted[model_name].predict(pd.DataFrame([row_dict]))[0])
