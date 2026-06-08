"""
pipeline.py
===========
Vehicle CO₂ Emissions – Data Processing & ML Pipeline
ADEME Car Labelling Dataset (cl_JUIN_2013-complet3.csv)

Usage:
    python pipeline.py --data cl_JUIN_2013-complet3.csv
    python pipeline.py --data cl_JUIN_2013-complet3.csv --output results/

Output:
    results/df_clean.csv        – cleaned dataset
    results/df_unique.csv       – deduplicated dataset (ES+GO)
    results/df_clustered.csv    – with cluster labels (K-Prototypes)
    results/model_best.pkl      – best trained model (pipeline object)
    results/model_meta.json     – metrics, feature importances, hyperparameters
"""

import argparse
import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ── Column mapping (French → English) ────────────────────────────────────────
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

GEAR_TYPE_MAP = {
    "M": "Manual", "A": "Automatic", "V": "CVT",
    "D": "DCT",    "N": "Automatic", "S": "Manual",
}

UNIQUE_COLS = [
    "Brand", "Folder Model", "Fuel", "Body", "Gearbox",
    "Maximum Power (kW)", "Empty Mass Euro Avg (kg)",
    "CO2 (g/km)", "Combined Consumption (l/100km)", "Range",
]

FEATURE_SETS = {
    "all_features":    ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)",
                        "Fuel", "GearType", "GearCount", "Body"],
    "no_body":         ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)",
                        "Fuel", "GearType", "GearCount"],
    "mass_power_fuel": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)", "Fuel"],
    "mass_power_only": ["Empty Mass Euro Avg (kg)", "Maximum Power (kW)"],
}

TARGET = "CO2 (g/km)"


# ── 1. LOAD ───────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV – auto-detects encoding and separator."""
    for enc in ["latin1", "utf-8", "cp1252"]:
        for sep in [";", ","]:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding=enc, low_memory=False)
                if df.shape[1] > 5:
                    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} cols "
                          f"(sep='{sep}', enc='{enc}')")
                    return df
            except Exception:
                continue
    raise ValueError(f"Could not read {filepath}")


# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Rename, impute, clean and engineer features."""
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items()
                             if k in df.columns})

    # HC/NOX imputation from sum
    if all(c in df.columns for c in ["HC (g/km)", "NOX (g/km)", "HC+NOX (g/km)"]):
        df["hc_c"]  = df["HC+NOX (g/km)"] - df["NOX (g/km)"]
        df["nox_c"] = df["HC+NOX (g/km)"] - df["HC (g/km)"]
        df["hc_c"]  = df["hc_c"].fillna(df["HC (g/km)"])
        df["nox_c"] = df["nox_c"].fillna(df["NOX (g/km)"])
        df["HC (g/km)"]     = df["hc_c"]
        df["NOX (g/km)"]    = df["nox_c"]
        df["HC+NOX (g/km)"] = df["hc_c"] + df["nox_c"]
        df.drop(columns=["hc_c", "nox_c"], inplace=True)

    # Gearbox fixes
    if "Gearbox" in df.columns:
        df["Gearbox"] = df["Gearbox"].replace({"N 0": "A 0", "N 1": "A 0", "S 6": "D 6"})

    # Electric vehicles: pollutant NaN → 0
    elec_cols = ["CO type 1 (g/km)", "Urban Consumption (l/100km)",
                 "Extra Urban Consumption (l/100km)", "Combined Consumption (l/100km)",
                 "CO2 (g/km)", "HC+NOX (g/km)", "HC (g/km)", "Particles (g/km)"]
    if "Fuel" in df.columns:
        el = df["Fuel"] == "EL"
        for c in elec_cols:
            if c in df.columns:
                df.loc[el, c] = df.loc[el, c].fillna(0)

    # Average kerb weight
    min_c, max_c = "Empty Mass Euro Min (kg)", "Empty Mass Euro Max (kg)"
    if min_c in df.columns and max_c in df.columns:
        df["Empty Mass Euro Avg (kg)"] = (
            pd.to_numeric(df[min_c], errors="coerce") +
            pd.to_numeric(df[max_c], errors="coerce")
        ) / 2
        df.drop(columns=[min_c, max_c], inplace=True)

    # Numeric types
    for col in [TARGET, "Combined Consumption (l/100km)",
                "Maximum Power (kW)", "Empty Mass Euro Avg (kg)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # GearType + GearCount from Gearbox string
    if "Gearbox" in df.columns:
        gs = df["Gearbox"].astype(str).str.split(" ", expand=True)
        df["GearType"]  = gs[0].map(GEAR_TYPE_MAP).fillna("Other")
        df["GearCount"] = pd.to_numeric(
            gs[1] if 1 in gs.columns else pd.Series([np.nan] * len(df)),
            errors="coerce"
        )

    print(f"  Preprocessed: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── 3. DEDUPLICATE ────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ES/GO and remove duplicate mechanical configurations."""
    if "Fuel" not in df.columns:
        return df

    df_combus = df[df["Fuel"].isin(["ES", "GO"])].copy()
    print(f"  ES/GO filter: {len(df_combus):,} rows")

    cols = [c for c in UNIQUE_COLS if c in df_combus.columns]
    df_unique = (
        df_combus.groupby(cols, dropna=False)
        .size()
        .reset_index(name="Clone_Count")
        .sort_values("Clone_Count", ascending=False)
        .reset_index(drop=True)
    )

    # Re-add GearType + GearCount
    if "Gearbox" in df_unique.columns:
        gs = df_unique["Gearbox"].astype(str).str.split(" ", expand=True)
        df_unique["GearType"]  = gs[0].map(GEAR_TYPE_MAP).fillna("Other")
        df_unique["GearCount"] = pd.to_numeric(
            gs[1] if 1 in gs.columns else pd.Series([np.nan] * len(df_unique)),
            errors="coerce"
        )

    pct = (1 - len(df_unique) / len(df_combus)) * 100
    print(f"  Deduplicated: {len(df_unique):,} unique configs "
          f"({pct:.1f}% duplicates removed)")
    return df_unique


# ── 4. CLUSTERING (K-Prototypes) ─────────────────────────────────────────────

def run_clustering(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """K-Prototypes clustering on mixed numeric + categorical features."""
    try:
        from kmodes.kprototypes import KPrototypes
    except ImportError:
        print("  kmodes not installed – skipping clustering. "
              "Install with: pip install kmodes")
        return df

    categorical_cols = [c for c in ["Body", "Fuel", "Gearbox"] if c in df.columns]
    numeric_cols     = [c for c in ["Maximum Power (kW)", "Empty Mass Euro Avg (kg)"]
                        if c in df.columns]
    feature_cols     = categorical_cols + numeric_cols

    df_c = df[feature_cols + [TARGET]].dropna().copy()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_kp  = df_c.copy()
    df_kp[numeric_cols] = scaler.fit_transform(df_kp[numeric_cols])
    for col in categorical_cols:
        df_kp[col] = df_kp[col].astype(str)

    X_matrix        = df_kp[feature_cols].to_numpy(dtype=object)
    categorical_idx = [feature_cols.index(c) for c in categorical_cols]

    kproto = KPrototypes(n_clusters=k, init="Cao", n_init=5,
                         verbose=0, random_state=RANDOM_STATE)
    df_c["Cluster"] = kproto.fit_predict(X_matrix, categorical=categorical_idx)

    print(f"  K-Prototypes k={k}: {dict(df_c['Cluster'].value_counts().sort_index())}")
    return df_c


# ── 5. PREDICTIVE MODELLING ───────────────────────────────────────────────────

def train_models(df: pd.DataFrame):
    """
    1. Feature-set comparison via 5-fold CV (Random Forest)
    2. Train all 5 models on best feature set
    3. Return best model, metrics, feature importances
    """
    all_needed = sorted(set(
        [TARGET] + [c for cols in FEATURE_SETS.values() for c in cols]
    ))
    df_m = df[[c for c in all_needed if c in df.columns]].dropna().copy()
    print(f"  Modelling dataset: {len(df_m):,} rows")

    def get_types(features):
        num = df_m[features].select_dtypes(include=["int64","float64"]).columns.tolist()
        cat = df_m[features].select_dtypes(include=["object","category"]).columns.tolist()
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

    # ── Feature-set comparison ────────────────────────────────────────────────
    print("\n  Feature-set comparison (5-fold CV, Random Forest):")
    fs_results = []
    for fs_name, fs_feats in FEATURE_SETS.items():
        feats = [f for f in fs_feats if f in df_m.columns]
        if not feats:
            continue
        num, cat = get_types(feats)
        _, tree_pre = preprocessors(num, cat)
        pipe = Pipeline([
            ("pre", tree_pre),
            ("m", RandomForestRegressor(200, random_state=RANDOM_STATE, n_jobs=-1))
        ])
        scores = cross_val_score(pipe, df_m[feats], df_m[TARGET],
                                 cv=5, scoring="neg_mean_absolute_error")
        mae_cv = -np.mean(scores)
        fs_results.append({"Feature_Set": fs_name, "Features": feats,
                            "CV_MAE": round(mae_cv, 3), "CV_Std": round(np.std(scores), 3)})
        print(f"    {fs_name:20s} CV MAE = {mae_cv:.2f} ± {np.std(scores):.2f}")

    best_fs   = min(fs_results, key=lambda x: x["CV_MAE"])
    feat_cols = best_fs["Features"]
    print(f"\n  Best feature set: {best_fs['Feature_Set']} "
          f"(CV MAE = {best_fs['CV_MAE']})")

    # ── Train/test split ──────────────────────────────────────────────────────
    X = df_m[feat_cols]
    y = df_m[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_STATE)
    num_f, cat_f = get_types(feat_cols)
    scaled_pre, tree_pre = preprocessors(num_f, cat_f)

    # ── Model definitions (optimised hyperparameters) ─────────────────────────
    model_defs = {
        "Linear Regression": Pipeline([("pre", scaled_pre), ("m", LinearRegression())]),
        "Ridge":             Pipeline([("pre", scaled_pre), ("m", Ridge(alpha=1.0))]),
        "Lasso":             Pipeline([("pre", scaled_pre), ("m", Lasso(alpha=0.1))]),
        "Random Forest":     Pipeline([("pre", tree_pre), ("m", RandomForestRegressor(
            n_estimators=300, max_depth=20, max_features=0.8,
            min_samples_split=2, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1
        ))]),
        "Gradient Boosting": Pipeline([("pre", tree_pre), ("m", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.2, max_depth=6,
            min_samples_split=10, subsample=1.0, max_features=0.5,
            random_state=RANDOM_STATE
        ))]),
    }

    # ── Train & evaluate ──────────────────────────────────────────────────────
    print("\n  Model comparison (80/20 split):")
    results = {}
    fitted  = {}
    best_name, best_mae_val, best_pipe = None, 1e9, None

    for name, pipe in model_defs.items():
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        y_tr_p = pipe.predict(X_tr)
        y_te_p = pipe.predict(X_te)
        train_mae = mean_absolute_error(y_tr, y_tr_p)
        test_mae  = mean_absolute_error(y_te, y_te_p)
        train_r2  = r2_score(y_tr, y_tr_p)
        test_r2   = r2_score(y_te, y_te_p)
        results[name] = {
            "Train_R2": round(train_r2, 4), "Test_R2": round(test_r2, 4),
            "Train_MAE": round(train_mae, 3), "Test_MAE": round(test_mae, 3),
        }
        print(f"    {name:20s} Test R²={test_r2:.4f}  Test MAE={test_mae:.2f}")
        if test_mae < best_mae_val:
            best_mae_val, best_name, best_pipe = test_mae, name, pipe

    print(f"\n  Best model: {best_name} (Test MAE={best_mae_val:.2f} g/km)")

    # ── Feature importance (Random Forest) ───────────────────────────────────
    rf_pipe   = fitted["Random Forest"]
    rf_pre    = rf_pipe.named_steps["pre"]
    rf_model  = rf_pipe.named_steps["m"]
    feat_names = rf_pre.get_feature_names_out()
    fi = dict(zip(feat_names.tolist(),
                  rf_model.feature_importances_.tolist()))

    meta = {
        "best_model":        best_name,
        "best_feature_set":  best_fs["Feature_Set"],
        "feature_cols":      feat_cols,
        "target":            TARGET,
        "model_results":     results,
        "feature_set_cv":    fs_results,
        "feature_importances": {k: round(v, 6) for k, v in
                                 sorted(fi.items(), key=lambda x: -x[1])[:15]},
        "hyperparameters": {
            "Random Forest": {
                "n_estimators": 300, "max_depth": 20, "max_features": 0.8,
                "min_samples_split": 2, "min_samples_leaf": 1,
            },
            "Gradient Boosting": {
                "n_estimators": 200, "learning_rate": 0.2, "max_depth": 6,
                "min_samples_split": 10, "subsample": 1.0, "max_features": 0.5,
            },
        },
    }

    return best_pipe, best_name, meta


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

def run_pipeline(data_path: str, output_dir: str = "results", k: int = 4):
    os.makedirs(output_dir, exist_ok=True)

    sep = "=" * 60

    print(f"\n{sep}\nSTEP 1 – Load data\n{sep}")
    df_raw = load_data(data_path)

    print(f"\n{sep}\nSTEP 2 – Preprocessing\n{sep}")
    df_clean = preprocess(df_raw)
    out = os.path.join(output_dir, "df_clean.csv")
    df_clean.to_csv(out, index=False)
    print(f"  Saved: {out}")

    print(f"\n{sep}\nSTEP 3 – Deduplication\n{sep}")
    df_unique = deduplicate(df_clean)
    out = os.path.join(output_dir, "df_unique.csv")
    df_unique.to_csv(out, index=False)
    print(f"  Saved: {out}")

    print(f"\n{sep}\nSTEP 4 – K-Prototypes Clustering (k={k})\n{sep}")
    df_clustered = run_clustering(df_unique, k=k)
    out = os.path.join(output_dir, "df_clustered.csv")
    df_clustered.to_csv(out, index=False)
    print(f"  Saved: {out}")

    print(f"\n{sep}\nSTEP 5 – Predictive Modelling\n{sep}")
    best_pipe, best_name, meta = train_models(df_unique)

    model_path = os.path.join(output_dir, "model_best.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best_pipe, "name": best_name,
                     "features": meta["feature_cols"]}, f)
    print(f"  Saved: {model_path}")

    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    print(f"\n{sep}\n✓ Pipeline complete!\n{sep}")
    print(f"  Output directory: {output_dir}/")
    return df_clean, df_unique, df_clustered, best_pipe, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle CO₂ Emissions Pipeline")
    parser.add_argument("--data",   required=True,
                        help="Path to CSV (cl_JUIN_2013-complet3.csv)")
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results/)")
    parser.add_argument("--k",      type=int, default=4,
                        help="Number of clusters for K-Prototypes (default: 4)")
    args = parser.parse_args()
    run_pipeline(args.data, args.output, args.k)
