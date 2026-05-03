"""
tune.py — Hyperparameter tuning for CS4168 Spotify project
Run with: uv run python tune.py
Saves:    clf_tuning_results.csv, reg_tuning_results.csv
Prints:   best params to paste into notebook.py
"""

import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import randint, uniform, loguniform

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 1. Reproduce EDA pipeline ────────────────────────────────────────────────

raw_df = pl.read_csv("tracks2026.csv")
_deduped = raw_df.unique()

numeric_cols = [
    "popularity", "duration_ms", "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
caetgorical_cols = ["track_genre", "time_signature", "mode", "key", "explicit"]

numeric_df   = _deduped.select(numeric_cols)
categorical_df = _deduped.select(caetgorical_cols)

numeric_df_indexed   = numeric_df.with_row_index("row_id")
categorical_df_indexed = categorical_df.with_row_index("row_id")
clean_numeric_df = numeric_df_indexed.drop_nulls()
clean_categorical_df = categorical_df_indexed.join(
    clean_numeric_df.select("row_id"), on="row_id", how="inner"
)
clean_numeric_df     = clean_numeric_df.drop("row_id")
clean_categorical_df = clean_categorical_df.drop("row_id")

_min_val = clean_numeric_df.select(pl.col("duration_ms").min()).item()
new_clean_numeric_df = clean_numeric_df.with_columns(
    (pl.col("duration_ms") + abs(_min_val) + 1).log().alias("duration_ms")
)

categorical_df_1 = clean_categorical_df.with_columns(
    (new_clean_numeric_df["instrumentalness"] > 0.5).cast(pl.Int8).alias("instrumentalness_binary")
)
numeric_df_1 = new_clean_numeric_df.drop("instrumentalness")

_scale_cols = ["loudness", "tempo"]
numeric_df_2 = numeric_df_1.with_columns([
    ((pl.col(c) - numeric_df_1[c].median()) /
     (numeric_df_1[c].quantile(0.75) - numeric_df_1[c].quantile(0.25))).alias(c)
    for c in _scale_cols
])

categorical_df_3 = (
    categorical_df_1
    .to_dummies(columns=["track_genre"])
    .with_columns(pl.col("explicit").cast(pl.Int8))
)

final_df = categorical_df_3.hstack(numeric_df_2)
print(f"Dataset ready: {final_df.shape[0]} rows × {final_df.shape[1]} cols\n")

# ── 2. Classification split ──────────────────────────────────────────────────

clf_median = final_df["popularity"].median()
clf_pdf = (
    final_df
    .with_columns(
        pl.when(pl.col("popularity") > clf_median).then(1).otherwise(0).alias("popularity_binary")
    )
    .drop("popularity")
    .to_pandas()
)
clf_X = clf_pdf.drop(columns=["popularity_binary"])
clf_y = clf_pdf["popularity_binary"].values
clf_X_train, clf_X_test, clf_y_train, clf_y_test = train_test_split(
    clf_X, clf_y, test_size=0.2, random_state=42, stratify=clf_y
)

# ── 3. Regression split ──────────────────────────────────────────────────────

reg_pdf = final_df.to_pandas()
reg_X = reg_pdf.drop(columns=["popularity"])
reg_y = reg_pdf["popularity"].values
reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(
    reg_X, reg_y, test_size=0.2, random_state=42
)

# ── 4. Helper ────────────────────────────────────────────────────────────────

N_ITER = 200
CV     = 5

def run_search(pipe, space, X_train, y_train, scoring, model_name, search_type="random"):
    print(f"  Tuning {model_name}...", flush=True)
    t0 = time.time()
    if search_type == "grid":
        search = GridSearchCV(pipe, space, cv=CV, scoring=scoring, n_jobs=-1, verbose=0)
    else:
        search = RandomizedSearchCV(
            pipe, space, n_iter=N_ITER, cv=CV, scoring=scoring,
            n_jobs=-1, random_state=42, verbose=0,
        )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"    best {scoring}: {search.best_score_:.4f}  [{elapsed:.0f}s]")
    return search


def extract_results(search, model_name, prefix):
    """Flatten cv_results_ into a clean DataFrame."""
    r = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in r.columns if c.startswith("param_")]
    df = r[["rank_test_score", "mean_test_score", "std_test_score"] + param_cols].copy()
    df.columns = (
        ["rank", "mean_cv_score", "std_cv_score"]
        + [c.replace(f"param_{prefix}__", "").replace("param_", "") for c in param_cols]
    )
    df.insert(0, "model", model_name)
    df = df.sort_values("rank").reset_index(drop=True)
    return df


# ── 5. Classification tuning ─────────────────────────────────────────────────

print("══════════════════════════════════════════")
print("  CLASSIFICATION  (scoring: accuracy)")
print("══════════════════════════════════════════")

lr_search = run_search(
    Pipeline([("pre", StandardScaler()), ("clf", LogisticRegression(solver="saga", max_iter=3000, random_state=42))]),
    {"clf__C": loguniform(1e-3, 1e2), "clf__penalty": ["l1", "l2"]},
    clf_X_train, clf_y_train, "accuracy", "LogisticRegression",
)

rf_clf_search = run_search(
    Pipeline([("pre", StandardScaler()), ("clf", RandomForestClassifier(random_state=42))]),
    {
        "clf__n_estimators":    randint(50, 800),
        "clf__max_depth":       [None, 5, 10, 15, 20, 25, 30],
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf":  randint(1, 10),
        "clf__max_features":    ["sqrt", "log2", 0.3, 0.5, 0.7],
    },
    clf_X_train, clf_y_train, "accuracy", "RandomForest",
)

gb_clf_search = run_search(
    Pipeline([("pre", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=42))]),
    {
        "clf__n_estimators":    randint(50, 500),
        "clf__max_depth":       randint(2, 8),
        "clf__learning_rate":   loguniform(5e-3, 5e-1),
        "clf__subsample":       uniform(0.6, 0.4),
        "clf__min_samples_split": randint(2, 20),
        "clf__max_features":    ["sqrt", "log2", 0.5, 0.7, 1.0],
    },
    clf_X_train, clf_y_train, "accuracy", "GradientBoosting",
)

clf_frames = [
    extract_results(lr_search,      "LogisticRegression", "clf"),
    extract_results(rf_clf_search,  "RandomForest",       "clf"),
    extract_results(gb_clf_search,  "GradientBoosting",   "clf"),
]
clf_results = pd.concat(clf_frames, ignore_index=True)
clf_results.to_csv("clf_tuning_results.csv", index=False)
print("\nSaved → clf_tuning_results.csv")

# ── 6. Regression tuning ─────────────────────────────────────────────────────

print("\n══════════════════════════════════════════")
print("  REGRESSION  (scoring: r2)")
print("══════════════════════════════════════════")

ridge_search = run_search(
    Pipeline([("pre", StandardScaler()), ("reg", Ridge())]),
    {"reg__alpha": np.logspace(-3, 3, 60).tolist()},
    reg_X_train, reg_y_train, "r2", "Ridge", search_type="grid",
)

rf_reg_search = run_search(
    Pipeline([("pre", StandardScaler()), ("reg", RandomForestRegressor(random_state=42))]),
    {
        "reg__n_estimators":    randint(50, 800),
        "reg__max_depth":       [None, 5, 10, 15, 20, 25, 30],
        "reg__min_samples_split": randint(2, 20),
        "reg__min_samples_leaf":  randint(1, 10),
        "reg__max_features":    ["sqrt", "log2", 0.3, 0.5, 0.7],
    },
    reg_X_train, reg_y_train, "r2", "RandomForest",
)

gb_reg_search = run_search(
    Pipeline([("pre", StandardScaler()), ("reg", GradientBoostingRegressor(random_state=42))]),
    {
        "reg__n_estimators":    randint(50, 500),
        "reg__max_depth":       randint(2, 8),
        "reg__learning_rate":   loguniform(5e-3, 5e-1),
        "reg__subsample":       uniform(0.6, 0.4),
        "reg__min_samples_split": randint(2, 20),
        "reg__max_features":    ["sqrt", "log2", 0.5, 0.7, 1.0],
    },
    reg_X_train, reg_y_train, "r2", "RandomForest Reg", search_type="random",
)
gb_reg_search.estimator.named_steps["reg"].__class__.__name__ = "GradientBoosting"

reg_frames = [
    extract_results(ridge_search,   "Ridge",            "reg"),
    extract_results(rf_reg_search,  "RandomForest",     "reg"),
    extract_results(gb_reg_search,  "GradientBoosting", "reg"),
]
reg_results = pd.concat(reg_frames, ignore_index=True)
reg_results.to_csv("reg_tuning_results.csv", index=False)
print("\nSaved → reg_tuning_results.csv")

# ── 7. Summary ───────────────────────────────────────────────────────────────

best_lr  = lr_search.best_params_
best_rfc = rf_clf_search.best_params_
best_gbc = gb_clf_search.best_params_
best_rid = ridge_search.best_params_
best_rfr = rf_reg_search.best_params_
best_gbr = gb_reg_search.best_params_

print("\n══════════════════════════════════════════")
print("  RESULTS SUMMARY")
print("══════════════════════════════════════════")
print(f"LR   clf  acc: {lr_search.best_score_:.4f}  {best_lr}")
print(f"RF   clf  acc: {rf_clf_search.best_score_:.4f}  {best_rfc}")
print(f"GB   clf  acc: {gb_clf_search.best_score_:.4f}  {best_gbc}")
print(f"Ridge reg  r2: {ridge_search.best_score_:.4f}  {best_rid}")
print(f"RF   reg  r2: {rf_reg_search.best_score_:.4f}  {best_rfr}")
print(f"GB   reg  r2: {gb_reg_search.best_score_:.4f}  {best_gbr}")

print("\n══════════════════════════════════════════")
print("  PASTE INTO NOTEBOOK")
print("══════════════════════════════════════════")

def _p(d, prefix):
    return {k.replace(f"{prefix}__", ""): v for k, v in d.items()}

lrp  = _p(best_lr,  "clf")
rfcp = _p(best_rfc, "clf")
gbcp = _p(best_gbc, "clf")
ridp = _p(best_rid, "reg")
rfrp = _p(best_rfr, "reg")
gbrp = _p(best_gbr, "reg")

def fmt(d):
    parts = []
    for k, v in d.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6g}")
        else:
            parts.append(f"{k}={v!r}")
    return ", ".join(parts)

print(f"\n# Classification")
print(f"LogisticRegression(solver='saga', max_iter=3000, random_state=42, {fmt(lrp)})")
print(f"RandomForestClassifier(random_state=42, {fmt(rfcp)})")
print(f"GradientBoostingClassifier(random_state=42, {fmt(gbcp)})")

print(f"\n# Regression")
print(f"Ridge({fmt(ridp)})")
print(f"RandomForestRegressor(random_state=42, {fmt(rfrp)})")
print(f"GradientBoostingRegressor(random_state=42, {fmt(gbrp)})")
