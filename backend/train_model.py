"""
CinePredict — Training Pipeline
=================================
Trains regression + classification models on TMDB 5000 dataset.

Usage:
    python -m backend.train_model
    # or from project root:
    python backend/train_model.py
"""

import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                              accuracy_score, precision_score, recall_score,
                              f1_score, classification_report)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_RAW   = os.path.join(BASE_DIR, "data", "raw")
DATA_OUT   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MOVIES_CSV = os.path.join(DATA_RAW, "tmdb_5000_movies.csv")
CREDITS_CSV= os.path.join(DATA_RAW, "tmdb_5000_credits.csv")
USE_PYCARET= False
USE_XGBOOST= True
USE_LGBM   = True
SEED       = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)


def banner(msg):
    print(f"\n{'═'*58}\n  {msg}\n{'═'*58}")


def save(obj, fname):
    path = os.path.join(MODEL_DIR, fname)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  ✓ Saved  {fname}")
    return path


# ── 1. Load ──────────────────────────────────────────────────────────────────
def load_data():
    print("\n[1/7] Loading TMDB dataset...")
    if not os.path.exists(MOVIES_CSV):
        print(f"""
  ✗ Dataset not found: {MOVIES_CSV}

  Download from Kaggle:
    https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

  Place both CSV files in:
    backend/data/raw/tmdb_5000_movies.csv
    backend/data/raw/tmdb_5000_credits.csv
""")
        sys.exit(1)

    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV) if os.path.exists(CREDITS_CSV) else None
    print(f"  ✓ Movies : {len(movies):,}")
    if credits is not None:
        print(f"  ✓ Credits: {len(credits):,}")
    return movies, credits


# ── 2. Preprocess ────────────────────────────────────────────────────────────
def preprocess(movies, credits):
    print("\n[2/7] Preprocessing & feature engineering...")
    from backend.utils.feature_engineering import preprocess_movies
    df = preprocess_movies(movies, credits)
    print(f"  ✓ Clean movies: {len(df):,}  (removed zero-budget/revenue rows)")
    return df


# ── 3. NLP ───────────────────────────────────────────────────────────────────
def add_nlp(df):
    print("\n[3/7] Computing NLP features...")
    from backend.utils.nlp_utils import add_nlp_features
    df = add_nlp_features(df)
    sent = df["sentiment_score"]
    print(f"  ✓ Sentiment  min={sent.min():.3f}  mean={sent.mean():.3f}  max={sent.max():.3f}")
    return df


# ── 4. Split ─────────────────────────────────────────────────────────────────
def split_data(df):
    print("\n[4/7] Splitting dataset...")
    from backend.utils.feature_engineering import get_feature_columns
    feat_cols = get_feature_columns(df)
    print(f"  ✓ Features: {len(feat_cols)}")

    X   = df[feat_cols].copy()
    y_r = df["log_revenue"].values
    y_c = df["performance_label"].values

    le   = LabelEncoder()
    y_ce = le.fit_transform(y_c)
    print(f"  ✓ Classes: {list(le.classes_)}")

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_r, y_ce, test_size=0.20, random_state=SEED, stratify=y_ce
    )
    print(f"  ✓ Train={len(X_tr):,}  Test={len(X_te):,}")
    return X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, feat_cols, le


# ── 5. Train regression ──────────────────────────────────────────────────────
def train_regression(X_tr, X_te, yr_tr, yr_te):
    print("\n[5/7] Training regression models...")

    candidates = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=SEED),
        "RandomForest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        "Ridge":            Ridge(alpha=1.0),
    }
    if USE_XGBOOST:
        try:
            from xgboost import XGBRegressor
            candidates["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=0)
        except ImportError:
            print("  ⚠ XGBoost not installed")
    if USE_LGBM:
        try:
            from lightgbm import LGBMRegressor
            candidates["LightGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=SEED, verbose=-1)
        except ImportError:
            print("  ⚠ LightGBM not installed")

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_tr)
    Xte = imp.transform(X_te)

    best, best_r2, best_name = None, -np.inf, ""
    comparison = []
    for name, mdl in candidates.items():
        mdl.fit(Xtr, yr_tr)
        p  = mdl.predict(Xte)
        r2 = r2_score(yr_te, p)
        rm = np.sqrt(mean_squared_error(yr_te, p))
        ma = mean_absolute_error(yr_te, p)
        comparison.append({"Model": name, "R2": round(r2,4), "RMSE": round(rm,4), "MAE": round(ma,4)})
        print(f"    {name:<22}  R²={r2:.4f}  RMSE={rm:.4f}  MAE={ma:.4f}")
        if r2 > best_r2:
            best_r2, best, best_name = r2, mdl, name

    print(f"  ✓ Best: {best_name}  R²={best_r2:.4f}")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("model", best)])
    pipe.fit(X_tr, yr_tr)

    rev_pred = np.expm1(pipe.predict(X_te))
    rev_true = np.expm1(yr_te)
    metrics  = {
        "reg_r2":         float(r2_score(yr_te, pipe.predict(X_te))),
        "reg_rmse":       float(np.sqrt(mean_squared_error(rev_true, rev_pred))),
        "reg_mae":        float(mean_absolute_error(rev_true, rev_pred)),
        "reg_best_model": best_name,
        "model_comparison": comparison,
    }
    return pipe, metrics, best


# ── 6. Train classification ──────────────────────────────────────────────────
def train_classification(X_tr, X_te, yc_tr, yc_te, le):
    print("\n[6/7] Training classification models...")

    candidates = {
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=SEED),
        "RandomForest":       RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=SEED),
    }
    if USE_XGBOOST:
        try:
            from xgboost import XGBClassifier
            candidates["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=0, eval_metric="mlogloss")
        except ImportError:
            pass
    if USE_LGBM:
        try:
            from lightgbm import LGBMClassifier
            candidates["LightGBM"] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=SEED, verbose=-1)
        except ImportError:
            pass

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_tr)
    Xte = imp.transform(X_te)

    best, best_f1, best_name = None, -np.inf, ""
    for name, mdl in candidates.items():
        mdl.fit(Xtr, yc_tr)
        p   = mdl.predict(Xte)
        acc = accuracy_score(yc_te, p)
        f1  = f1_score(yc_te, p, average="weighted")
        print(f"    {name:<22}  Acc={acc:.4f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best, best_name = f1, mdl, name

    print(f"  ✓ Best: {best_name}  F1={best_f1:.4f}")
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("model", best)])
    pipe.fit(X_tr, yc_tr)
    preds = pipe.predict(X_te)

    print("\n  Classification Report:")
    print(classification_report(yc_te, preds, target_names=le.classes_))

    return pipe, {
        "clf_accuracy":   float(accuracy_score(yc_te, preds)),
        "clf_precision":  float(precision_score(yc_te, preds, average="weighted")),
        "clf_recall":     float(recall_score(yc_te, preds, average="weighted")),
        "clf_f1":         float(f1_score(yc_te, preds, average="weighted")),
        "clf_best_model": best_name,
    }, best


# ── 7. Feature importance ────────────────────────────────────────────────────
def feature_importance(model, feat_cols):
    try:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
        elif hasattr(model, "coef_"):
            fi = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
        else:
            return []
        return sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(feat_cols, fi)],
            key=lambda x: x["importance"], reverse=True
        )
    except Exception:
        return []


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    banner("CinePredict — Training Pipeline")

    movies, credits = load_data()
    df              = preprocess(movies, credits)
    df              = add_nlp(df)

    # Save processed data
    processed_path = os.path.join(DATA_OUT, "processed_movies.csv")
    df.to_csv(processed_path, index=False)
    print(f"\n  ✓ Saved processed_movies.csv  ({len(df):,} rows)")

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, feat_cols, le = split_data(df)

    reg_pipe, reg_m, raw_reg = train_regression(X_tr, X_te, yr_tr, yr_te)
    clf_pipe, clf_m, raw_clf = train_classification(X_tr, X_te, yc_tr, yc_te, le)

    print("\n[7/7] Saving artifacts...")
    save(reg_pipe,  "regression_model.pkl")
    save(clf_pipe,  "classification_model.pkl")
    save(feat_cols, "feature_columns.pkl")
    save(le,        "label_encoder.pkl")

    all_metrics = {**reg_m, **clf_m}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("  ✓ Saved  metrics.json")

    fi = feature_importance(raw_reg, feat_cols)
    if fi:
        with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
            json.dump(fi, f, indent=2)
        print("  ✓ Saved  feature_importance.json")

    banner("Training Complete")
    print(f"  Regression  R²  : {all_metrics['reg_r2']:.4f}")
    print(f"  Classifier  F1  : {all_metrics['clf_f1']:.4f}")
    print(f"  Classifier  Acc : {all_metrics['clf_accuracy']:.4f}")
    print(f"\n  Next step → uvicorn backend.api:app --reload\n")


if __name__ == "__main__":
    main()
